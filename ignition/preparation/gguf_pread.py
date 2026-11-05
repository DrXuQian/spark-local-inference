# SPDX-License-Identifier: Apache-2.0

"""Local GGUF range reads using POSIX and the platform CUDA copy API."""

import ctypes
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path

from .checkpoint import CheckpointError
from .checkpoint_source import BLOCK_BYTES, tensor_dtype
from .weight_ranges import chunks, destinations_overlap


ALIGNMENT = 4096
# Failed device drains must not release memory that a queued copy can reference.
_undrained = []


def _identity(fd):
    stat = os.fstat(fd)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_direct(fd, offset, size, staging):
    """Read a bounded, possibly unaligned range into aligned caller-owned memory."""
    if offset < 0 or not 0 < size <= BLOCK_BYTES:
        raise ValueError('invalid GGUF read range')
    start = offset // ALIGNMENT * ALIGNMENT
    skip = offset - start
    length = (skip + size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    if len(staging) < length or ctypes.addressof(ctypes.c_char.from_buffer(staging)) % ALIGNMENT:
        raise ValueError('GGUF direct reads require aligned bounded staging')
    received = 0
    while received < skip + size:
        count = os.preadv(fd, [staging[received:length]], start + received)
        received += count
        if received < skip + size and (not count or received % ALIGNMENT):
            raise RuntimeError('short GGUF checkpoint read')
    return staging[skip:skip + size], received


@dataclass(frozen=True)
class _Source:
    tensor: object
    fd: int
    offset: int
    nbytes: int
    version: int


def routed_rows(source, destination, *, offset, nbytes):
    """Map ordinary tensor views to exact row copies, including per-expert joins."""
    import torch

    if source.shape != destination.shape or source.is_neg() or source.is_conj() or destination.is_neg() or destination.is_conj():
        raise ValueError('GGUF transfer requires equal shapes and untransformed views')
    expand = source.dtype == torch.bfloat16 and destination.dtype == torch.float32
    byte_cast = {source.dtype, destination.dtype} <= {torch.uint8, torch.int8}
    if source.dtype != destination.dtype and not expand and not byte_cast:
        raise ValueError('GGUF transfer cannot change stored precision')
    if not source.numel():
        return ()
    for tensor in (source, destination):
        extent = 1
        for stride, count in sorted((stride, count) for count, stride in zip(tensor.shape, tensor.stride()) if count > 1):
            if stride < extent:
                raise ValueError('overlapping GGUF tensor views')
            extent += (count - 1) * stride
        if (tensor.storage_offset() + extent) * tensor.element_size() > tensor.untyped_storage().nbytes():
            raise ValueError('GGUF view exceeds its storage')
    first = source.storage_offset() * source.element_size()
    extent = source.element_size() * (1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride())))
    if first < 0 or first + extent > nbytes:
        raise ValueError('GGUF source view exceeds its inspected tensor range')
    dimensions = [(n, ss, ds) for n, ss, ds in zip(source.shape, source.stride(), destination.stride()) if n > 1]
    elements = 1
    while dimensions and dimensions[-1][1:] == (elements, elements):
        elements *= dimensions.pop()[0]
    rows, ss, ds = dimensions.pop() if dimensions else (1, elements, elements)
    records = []
    for coordinates in product(*(range(n) for n, _, _ in dimensions)):
        source_row = sum(i * d[1] for i, d in zip(coordinates, dimensions))
        destination_row = sum(i * d[2] for i, d in zip(coordinates, dimensions))
        records.append((offset + first + source_row * source.element_size(), elements * source.element_size(),
            rows, ss * source.element_size(), destination.data_ptr() + destination_row * destination.element_size(),
            ds * destination.element_size(), int(expand)))
    return tuple(records)


class GgufPreadSession:
    """Own metadata views, bounded pinned staging, and destination lifetimes.

    GGUF pread is explicit. Other checkpoint transports keep their existing
    readers; unsupported O_DIRECT or CUDA APIs never silently switch transport.
    All transfers finish during loading, before compilation or capture.
    """

    def __init__(self, checkpoint, *, device):
        if not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv'):
            raise ValueError('GGUF pread requires POSIX O_DIRECT and preadv')
        checkpoint.validate_files()
        self.checkpoint, self.device = checkpoint, device
        self.files, self.sources, self.pending = {}, {}, []
        self.staging = self.expanded = self.stream = None
        self.opened = self.retained = False
        self.read_mode = 'pread'
        self.payload_bytes = self.physical_bytes = self.reads = self.loaded_tensors = 0
        self.bf16_expansion_bytes = self.torch_copy_bytes = 0

    def __enter__(self):
        if self.opened:
            raise RuntimeError('GGUF read session is already open')
        self.opened = True
        return self

    def _validate(self):
        if not self.opened or self.retained:
            raise RuntimeError('GGUF read session is not available')
        for fd, expected in self.files.values():
            if _identity(fd) != expected:
                raise RuntimeError('GGUF source changed after inspection')
        for source in self.sources.values():
            if source.tensor._version != source.version:
                raise ValueError('GGUF source view was modified')

    def weights(self, paths, *, index_path=None, skip=lambda name: False, needs_values=lambda entry: False):
        import torch

        self._validate()
        if index_path is not None:
            raise ValueError('GGUF uses its inspected shard headers, not a separate index')
        shards = {self.checkpoint.snapshot / shard.path: shard for shard in self.checkpoint.shards}
        for path in paths:
            path = Path(path).resolve()
            if path not in shards or path in self.files:
                raise CheckpointError('GGUF shard is unknown or already open')
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
            self.files[path] = fd, shards[path].identity
            self._validate()
            for entry in self.checkpoint.tensors.values():
                if entry.shard != shards[path].path or skip(entry.name):
                    continue
                if needs_values(entry):
                    raise ValueError('GGUF metadata views cannot evaluate tensor values on the host')
                with torch.inference_mode(False):
                    tensor = torch.empty(entry.storage_shape, dtype=tensor_dtype(entry.dtype), device='meta')
                if tensor.nbytes != entry.nbytes:
                    raise CheckpointError('GGUF storage shape differs from its inspected byte range')
                self.sources[tensor.untyped_storage()._cdata] = _Source(tensor, fd, entry.offset, entry.nbytes, tensor._version)
                yield entry.name, tensor
        self._validate()

    def _routes(self, destination, source):
        origin = self.sources.get(source.untyped_storage()._cdata)
        if origin is None or source.device.type != 'meta' or source.dtype != origin.tensor.dtype:
            raise ValueError('GGUF source is not an unchanged inspected tensor view')
        records = routed_rows(source, destination, offset=origin.offset, nbytes=origin.nbytes)
        return origin.fd, records

    def __call__(self, destination, source):
        import torch

        self._validate()
        if not destination.is_cuda or destination.device.index != self.device:
            raise ValueError('GGUF destination must belong to the selected CUDA device')
        fd, records = self._routes(destination, source)
        self.pending.append((destination, source, source._version, fd, records))
        torch.autograd.graph.increment_version(destination)
        self.payload_bytes += source.nbytes
        self.bf16_expansion_bytes += source.nbytes * (source.dtype == torch.bfloat16 and destination.dtype == torch.float32)
        self.loaded_tensors += 1
        return True

    def _drain(self):
        try:
            self.stream.synchronize()
        except BaseException:
            if not self.retained:
                self.retained = True
                _undrained.append(self)
            raise

    def flush(self):
        if not self.pending:
            return
        self._validate()
        import torch
        from cuda.bindings import runtime as cuda

        destinations = []
        for destination, source, version, fd, records in self.pending:
            if source._version != version or self._routes(destination, source) != (fd, records):
                raise ValueError('queued GGUF tensor view was modified')
            destinations.extend((pointer, width * (1 + expand), rows, stride)
                for _, width, rows, _, pointer, stride, expand in records)
        destinations.sort()
        for index, previous in enumerate(destinations):
            end = previous[0] + previous[1] + (previous[2] - 1) * previous[3]
            for following in destinations[index + 1:]:
                if following[0] >= end:
                    break
                if destinations_overlap(previous, following):
                    raise ValueError('overlapping checkpoint destinations need an explicit dependency')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('checkpoint reads must finish before graph capture')
            self.stream = torch.cuda.current_stream(self.device)
            self._drain()
            if self.staging is None:
                self.staging = torch.empty(BLOCK_BYTES + 2 * ALIGNMENT, dtype=torch.uint8, pin_memory=True)
            padding = -self.staging.data_ptr() % ALIGNMENT
            host = memoryview(self.staging.numpy())[padding:]
            for _, _, _, fd, records in self.pending:
                for record in records:
                    for start, take, count, pitch, pointer, stride in chunks(*record):
                        size = take + (count - 1) * pitch
                        content, physical = read_direct(fd, start, size, host)
                        self.physical_bytes += physical
                        self.reads += 1
                        width, source_pitch = take, pitch if count > 1 else take
                        source_pointer = ctypes.addressof(ctypes.c_char.from_buffer(content))
                        if record[-1]:
                            if self.expanded is None:
                                self.expanded = torch.empty(2 * BLOCK_BYTES, dtype=torch.uint8, pin_memory=True)
                            values = torch.frombuffer(content, dtype=torch.bfloat16).as_strided(
                                (count, take // 2), (source_pitch // 2, 1))
                            self.expanded[:count * take * 2].view(torch.float32).view_as(values).copy_(values)
                            source_pointer = self.expanded.data_ptr()
                            width = source_pitch = take * 2
                        try:
                            status, = cuda.cudaMemcpy2DAsync(pointer, stride if count > 1 else width,
                                source_pointer, source_pitch, width, count,
                                cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream.cuda_stream)
                            if int(status):
                                raise RuntimeError(f'checkpoint CUDA copy failed: {status}')
                        finally:
                            self._drain()
            self._validate()
            self.pending.clear()

    def stats(self, *, flush=True):
        if flush:
            self.flush()
        return dict(read_mode=self.read_mode, payload_bytes=self.payload_bytes, physical_bytes=self.physical_bytes,
            reads=self.reads, loaded_tensors=self.loaded_tensors, bf16_expansion_bytes=self.bf16_expansion_bytes,
            torch_copy_bytes=0, metadata_bytes=0, transform_bytes=0, metadata_host_copy_bytes=0,
            pinned_staging_bytes=sum(0 if value is None else value.nbytes for value in (self.staging, self.expanded)))

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.flush()
        finally:
            if not self.retained:
                for fd, _ in self.files.values():
                    os.close(fd)
                self.files.clear()
                self.sources.clear()
                self.pending.clear()
                self.staging = self.expanded = self.stream = None
                self.opened = False
