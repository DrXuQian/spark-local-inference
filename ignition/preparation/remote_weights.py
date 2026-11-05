# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt b12x's exact routed reads to an owned network checkpoint source."""

from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os

from b12x.loader._checkpoint import DirectWeightSession, TensorRange

from .checkpoint_source import BLOCK_BYTES, row_extent, tensor_dtype
from .weight_ranges import chunks, destinations_overlap


class RemoteWeightSession(DirectWeightSession):
    """Keep b12x routing and native copies while replacing only checkpoint reads.

    Source IDs identify authenticated remote files, never local descriptors. Only
    received bounded buffers have a memory-backed descriptor, private to this session.
    """

    def __init__(self, source, manifest, *, device):
        super().__init__(device=device, io_threads=1, read_mode="bounce")
        self.source, self.checkpoint = source, manifest
        self.names = tuple(source.name(manifest.snapshot / shard.path) for shard in manifest.shards)
        self.ids = {name: index for index, name in enumerate(self.names)}
        self.shards = {name: shard.path for name, shard in zip(self.names, manifest.shards, strict=True)}
        self.buffer_write = os.memfd_create("flare-checkpoint-read", os.MFD_CLOEXEC)
        try:
            # Keep b12x's bounded bounce reader without writing every network block to NVMe.
            self.buffer_read = os.open(f"/proc/self/fd/{self.buffer_write}", os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
        except BaseException:
            os.close(self.buffer_write)
            super()._close()
            raise

    def manifest(self, path):
        name = self.source.name(path)
        identifier = self.ids[name]
        return [TensorRange(tensor.name, identifier, tensor.offset, tensor.nbytes,
                            tensor.shape, tensor_dtype(tensor.dtype))
                for tensor in self.checkpoint.tensors.values() if tensor.shard == self.shards[name]]

    def _metadata_values(self, entries):
        import math
        import torch

        values = {}
        def requests():
            for entry in entries:
                self.metadata_bytes += entry.nbytes
                if self.metadata_bytes > 64 << 20:
                    raise ValueError("declared metadata spans exceed the 64 MiB session budget")
                if not entry.nbytes:
                    yield None, (entry, True)
                for offset in range(0, entry.nbytes, BLOCK_BYTES):
                    count = min(BLOCK_BYTES, entry.nbytes - offset)
                    yield ((self.names[entry.fd], entry.offset + offset, count),
                           (entry, offset + count == entry.nbytes))
        content = bytearray()
        with closing(self._prefetch(requests())) as received:
            for (entry, last), block in received:
                if block is not None:
                    content.extend(block)
                if last:
                    values[entry.name] = (torch.frombuffer(content, dtype=entry.dtype, count=math.prod(entry.shape))
                                          .reshape(entry.shape) if entry.nbytes
                                          else torch.empty(entry.shape, dtype=entry.dtype))
                    content = bytearray()
        return values

    def _execute(self, records):
        if len(records) % 8:
            raise ValueError("invalid b12x checkpoint descriptor size")
        destinations = []
        for index in range(0, len(records), 8):
            identifier, offset, width, pointer, transform, rows, source_stride, stride = records[index:index + 8]
            if transform not in (0, 1, 2) or rows < 1:
                raise ValueError("unsupported b12x checkpoint transform")
            if width:
                written = width * (1 + (transform == 1))
                if rows > 1 and stride < written:
                    raise ValueError("routed checkpoint destinations overlap")
                destinations.append((pointer, written, rows, stride))
            if transform != 2:
                if identifier >= len(self.names):
                    raise ValueError("unknown remote checkpoint source ID")
                size = self.source.files[self.names[identifier]]["bytes"]
                if offset + (rows - 1) * source_stride + width > size:
                    raise ValueError("routed checkpoint view exceeds its source file")
        destinations.sort()
        for index, previous in enumerate(destinations):
            end = previous[0] + previous[1] + (previous[2] - 1) * previous[3]
            for following in destinations[index + 1:]:
                if following[0] >= end:
                    break
                if destinations_overlap(previous, following):
                    raise ValueError("overlapping checkpoint destinations require an explicit dependency")
        def requests():
            for index in range(0, len(records), 8):
                identifier, offset, width, pointer, transform, rows, source_stride, destination_stride = records[index:index + 8]
                if transform == 2:
                    # These records name owned CPU scalar storage, not a checkpoint source ID.
                    yield None, array("Q", records[index:index + 8])
                    continue
                name = self.names[identifier]
                for start, count, batch, stride, target, target_stride in chunks(
                        offset, width, rows, source_stride, pointer, destination_stride, transform):
                    row_extent(self.source.files[name]["bytes"], start, count, batch, stride)
                    yield ((name, start, count, batch, stride),
                           array("Q", (self.buffer_read, 0, count, target, transform, batch, count, target_stride)))

        with closing(self._prefetch(requests())) as received:
            for record, content in received:
                if content is None:
                    super()._execute(record)
                    continue
                os.ftruncate(self.buffer_write, len(content))
                written = 0
                while written < len(content):
                    size = os.pwrite(self.buffer_write, content[written:], written)
                    if size <= 0:
                        raise OSError("checkpoint transfer buffer write did not advance")
                    written += size
                # The synchronous native reader owns all destination and buffer access until it returns.
                super()._execute(record)

    def _prefetch(self, requests):
        """Hold at most one bounded read per channel; native destination writes stay serial."""
        workers = self.source.streams
        if workers == 1:
            for request, record in requests:
                yield record, None if request is None else self.source.read(*request)
            return
        pending = deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flare-checkpoint-read") as executor:
            try:
                exhausted = False
                while pending or not exhausted:
                    while len(pending) < workers and not exhausted:
                        item = next(requests, None)
                        if item is None:
                            exhausted = True
                        else:
                            request, record = item
                            pending.append((record, None if request is None else executor.submit(self.source.read, *request)))
                    if pending:
                        record, future = pending[0]
                        content = None if future is None else future.result()
                        yield record, content
                        # Retain the slot through the synchronous GPU copy before replacing its buffer.
                        pending.popleft()
                        del content, future
            finally:
                if pending:
                    self.source.abort_reads()
                    for _, future in pending:
                        if future is not None:
                            future.cancel()

    def stats(self, *, flush=True):
        result = super().stats(flush=flush)
        result["remote_source"] = self.source.report()
        return result

    def _close(self):
        try:
            super()._close()
        finally:
            os.close(self.buffer_read)
            os.close(self.buffer_write)
