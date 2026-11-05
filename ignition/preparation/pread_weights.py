# SPDX-License-Identifier: Apache-2.0

"""Explicit bounded checkpoint transport for hosts without io_uring."""

import ctypes
import os

from b12x.loader._checkpoint import DirectWeightSession

from .checkpoint_source import BLOCK_BYTES
from .weight_ranges import chunks, destinations_overlap


class PreadWeightSession(DirectWeightSession):
    """Reuse native O_DIRECT reads and CUDA copies during preparation only.

    This mode is selected explicitly; an io_uring or GDS error never changes
    transports silently. Routed source validation and destination ownership
    remain in DirectWeightSession. Host staging is bounded independently of
    model size and is released when loading ends.
    """

    def __init__(self, **kwargs):
        mode = kwargs.pop('read_mode', 'pread')
        super().__init__(read_mode='bounce' if mode == 'pread' else mode, **kwargs)
        self.read_mode = mode if mode == 'pread' else self.read_mode
        self.staging = None

    def manifest(self, path):
        if self.read_mode != 'pread':
            return super().manifest(path)
        stat = os.stat(path)
        expected = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        entries = super().manifest(path)
        self.file_identities[self.files[-1]] = expected
        self._validate_source_identities()
        return entries

    def _metadata_values(self, entries):
        self._validate_source_identities()
        values = super()._metadata_values(entries)
        self._validate_source_identities()
        return values

    def _execute(self, records):
        self._validate_source_identities()
        if self.read_mode != 'pread':
            super()._execute(records)
            self._validate_source_identities()
            return
        import torch
        from cuda.bindings import runtime as cuda

        if len(records) % 8:
            raise ValueError('invalid checkpoint transfer descriptor')
        destinations = []
        for offset in range(0, len(records), 8):
            _, _, width, pointer, transform, rows, _, stride = records[offset:offset + 8]
            if transform not in (0, 1, 2) or rows < 1 or (transform == 1 and width % 2):
                raise ValueError('invalid checkpoint transfer geometry')
            extent = width * (1 + (transform == 1))
            if rows > 1 and stride < extent:
                raise ValueError('overlapping checkpoint destination rows')
            destinations.append((pointer, extent, rows, stride))
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
            stream = torch.cuda.current_stream(self.device)
            stream.synchronize()
            if self.staging is None:
                self.staging = torch.empty(2 * BLOCK_BYTES, dtype=torch.uint8, pin_memory=True)
            host = memoryview(self.staging.numpy()).cast('B')
            for index in range(0, len(records), 8):
                fd, offset, width, pointer, transform, rows, source_stride, stride = records[index:index + 8]
                for start, take, count, pitch, destination, destination_pitch in chunks(
                        offset, width, rows, source_stride, pointer, stride, int(transform == 1)):
                    size = take + (count - 1) * pitch
                    content = (ctypes.string_at(start, size) if transform == 2 else
                               self.native.direct_bytes(self.reader, fd, start, size))
                    host[:size] = content
                    copy_width, copy_pitch = take, pitch if count > 1 else take
                    if transform == 1:
                        source = self.staging[:size].view(torch.bfloat16).as_strided(
                            (count, take // 2), (copy_pitch // 2, 1))
                        # Conversion is CPU preparation work; retain source bytes
                        # before writing the expanded result into the same staging.
                        expanded = source.float()
                        copy_width = take * 2
                        copy_pitch = copy_width
                        self.staging[:count * copy_width].view(torch.float32).view_as(expanded).copy_(expanded)
                    status, = cuda.cudaMemcpy2DAsync(destination, destination_pitch if count > 1 else copy_width,
                        self.staging.data_ptr(), copy_pitch, copy_width, count,
                        cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream.cuda_stream)
                    stream.synchronize()
                    if int(status):
                        raise RuntimeError(f'checkpoint CUDA copy failed: {status}')
        self._validate_source_identities()

    def stats(self, **kwargs):
        return dict(super().stats(**kwargs), pinned_staging_bytes=0 if self.staging is None else self.staging.nbytes)

    def _close(self):
        self.staging = None
        super()._close()
