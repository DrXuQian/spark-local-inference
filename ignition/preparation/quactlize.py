# SPDX-License-Identifier: Apache-2.0

"""Explicit NVIDIA implementation of the QUactlize PPU matmul records."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class QuactlizeLibrarySpec:
    path: Path
    sha256: str

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('QUactlize library changed during input inspection')
        return cls(path, digest)

    def identity(self):
        return dict(abi='qpm_v1', target='nvidia', library=self.sha256)

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('QUactlize library changed after input inspection')
        import torch

        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured QUactlize matmul record implementation requires an SM120 device')
        from quactlize.cuda_compat import CudaPpuMatmulLibrary

        return CudaPpuMatmulLibrary(self.path)
