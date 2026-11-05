# SPDX-License-Identifier: Apache-2.0

"""Explicit, content-bound actlizeLA target and preparation library."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class ActlizeLibrarySpec:
    path: Path
    sha256: str
    target: str = 'cuda_sm120'

    def __post_init__(self):
        if self.target not in ('cuda_sm120', 'ppu10', 'ppu15'):
            raise ValueError('actlizeLA target must be cuda_sm120, ppu10 or ppu15')

    @classmethod
    def read(cls, path, *, target='cuda_sm120'):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('actlizeLA library changed during input inspection')
        return cls(path, digest, target)

    def identity(self):
        return dict(gdn='actlize', qsa='actlize', target=self.target,
                    library=self.sha256, kv_dtype='bfloat16')

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('actlizeLA library changed after input inspection')
        import torch

        if torch.device(device).type != 'cuda':
            raise ValueError('actlizeLA requires a CUDA-compatible device')
        if self.target == 'cuda_sm120' and torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured actlizeLA GGUF recipe requires an SM120 device')
        from actlize_la.serving import NativeLibrary

        return NativeLibrary(self.path, target=self.target)
