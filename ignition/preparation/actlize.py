# SPDX-License-Identifier: Apache-2.0

"""Explicit, content-bound actlizeLA preparation inputs for the SM120 GGUF recipe."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class ActlizeLibrarySpec:
    path: Path
    sha256: str

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('actlizeLA library changed during input inspection')
        return cls(path, digest)

    def identity(self):
        return dict(gdn='actlize', qsa='actlize', target='cuda_sm120',
                    library=self.sha256, kv_dtype='bfloat16')

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('actlizeLA library changed after input inspection')
        import torch

        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured actlizeLA GGUF recipe requires an SM120 device')
        from actlize_la.serving import NativeLibrary

        return NativeLibrary(self.path, target='cuda_sm120')
