# SPDX-License-Identifier: Apache-2.0

"""Explicit, content-bound actlizeLA target and preparation library."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class ActlizeLibrarySpec:
    path: Path
    sha256: str
    target: str = 'cuda_sm120'
    package_sha256: str | None = None

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

    @classmethod
    def installed(cls, *, target):
        if target not in ('ppu10', 'ppu15'):
            raise ValueError('installed actlizeLA discovery requires an explicit PPU target')
        try:
            from actlize_la import get_library_info
        except ImportError as exc:
            raise ValueError('install the complete actlize-la serving wheel (0.2.0 or compatible newer version)') from exc
        info = get_library_info(target=target)
        # Discovery verifies all bindings and native resources without loading
        # a DSO or initializing a device. Include that closure in cache identity.
        return replace(cls.read(info['library_path'], target=target), package_sha256=info['manifest_sha256'])

    def validate(self):
        current = (self.installed(target=self.target) if self.package_sha256 is not None
                   else self.read(self.path, target=self.target))
        if current != self:
            raise ValueError('actlizeLA library or package changed after input inspection')

    def identity(self):
        result = dict(gdn='actlize', qsa='actlize', target=self.target,
                      library=self.sha256, kv_dtype='bfloat16')
        if self.package_sha256 is not None:
            result['package_manifest'] = self.package_sha256
        return result

    def load(self, device):
        self.validate()
        import torch

        if torch.device(device).type != 'cuda':
            raise ValueError('actlizeLA requires a CUDA-compatible device')
        if self.target == 'cuda_sm120' and torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured actlizeLA GGUF recipe requires an SM120 device')
        from actlize_la.serving import NativeLibrary

        return NativeLibrary(self.path, target=self.target)


def resolve_actlize(path=None, *, target):
    """Explicit paths win; a PPU selection otherwise uses the installed wheel."""
    if path is not None:
        return ActlizeLibrarySpec.read(path, target=target)
    if target in ('ppu10', 'ppu15'):
        return ActlizeLibrarySpec.installed(target=target)
    return None
