# SPDX-License-Identifier: Apache-2.0

"""Explicit QUactlize matmul and serving-helper library inputs."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from .gguf import file_identity


class CudaMatmulLibrary:
    """NVIDIA debug implementation of QUactlize's native matmul v1 records."""

    def __init__(self, path):
        import ctypes as C
        from quactlize.compat.native import Problem, Sizes, Buffers

        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        for name, args, result in (
            ('query', [C.POINTER(Problem), C.c_void_p, C.POINTER(Sizes)], C.c_int),
            ('prepare', [C.POINTER(Problem), C.c_void_p, C.POINTER(Buffers), C.POINTER(C.c_void_p)], C.c_int),
            ('run', [C.c_void_p, C.POINTER(Buffers)], C.c_int),
            ('destroy', [C.c_void_p], None), ('error', [], C.c_char_p)):
            function = getattr(self.lib, 'quactlize_cuda_matmul_' + name + '_v1')
            function.argtypes, function.restype = args, result
            setattr(self, name, function)

    def check(self, result):
        if result:
            raise RuntimeError(f'QUactlize CUDA debug matmul rc={result}: {self.error().decode(errors="replace")}')


@dataclass(frozen=True)
class QuactlizeLibrarySpec:
    path: Path
    sha256: str
    target: str = 'nvidia'
    package_sha256: str | None = None

    def __post_init__(self):
        if self.target not in ('nvidia', 'ppu10', 'ppu15'):
            raise ValueError('QUactlize target must be nvidia, ppu10 or ppu15')

    @classmethod
    def read(cls, path, *, target='nvidia'):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('QUactlize library changed during input inspection')
        return cls(path, digest, target)

    @classmethod
    def installed(cls, *, target):
        if target not in ('ppu10', 'ppu15'):
            raise ValueError('installed QUactlize discovery requires an explicit PPU target')
        try:
            from quactlize import get_library_info
        except ImportError as exc:
            raise ValueError('install the complete QUactlize PPU wheel (0.2.0 or compatible newer version)') from exc
        info = get_library_info(target=target)
        # The package verifies its Python bindings and complete native resource
        # closure without importing Torch or initializing a device context.
        return replace(cls.read(info['library_path'], target=target), package_sha256=info['manifest_sha256'])

    def validate(self):
        current = (self.installed(target=self.target) if self.package_sha256 is not None
                   else self.read(self.path, target=self.target))
        if current != self:
            raise ValueError('QUactlize library or package changed after input inspection')

    def identity(self):
        result = dict(abi='qpm_v1', target=self.target, library=self.sha256)
        if self.target != 'nvidia':
            result.update(expert_layout='canonical_kpack_v1', q5_1_execution='packed_auto_v1')
        if self.package_sha256 is not None:
            result['package_manifest'] = self.package_sha256
        return result

    def load(self, device):
        self.validate()
        import torch

        if torch.device(device).type != 'cuda':
            raise ValueError('QUactlize requires a CUDA-compatible device')
        if self.target != 'nvidia':
            from quactlize.compat.native import Library
            from .gguf_packing import GgufPacker

            library = Library(self.path)
            library.packed = Library(self.path, packed_route=0)
            library.packing = GgufPacker(library)
            return library
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured QUactlize matmul record implementation requires an SM120 device')
        return CudaMatmulLibrary(self.path)


@dataclass(frozen=True)
class QuactlizeHelpersSpec(QuactlizeLibrarySpec):
    def identity(self):
        result = dict(router='v2', swiglu='v2', gather_rows='v1', library=self.sha256)
        if self.target != 'nvidia':
            result['target'] = self.target
        if self.package_sha256 is not None:
            result['package_manifest'] = self.package_sha256
        return result

    def load(self, device):
        self.validate()
        import torch
        from quactlize.serving import Serving
        from ignition.components.quactlize_helpers import PreparedQuactlizeHelpers

        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('QUactlize helpers require the CUDA-compatible device API')
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('prepare QUactlize helpers before capture')
            index = torch.cuda.current_device()
            stream = torch.cuda.Stream(device=index)
            # prepare warms all variants and synchronizes this explicit stream.
            return PreparedQuactlizeHelpers(Serving(self.path, device=index, stream=stream))


def resolve_quactlize(matmul_path=None, helpers_path=None, *, target):
    """Explicit paths win; a PPU selection otherwise uses the installed wheel."""
    default = None
    specs = []
    for path, cls in ((matmul_path, QuactlizeLibrarySpec), (helpers_path, QuactlizeHelpersSpec)):
        if path is not None:
            spec = cls.read(path, target=target)
        elif target in ('ppu10', 'ppu15'):
            if default is None:
                default = QuactlizeLibrarySpec.installed(target=target)
            spec = cls(default.path, default.sha256, default.target, default.package_sha256)
        else:
            spec = None
        specs.append(spec)
    return tuple(specs)
