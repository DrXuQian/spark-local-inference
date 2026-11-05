# SPDX-License-Identifier: Apache-2.0

"""Explicit QUactlize matmul and serving-helper library inputs."""

from dataclasses import dataclass
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

    def identity(self):
        return dict(abi='qpm_v1', target=self.target, library=self.sha256)

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('QUactlize library changed after input inspection')
        import torch

        if torch.device(device).type != 'cuda':
            raise ValueError('QUactlize requires a CUDA-compatible device')
        if self.target != 'nvidia':
            from quactlize.compat.native import Library

            return Library(self.path)
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('the configured QUactlize matmul record implementation requires an SM120 device')
        return CudaMatmulLibrary(self.path)


@dataclass(frozen=True)
class QuactlizeHelpersSpec(QuactlizeLibrarySpec):
    def identity(self):
        result = dict(router='v2', swiglu='v2', gather_rows='v1', library=self.sha256)
        if self.target != 'nvidia':
            result['target'] = self.target
        return result

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('QUactlize helpers library changed after input inspection')
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
