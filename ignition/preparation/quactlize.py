# SPDX-License-Identifier: Apache-2.0

"""Explicit QUactlize matmul and serving-helper library inputs."""

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


@dataclass(frozen=True)
class QuactlizeHelpersSpec(QuactlizeLibrarySpec):
    def identity(self):
        return dict(router='v2', swiglu='v2', gather_rows='v1', library=self.sha256)

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
