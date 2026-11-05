# SPDX-License-Identifier: Apache-2.0

"""Content-bound cuBLAS/acBLAS selection for GGUF dense projections."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class BlasLibrarySpec:
    path: Path
    sha256: str
    api: str
    workspace_bytes: int

    def __post_init__(self):
        if self.api not in ('cublas', 'acblas'):
            raise ValueError('dense_blas api must be cublas or acblas')
        if (type(self.workspace_bytes) is not int or not 0 < self.workspace_bytes < 1 << 63
                or self.workspace_bytes % 256):
            raise ValueError('dense_blas workspace_bytes must be a positive int64 multiple of 256')

    @classmethod
    def read(cls, directory, value):
        if (not isinstance(value, dict) or set(value) != {'library', 'api', 'workspace_bytes'}
                or not isinstance(value['library'], str) or not value['library']):
            raise ValueError('dense_blas requires library, api and workspace_bytes')
        path = (Path(directory) / value['library']).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('BLAS library changed during input inspection')
        return cls(path, digest, value['api'], value['workspace_bytes'])

    def identity(self):
        return dict(api=self.api, library=self.sha256, workspace_bytes=self.workspace_bytes,
                    compute='fp32', reduced_precision_reduction=False)

    def load(self):
        current = self.read(self.path.parent, dict(library=self.path.name, api=self.api,
                                                  workspace_bytes=self.workspace_bytes))
        if current != self:
            raise ValueError('BLAS library changed after input inspection')
        from ignition.components.blas_projection import BlasLibrary

        return BlasLibrary(self.path, api=self.api)
