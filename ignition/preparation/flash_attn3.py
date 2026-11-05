# SPDX-License-Identifier: Apache-2.0

"""Bind the installed FA3 extension by content before preparing paged attention."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .gguf import file_identity


@dataclass(frozen=True)
class FlashAttention3Spec:
    path: Path
    sha256: str

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if file_identity(path) != before:
            raise ValueError('FA3 library changed during input inspection')
        return cls(path, digest)

    def identity(self):
        return {'operator': 'flash_attn_3.fwd', 'library': self.sha256, 'kv_dtype': 'bfloat16'}

    def load(self, device):
        if self.read(self.path).sha256 != self.sha256:
            raise ValueError('FA3 library changed after input inspection')
        import importlib

        import torch

        if torch.device(device).type != 'cuda':
            raise ValueError('FA3 requires a CUDA-compatible device')
        library = importlib.import_module('flash_attn_3._C')
        if Path(library.__file__).resolve() != self.path:
            raise ValueError('installed flash_attn_3._C differs from the selected FA3 library')
        fields = {argument.name for argument in torch.ops.flash_attn_3.fwd.default._schema.arguments}
        required = {'q', 'k', 'v', 'out', 'cu_seqlens_q', 'seqused_k', 'max_seqlen_q',
                    'page_table', 'is_causal', 'num_splits'}
        if not required <= fields:
            raise ValueError(f'FA3 paged forward ABI is missing {sorted(required - fields)}')
        return library
