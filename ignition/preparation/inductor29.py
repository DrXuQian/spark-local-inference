# SPDX-License-Identifier: Apache-2.0

"""Torch 2.9 standalone compiler artifacts for fixed-buffer graph qualification.

This preserves compilation and cache loading; it does not provide the newer
Dynamo guard serialization required by the complete model preparation catalog.
"""

import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class StandaloneArtifact:
    def __init__(self, artifact, serialized=None):
        self.artifact = artifact
        self.serialized = serialized

    @classmethod
    def compile(cls, module, inputs, options):
        from torch._inductor import standalone_compile

        if torch.__version__.split('+')[0].split('.')[:2] != ['2', '9']:
            raise RuntimeError('this standalone artifact adapter requires platform Torch 2.9')
        return cls(standalone_compile(module, inputs, dynamic_shapes='from_example_inputs',
                                     options={'config_patches': dict(options)}))

    @classmethod
    def load(cls, data):
        from torch._inductor.standalone_compile import CompiledArtifact

        with TemporaryDirectory(prefix='flare-torch29-') as directory:
            path = Path(directory) / 'forward.bin'
            path.write_bytes(data)
            artifact = CompiledArtifact.load(path=str(path))
        return cls(artifact, zlib.compress(data, level=1))

    def serialize(self):
        if self.serialized is None:
            with TemporaryDirectory(prefix='flare-torch29-') as directory:
                path = Path(directory) / 'forward.bin'
                self.artifact.save(path=str(path))
                self.serialized = zlib.compress(path.read_bytes(), level=1)
            # The callable retains live kernels; keep only one compressed copy
            # of the portable cache payload after publication.
            self.artifact._artifacts = None
        return zlib.decompress(self.serialized)

    def __call__(self, *inputs):
        return self.artifact(*inputs)
