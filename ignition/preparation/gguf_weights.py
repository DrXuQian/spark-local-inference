# SPDX-License-Identifier: Apache-2.0

"""Route inspected GGUF byte ranges through b12x's existing native reader."""

import os
from pathlib import Path

from b12x.loader._checkpoint import TensorRange

from .checkpoint import CheckpointError
from .checkpoint_source import tensor_dtype
from .gguf import GgufCheckpoint
from .pread_weights import PreadWeightSession


class GgufWeightSession(PreadWeightSession):
    """Preserve packed bytes and reuse b12x's bounded transfers and view routing.

    Quantized source tensors expose their physical block dimensions, never a
    fictitious one-byte-per-weight logical shape. Plain tensors retain their
    native dtype, including BF16. Model selection controls which ranges are read;
    opening a shard does not transfer its PLE table or unrelated expert weights.
    """

    def __init__(self, checkpoint: GgufCheckpoint, **kwargs):
        checkpoint.validate_files()
        self.checkpoint = checkpoint
        self.shards = {checkpoint.snapshot / shard.path: shard for shard in checkpoint.shards}
        super().__init__(**kwargs)

    def manifest(self, path):
        path = Path(path).resolve()
        shard = self.shards.get(path)
        if shard is None:
            raise CheckpointError(f"{path}: shard is outside the inspected GGUF checkpoint")
        fd = (self.native.direct_open(os.fspath(path)) if os.name == "nt" else
              os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC))
        try:
            if self._file_identity(fd) != shard.identity:
                raise CheckpointError(f"{shard.path}: GGUF changed after inspection")
            entries = [TensorRange(tensor.name, fd, tensor.offset, tensor.nbytes,
                                   tensor.storage_shape, tensor_dtype(tensor.dtype))
                       for tensor in self.checkpoint.tensors.values() if tensor.shard == shard.path]
        except BaseException:
            os.close(fd)
            raise
        self.files.append(fd)
        self.file_identities[fd] = shard.identity
        return entries
