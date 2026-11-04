# SPDX-License-Identifier: Apache-2.0

"""Exact bounded checkpoint transfers on hosts that prohibit io_uring."""

import os

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.checkpoint_source import BLOCK_BYTES


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {'bf16': torch.arange(96, dtype=torch.bfloat16).reshape(8, 12),
               'wide': torch.arange(2 * (BLOCK_BYTES + 13), dtype=torch.int64).to(torch.uint8).reshape(2, -1),
               'calibration.weight_scale_2': torch.tensor([.75]),
               'empty': torch.empty(0, dtype=torch.uint8)}
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    return inspect_indexed_checkpoint(tmp_path), tensors


def test_pread_bits_strides_expansion_and_multiple_chunks(checkpoint):
    manifest, tensors = checkpoint
    with DirectWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        expanded = loader.load_tensor('bf16', dtype=torch.float32, extent=(slice(None), slice(1, 8)))
        torch.testing.assert_close(expanded.cpu(), tensors['bf16'][:, 1:8].float(), rtol=0, atol=0)
        wide = loader.load_tensor('wide')
        assert torch.equal(wide.cpu(), tensors['wide'])
        assert loader.load_tensor('calibration.weight_scale_2').item() == .75
        assert loader.load_tensor('empty').numel() == 0
        destination = torch.full((8, 12), -17, dtype=torch.bfloat16, device='cuda')
        loader._copy(destination[:, :4], loader._sources['bf16'][:, 2:6])
        loader._copy(destination[:, 4:8], loader._sources['bf16'][:, 8:12])
        loader._session.flush()
        assert torch.equal(destination[:, :4].cpu(), tensors['bf16'][:, 2:6])
        assert torch.equal(destination[:, 4:8].cpu(), tensors['bf16'][:, 8:12])
        assert bool((destination[:, 8:] == -17).all())
        assert loader._session.bf16_expansion_bytes == expanded.numel() * 2
        assert loader._session.stats()['pinned_staging_bytes'] == 2 * BLOCK_BYTES
        assert loader._session.torch_copy_bytes == 0


def test_pread_rejects_overlapping_destination(checkpoint):
    manifest, _ = checkpoint
    with DirectWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        destination = torch.empty((8, 4), dtype=torch.bfloat16, device='cuda')
        loader._copy(destination, loader._sources['bf16'][:, :4])
        loader._copy(destination, loader._sources['bf16'][:, 4:8])
        with pytest.raises(ValueError, match='overlapping checkpoint destinations'):
            loader._session.flush()


def test_pread_rejects_changed_source(checkpoint):
    manifest, _ = checkpoint
    with DirectWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        path = manifest.snapshot / manifest.shards[0].path
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with pytest.raises(RuntimeError, match='source changed'):
            loader.load_tensor('bf16')
