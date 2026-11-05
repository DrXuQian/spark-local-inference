# SPDX-License-Identifier: Apache-2.0

"""Real cache/snapshot geometry without importing execution operator libraries."""

import pytest
import torch

from ignition.components.state_layout import StateLayout
from ignition.recipes.qwen38.attention_state import QSAMetadata, QSAState
from ignition.recipes.qwen38.cache_layout import cache_requirements
from ignition.recipes.qwen38.config import PreparationConfig
from ignition.recipes.qwen38.recipe import RECIPE
from ignition.recipes.qwen38.state_layout import HybridStateLayout


@pytest.mark.parametrize('pagesize', [64, 128])
@pytest.mark.parametrize('heads', [1, 2])
@pytest.mark.parametrize('drafts,ring', [(0,4), (1,8), (4,8), (5,16), (7,16)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_cache_storage_and_rollback_horizon(pagesize, heads, drafts, ring, dtype):
    geometry = cache_requirements(main_page_size=pagesize, kv_heads=heads,
                                  max_speculative_tokens=drafts, kv_dtype=dtype)
    state = QSAState.allocate(pages=3, slots=2, device='meta', page_size=pagesize,
                             draft_tokens=drafts, kv_heads=heads, kv_dtype=dtype)
    assert state.main_k_cache.shape == state.main_v_cache.shape == (3,pagesize,heads,256)
    assert state.main_k_cache.dtype == state.main_v_cache.dtype == dtype
    assert state.compressed_k_cache.shape == (3,pagesize//4,128)
    assert state.compressed_k_cache.dtype == torch.bfloat16
    assert state.raw_k_ring.shape == (2,ring,128)
    assert state.raw_logical_positions.shape == (2,ring)
    assert state.raw_rope_positions.shape == (2,ring,3)
    assert state.raw_interval_start_positions.shape == (2,)
    assert geometry.raw_ring_capacity == ring and geometry.selection_width == 2051
    assert geometry.page_nbytes == sum(getattr(state,name).nbytes for name in
                                      ('main_k_cache','main_v_cache','compressed_k_cache'))//3
    assert geometry.page_nbytes == pagesize*(512*heads*dtype.itemsize+64)


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.parametrize('drafts', [0, 7])
def test_snapshot_offsets_preserve_previous_package_sizes(world_size, drafts):
    fp8 = HybridStateLayout.build(world_size=world_size, speculative_tokens=drafts)
    reference = (RECIPE.ranks if drafts else RECIPE.plain_ranks)[world_size]
    assert fp8.row_bytes == reference.state_row_bytes
    bf16 = HybridStateLayout.build(world_size=world_size, speculative_tokens=drafts, kv_dtype=torch.bfloat16)
    heads = max(1,2//world_size)
    assert bf16.row_bytes == fp8.row_bytes+(12*2+(4 if drafts else 0))*64*heads*256
    # 72 GDN + 5 shared + 84 QSA regions, plus 12 MTP continuation regions.
    assert len(fp8.regions) == (173 if drafts else 161)
    for layout in (fp8,bf16):
        for previous,current in zip(layout.regions,layout.regions[1:]):
            assert previous.offset+previous.nbytes <= current.offset
            assert current.offset%256 == 0
        assert layout.regions[-1].offset+layout.regions[-1].nbytes <= layout.row_bytes
        assert layout.region('gdn.0.recurrent').shape == (48//world_size,128,128)
        assert layout.region('gdn.0.convolution').shape == (10240//world_size,3+drafts)
        assert layout.region('ple_convolution').shape == (10240,9+drafts)
        assert layout.snapshot_capacity(4<<30) == (4<<30)//layout.row_bytes
    for old in fp8.regions:
        new = bf16.region(old.name)
        assert new.shape == old.shape
        boundary = 'boundary_key' in old.name or 'boundary_value' in old.name
        assert new.dtype == (torch.bfloat16 if boundary else old.dtype)
        assert new.nbytes == old.nbytes*(2 if boundary else 1)


def test_cache_initialization_and_resize_keep_selector_history():
    state = QSAState.allocate(pages=1,slots=2,device='cpu',draft_tokens=7,kv_dtype=torch.bfloat16)
    assert not state.main_k_cache.count_nonzero()
    assert torch.all(state.raw_logical_positions == -1) and torch.all(state.raw_rope_positions == -1)
    state.raw_k_ring[1].fill_(3)
    grown = state.resize_pages_for_preparation(3)
    assert grown.raw_k_ring is state.raw_k_ring and torch.all(grown.raw_k_ring[1] == 3)
    assert grown.main_k_cache.shape[0] == 3 and not grown.main_k_cache.count_nonzero()
    assert grown.main_k_cache.dtype == torch.bfloat16
    metadata = QSAMetadata.allocate(tokens=8,batch=2,context=128,device='cpu')
    assert metadata.main_block_table.shape == metadata.compressed_block_table.shape == (2,2)
    assert torch.all(metadata.main_block_table == -1)
    assert not metadata.sequence_lengths.count_nonzero()


@pytest.mark.parametrize('options', [dict(main_page_size=65),dict(kv_heads=0),dict(position_axes=2),
    dict(max_speculative_tokens=-1),dict(max_speculative_tokens=True),dict(max_speculative_tokens=64)])
def test_invalid_geometry_is_rejected(options):
    with pytest.raises(ValueError):
        cache_requirements(**(dict(main_page_size=64)|options))
    with pytest.raises(TypeError):
        cache_requirements(main_page_size=64,kv_dtype=torch.float32)


def test_config_validates_capacities_without_kernel_imports():
    config = PreparationConfig(mtp=True,max_active=1,prefill_chunk_size=64)
    assert config.completion_token_capacity == 6
    assert config.token_capacity >= 64 and config.preparation_state_rows >= 3
    with pytest.raises(ValueError,match='snapshot budget'):
        PreparationConfig(snapshot_budget_bytes=1)
    layout = StateLayout.from_specs([('a',(3,),torch.bfloat16),('b',(5,),torch.int64)])
    assert layout.region('a').offset == 0 and layout.region('b').offset == 256 and layout.row_bytes == 512
