# SPDX-License-Identifier: Apache-2.0

"""Actual GGUF QSA projections composed with the external BF16 cache transaction."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from b12x.attention.qsa.reference import (
    prepare_index_reference, score_select_reference, sparse_paged_gqa_reference, stream_compress_reference,
)
from b12x.preparation import PreparationSession

from ignition.preparation.aot import ForwardCache
from ignition.preparation.execution import GraphSignature
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.attention import PreparedAttention, QSAMetadata, QSAScratch, QSAState, RotaryTables
from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader
from qualify.qwen38_gguf import component_checkpoint
from tests.recipes.test_qwen38_gguf_mtp_cuda import AttentionForward


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


@pytest.fixture(scope='module')
def checkpoint():
    names = ('FLARE_GGUF_TARGET_SHARD', 'FLARE_GGUF_METADATA', 'FLARE_GGUF_FRONTEND', 'FLARE_ACTLIZE_LIBRARY')
    if any(not os.environ.get(name) for name in names):
        pytest.skip('actual target shard, metadata, frontend and SM120 serving library required')
    from actlize_la.serving import NativeLibrary
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

    shard, headers, frontend, library = (Path(os.environ[name]) for name in names)
    manifest = component_checkpoint(shard, headers, (47,))
    config = Qwen4ExpTextConfig.from_dict(json.loads((frontend / 'config.json').read_text())['text_config'])
    with torch.device('meta'):
        module = Qwen4ExpTextAttention(config, layer_idx=47)
    with QwenGgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        weights = loader.load_attention(module, 47)
    return module.eval().requires_grad_(False), weights, NativeLibrary(library, target='cuda_sm120')


@pytest.mark.parametrize('tokens', [1, 6, 64])
def test_actual_qsa_compiled_projection_cache_and_selection(checkpoint, tokens, tmp_path):
    module, weights, library = checkpoint
    device = torch.device('cuda:0')
    context, pages = 2176, 34
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
    scratch = QSAScratch.allocate(tokens, device)
    state = QSAState.allocate(pages=pages, slots=3, device=device, draft_tokens=7, kv_dtype=torch.bfloat16)
    metadata = QSAMetadata.allocate(tokens=tokens, batch=1, context=context, device=device)
    rotary = RotaryTables.allocate(module.config, context + 32, device)
    live = torch.zeros(1, dtype=torch.int32, device=device)
    arena = ScratchArena(device)
    cosine, sine, axes = rotary.cos.cpu(), rotary.sin.cpu(), rotary.axes.cpu()
    qweight, kweight = module.indexer.q_layernorm.weight.cpu(), module.indexer.k_layernorm.weight.cpu()

    def rope(value, positions):
        column = torch.arange(32)
        co, si = cosine[positions[..., axes], column], sine[positions[..., axes], column]
        while co.ndim < value.ndim:
            co, si = co.unsqueeze(-2), si.unsqueeze(-2)
        value = value.float(); lo, hi = value[..., :32], value[..., 32:64]
        return torch.cat((lo * co - hi * si, hi * co + lo * si, value[..., 64:]), -1)

    def compressed(raw, positions):
        _, result = stream_compress_reference(raw, torch.arange(raw.shape[0]), positions,
            torch.zeros(16, 128, dtype=torch.bfloat16), torch.full((16,), -1, dtype=torch.int64),
            torch.full((16, 3), -1, dtype=torch.int64), 4, kweight, module.config.rms_norm_eps, rope)
        return result.bfloat16()

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = PreparedAttention(module, weights, source, scratch, state, metadata, rotary,
            session=session, name='actual.gguf.qsa', workspace=arena, projection_workspace=arena,
            max_cache_pages=40, draft_tokens=7, actlize_library=library, live_rows=live)
        arena.finalize(); arena.freeze()
        state = state.resize_pages_for_preparation(40)
        operation.rebind_pages_for_preparation(state)
        forward = ForwardCache(namespace={'model': 'actual-gguf-actlize-qsa'}, device=device,
            root=tmp_path).prepare_many({'qsa': AttentionForward(operation, source)},
            {'qsa': GraphSignature('prefill', 1, tokens)})['qsa']
        forward(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            permutation, compressed_pages = torch.arange(pages).roll(3), torch.arange(pages).roll(7)
            for history, count in ((0, tokens), (63, tokens), (2051, tokens), (63, 0)):
                for name in state.__dataclass_fields__:
                    value = getattr(state, name)
                    value.fill_(-1 if value.dtype == torch.int64 else 0)
                source.normal_(std=.2)
                positions = torch.arange(history + count)[:, None].expand(-1, 3).clone()
                positions[:, 1] += 1; positions[:, 2] += 2
                raw = (torch.randn(history, 128) * .2).bfloat16()
                if history >= 2048:
                    raw.zero_()  # Preserve the reference's stable selection of tied historical groups.
                for name in ('main_k_cache', 'main_v_cache'):
                    cache = getattr(state, name)
                    for t in range(history):
                        cache[permutation[t // 64], t % 64].normal_(std=.2)
                for group, value in enumerate(compressed(raw, positions[:history])):
                    state.compressed_k_cache[compressed_pages[group // 16], group % 16].copy_(value)
                for t in range(max(0, history - 16), history):
                    state.raw_k_ring[1, t % 16].copy_(raw[t])
                    state.raw_logical_positions[1, t % 16] = t
                    state.raw_rope_positions[1, t % 16].copy_(positions[t])
                state.raw_interval_start_positions[1] = history - 1
                metadata.main_block_table[0].copy_(permutation)
                metadata.compressed_block_table[0].copy_(compressed_pages)
                metadata.raw_state_slot_ids.fill_(1 if count else -1)
                metadata.request_ids.fill_(-1); metadata.request_ids[:count].fill_(0)
                metadata.query_positions.fill_(-1)
                metadata.query_positions[:count].copy_(torch.arange(history, history + count))
                metadata.rope_positions.zero_(); metadata.rope_positions[:count].copy_(positions[history:])
                metadata.slot_mapping.fill_(-1)
                logical = torch.arange(history, history + count)
                metadata.slot_mapping[:count].copy_(permutation[logical // 64] * 64 + logical % 64)
                metadata.sequence_lengths.fill_(history + count if count else 0)
                metadata.query_start_loc[1].fill_(count)
                metadata.num_accepted_tokens.fill_(1); metadata.is_prefilling.fill_(tokens > 8)
                live.fill_(1 if count else 0)
                before = {name: getattr(state, name).cpu().clone() for name in state.__dataclass_fields__}
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert all(not error.any().item() for error in operation.device_errors)
                if not count:
                    assert not scratch.output.count_nonzero().item()
                    for name, value in before.items():
                        assert torch.equal(getattr(state, name).cpu(), value)
                    continue
                for key, value in (('qkv_proj', scratch.projected_qkv),
                                   ('indexer.index_qk_proj', scratch.projected_index)):
                    torch.testing.assert_close(value, F.linear(source, weights[key]), rtol=.02, atol=.02)
                for t in range(count):
                    page, offset = int(permutation[(history + t) // 64]), (history + t) % 64
                    before['main_k_cache'][page, offset].copy_(scratch.key[t].cpu())
                    before['main_v_cache'][page, offset].copy_(scratch.projected_v[t].view(2, 256).cpu())
                for name in ('main_k_cache', 'main_v_cache'):
                    assert torch.equal(getattr(state, name).cpu(), before[name])
                raw = torch.cat((raw, scratch.projected_index[:, 512:].cpu()))
                compressed_keys = compressed(raw, positions)
                for group, expected in enumerate(compressed_keys):
                    actual = state.compressed_k_cache[compressed_pages[group // 16], group % 16].cpu()
                    torch.testing.assert_close(actual, expected, rtol=2**-6, atol=2e-3)
                encoded, _ = prepare_index_reference(scratch.projected_index[:, :512].view(tokens, 4, 128).cpu(),
                    scratch.projected_index[:, 512:].cpu(), qweight, module.config.rms_norm_eps, rope, positions[history:])
                _, selected = score_select_reference(encoded, compressed_keys, logical, history + count, 4, 2048)
                assert torch.equal(scratch.selected_positions.cpu(), selected)
                expected = sparse_paged_gqa_reference(scratch.query.cpu(), before['main_k_cache'], before['main_v_cache'],
                    metadata.main_block_table.cpu(), metadata.request_ids.cpu(), selected, metadata.query_positions.cpu())
                actual = scratch.attended.cpu().float()
                relative = torch.linalg.vector_norm(actual - expected.float()) / torch.linalg.vector_norm(expected.float())
                assert relative < .006
                torch.testing.assert_close(actual, expected.float(), rtol=.01, atol=2e-5)
                gate = scratch.projected_q.view(tokens, 24, 512)[..., 256:]
                output = F.linear((expected.to(device) * gate.sigmoid()).flatten(1), weights['o_proj'])
                torch.testing.assert_close(scratch.output, output, rtol=.02, atol=.02)
                for name in ('raw_k_ring', 'raw_logical_positions', 'raw_rope_positions', 'raw_interval_start_positions'):
                    assert torch.equal(getattr(state, name)[0].cpu(), before[name][0])
                    assert torch.equal(getattr(state, name)[2].cpu(), before[name][2])
        finally:
            torch.cuda.synchronize(); graph.reset()
