# SPDX-License-Identifier: Apache-2.0

"""Actual shared-Q8 feedback/full-attention weights through Flare forward compilation."""

import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession

from ignition.preparation.aot import ForwardCache
from ignition.preparation.checkpoint import CheckpointManifest
from ignition.preparation.execution import GraphSignature
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.preparation.qwen38 import CONFIG_SHA256
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.attention import PreparedAttention, QSAMetadata, QSAScratch, QSAState, RotaryTables
from ignition.recipes.qwen38.mtp import PreparedMTPFeedback
from ignition.recipes.qwen38.mtp_weights import MTPWeights


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


@pytest.fixture(scope='module')
def checkpoint():
    path, frontend = os.environ.get('FLARE_GGUF_MTP'), os.environ.get('FLARE_GGUF_FRONTEND')
    if not path or not frontend:
        pytest.skip('actual MTP GGUF and pinned frontend assets required')
    structure = CheckpointManifest(Path(frontend), {}, (), {'config.json': CONFIG_SHA256}, '', {})
    weights = MTPWeights.load_gguf(inspect_gguf_checkpoint(Path(path)), structure, 'cuda:0',
        embedding_weight=None, lm_head_weight=None, read_mode='pread')
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def expand(name):
        tensor = tensors['blk.48.' + name]
        return torch.from_numpy(np.array(gguf.dequantize(tensor.data, tensor.tensor_type))).cuda()

    return weights, expand


class FeedbackForward:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self):
        self.operator()


class AttentionForward:
    def __init__(self, operator, source):
        self.operator, self.source = operator, source

    def __call__(self):
        self.operator(self.source)


def rms(source, gain):
    value = source.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * gain).bfloat16()


def test_shared_q8_feedback(checkpoint, tmp_path):
    weights, expand = checkpoint
    hidden = torch.randn((7, 10240), device='cuda', dtype=torch.bfloat16) * .2
    embedding = torch.randn((7, 2560), device='cuda', dtype=torch.bfloat16) * .2
    output = torch.empty_like(hidden)
    arena = ScratchArena('cuda:0')
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=0) as session:
        op = PreparedMTPFeedback(weights.module, embedding, hidden, output, arena, session=session, name='gguf.feedback')
        arena.finalize(); arena.freeze()
        cache = ForwardCache(namespace={'model': 'gguf-mtp-feedback'}, device='cuda:0', root=tmp_path)
        forward = cache.prepare_many({'feedback': FeedbackForward(op)},
            {'feedback': GraphSignature('prefill', 1, 7)})['feedback']
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            eh = expand('nextn.eh_proj.weight').bfloat16()
            for _ in range(2):
                hidden.normal_(std=.2); embedding.normal_(std=.2)
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                e = rms(embedding, expand('nextn.enorm.weight'))
                h = rms(hidden, expand('nextn.hnorm.weight')).view(7, 4, 2560)
                expected = F.linear(e, eh[:, :2560])[:, None] + F.linear(h, eh[:, 2560:])
                torch.testing.assert_close(output.view_as(expected), expected, rtol=2e-2, atol=4e-2)
        finally:
            torch.cuda.synchronize(); graph.reset()


@pytest.mark.parametrize('tokens', [1, 7, 128])
@pytest.mark.parametrize('kv_dtype', [torch.float8_e4m3fn, torch.bfloat16], ids=['fp8', 'bf16'])
def test_shared_q8_full_attention(checkpoint, tokens, tmp_path, kv_dtype):
    weights, expand = checkpoint
    module = weights.module.layers[0].self_attn
    assert module.compress_ratio == 0
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device='cuda')
    scratch = QSAScratch.allocate(tokens, 'cuda:0')
    state = QSAState.allocate(pages=8, slots=2, device='cuda:0', draft_tokens=7, kv_dtype=kv_dtype)
    metadata = QSAMetadata.allocate(tokens=tokens, batch=1, context=512, device='cuda:0')
    rotary = RotaryTables.allocate(module.config, 512, 'cuda:0')
    arena = ScratchArena('cuda:0')
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=0) as session:
        op = PreparedAttention(module, weights.attention, source, scratch, state, metadata, rotary,
            session=session, name='gguf.full', workspace=arena, projection_workspace=arena, max_cache_pages=16)
        arena.finalize(); arena.freeze()
        # Rebinding final page storage must work before compilation/capture.
        state = state.resize_pages_for_preparation(16)
        op.rebind_pages_for_preparation(state)
        cache = ForwardCache(namespace={'model': 'gguf-mtp-full'}, device='cuda:0', root=tmp_path)
        forward = cache.prepare_many({'attention': AttentionForward(op, source)},
            {'attention': GraphSignature('prefill', 1, tokens)})['attention']
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            projections = [expand(f'attn_{name}.weight').bfloat16() for name in ('q', 'k', 'v')]
            out_weight = expand('attn_output.weight').bfloat16()
            qgain, kgain = expand('attn_q_norm.weight'), expand('attn_k_norm.weight')
            from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

            for live in (tokens, max(1, tokens // 2), 0):
                source.normal_(std=.2)
                history = 63
                state.main_k_cache.copy_(torch.randn(state.main_k_cache.shape, device='cuda') * .2)
                state.main_v_cache.copy_(torch.randn(state.main_v_cache.shape, device='cuda') * .2)
                key_before = state.main_k_cache.clone()
                value_before = state.main_v_cache.clone()
                metadata.main_block_table.copy_(torch.arange(8, device='cuda', dtype=torch.int32)[None])
                metadata.query_start_loc.copy_(torch.tensor([0, live], device='cuda', dtype=torch.int32))
                metadata.sequence_lengths.fill_(history + live if live else 0)
                metadata.request_ids.fill_(-1); metadata.request_ids[:live] = 0
                positions = torch.arange(tokens, device='cuda') + history
                metadata.rope_positions.copy_(positions[:, None].expand(-1, 3))
                metadata.query_positions.copy_(positions)
                metadata.slot_mapping.fill_(-1); metadata.slot_mapping[:live].copy_(positions[:live])
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert torch.count_nonzero(scratch.output[live:]).item() == 0
                if not live:
                    assert torch.equal(state.main_k_cache.view(torch.uint8), key_before.view(torch.uint8))
                    assert torch.equal(state.main_v_cache.view(torch.uint8), value_before.view(torch.uint8))
                    continue
                q = F.linear(source[:live], projections[0]).view(live, 24, 512)
                k = F.linear(source[:live], projections[1]).view(live, 2, 256)
                v = F.linear(source[:live], projections[2]).view(live, 2, 256)
                cos, sin = rotary.gather(metadata.rope_positions[:live])
                query, key = apply_rotary_pos_emb(rms(q[..., :256], qgain), rms(k, kgain), cos, sin, unsqueeze_dim=1)
                all_k = torch.cat((key_before.flatten(0, 1)[:history].float(), key.to(kv_dtype).float()))
                all_v = torch.cat((value_before.flatten(0, 1)[:history].float(), v.to(kv_dtype).float()))
                logits = torch.einsum('thd,shd->hts', query.float(), all_k.repeat_interleave(12, dim=1)) / 16
                mask = torch.arange(history + live, device='cuda')[None] > positions[:live, None]
                logits.masked_fill_(mask[None], float('-inf'))
                attended = torch.einsum('hts,shd->thd', logits.softmax(-1), all_v.repeat_interleave(12, dim=1)).bfloat16()
                expected = F.linear((attended * q[..., 256:].sigmoid()).flatten(1), out_weight)
                torch.testing.assert_close(scratch.output[:live], expected, rtol=.02, atol=.02)
        finally:
            torch.cuda.synchronize(); graph.reset()
