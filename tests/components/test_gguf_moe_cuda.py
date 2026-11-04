# SPDX-License-Identifier: Apache-2.0

"""GGUF experts through actual files, external kernels and compiled graph replay."""

import os

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession

from ignition.components.gguf_moe import GgufExperts, GgufMoeWorkspace
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.preparation.gguf import inspect_gguf_checkpoint


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


def raw_blocks(shape, qtype, seed):
    block, width = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (*shape[:-1], shape[-1] // block, width), dtype=np.uint8)
    for offset in (0, 2):
        scales = rng.uniform(-.005, .005, raw.shape[:-1]).astype('<f2')
        raw[..., offset:offset + 2] = scales[..., None].view(np.uint8)
    return raw


@pytest.fixture(scope='module')
def experts(tmp_path_factory):
    path = tmp_path_factory.mktemp('gguf-experts') / 'experts.gguf'
    q4, q5 = gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_1
    names = ('blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight')
    raw = (raw_blocks((512, 64, 256), q4, 41), raw_blocks((512, 64, 256), q4, 43),
           raw_blocks((512, 256, 64), q5, 47))
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    for name, value, kind in zip(names, raw, (q4, q4, q5), strict=True):
        writer.add_tensor(name, value.reshape(*value.shape[:2], -1), raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with GgufWeightLoader(inspect_gguf_checkpoint(path), 'cuda:0', read_mode='pread') as loader:
        weights = GgufExperts(loader.load_quantized_group(names[:2]), loader.load_quantized(names[2]))
    expanded = tuple(torch.from_numpy(gguf.dequantize(value.reshape(*value.shape[:2], -1), kind))
                     .bfloat16().cuda() for value, kind in zip(raw, (q4, q4, q5), strict=True))
    return weights, (torch.cat(expanded[:2], dim=1), expanded[2])


def reference(source, ids, probabilities, weights):
    up, down = weights
    result = []
    for start in range(0, len(source), 64):
        selected = ids[start:start + 64].long()
        valid = (selected >= 0) & (selected < len(up))
        selected = selected.clamp(0, len(up) - 1)
        x = source[start:start + 64, None, None, :].float()
        gate_up = (x @ up[selected].float().transpose(-1, -2)).squeeze(-2).bfloat16()
        gate, value = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * value
        projected = (intermediate.float().unsqueeze(-2) @ down[selected].float().transpose(-1, -2)).squeeze(-2).bfloat16()
        weighted = projected.float() * probabilities[start:start + 64, :, None]
        result.append(torch.where(valid[..., None], weighted, 0.).sum(dim=1).bfloat16())
    return torch.cat(result)


def assert_moe(output, expected):
    difference = output.float() - expected.float()
    # The two BF16 GEMMs and the BF16 SiLU/product boundaries follow the same
    # arithmetic contract as Flare's existing Qwen shared-expert checks.
    relative = difference.norm() / expected.float().norm().clamp_min(1e-12)
    assert relative.item() < .008
    torch.testing.assert_close(output, expected, rtol=.015, atol=expected.float().abs().max().item() * .01 + 1e-6)


class RoutedForward:
    def __init__(self, experts, router=None, logits=None):
        self.experts, self.router, self.logits = experts, router, logits

    def __call__(self):
        if self.router is not None:
            self.router(self.logits)
        self.experts()


@pytest.mark.parametrize('tokens,chunk_rows', [(1, 8), (17, 8), (4096, 64)])
def test_chunked_dynamic_experts_graph(experts, tokens, chunk_rows, tmp_path):
    packed, expanded = experts
    source = torch.randn((tokens, 256), dtype=torch.bfloat16, device='cuda') * .2
    output = torch.empty_like(source)
    ids = torch.zeros((tokens, 10), dtype=torch.int32, device='cuda')
    probabilities = torch.full((tokens, 10), .1, device='cuda')
    workspace = GgufMoeWorkspace.allocate(packed, chunk_rows=chunk_rows, top_k=10)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        prepared = PreparedRoutedExperts(source, packed, ids, probabilities, output,
            workspace=workspace, session=session, name='gguf-experts')
        forward = RoutedForward(prepared)
        from ignition.preparation.aot import ForwardCache
        from ignition.preparation.execution import GraphSignature

        cache = ForwardCache(namespace={'model': 'gguf-experts-test'}, device='cuda:0', root=tmp_path)
        compiled = cache.prepare_many({'moe': forward}, {'moe': GraphSignature('prefill', 1, tokens)})['moe']
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=stream):
            compiled()
        try:
            for iteration in range(2):
                source.normal_(std=.2)
                ids.random_(0, 512)
                ids[0, :3] = 511  # Duplicate routes must remain independent slots.
                ids[0, 3] = -1
                if iteration:
                    ids[0, 4] = 512  # The first chunk's error must survive later chunks.
                if tokens > 1:
                    ids[-1].fill_(-1)
                probabilities.copy_(torch.randn_like(probabilities).softmax(-1))
                output.fill_(float('nan'))
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert_moe(output, reference(source, ids, probabilities, expanded))
                if tokens > 1:
                    assert torch.count_nonzero(output[-1]).item() == 0
                assert prepared.errors[0].tolist() == [iteration, iteration]
                assert torch.count_nonzero(prepared.errors[1:]).item() == 0
            # Capacity changes never expand scratch to the complete prefill token count.
            assert workspace.gate_up.shape[0] == chunk_rows
            assert workspace.matmul.nbytes < (128 << 20)
        finally:
            stream.synchronize()
            graph.reset()


def test_flare_router_and_gguf_experts_share_one_graph(experts):
    packed, expanded = experts
    source = torch.randn((17, 256), dtype=torch.bfloat16, device='cuda') * .2
    output = torch.empty_like(source)
    logits = torch.zeros((17, 512), dtype=torch.bfloat16, device='cuda')
    valid = torch.tensor([17], dtype=torch.int32, device='cuda')
    routes = RouterScratch.allocate(17, source.device, experts=512, top_k=10)
    workspace = GgufMoeWorkspace.allocate(packed, chunk_rows=8, top_k=10)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        router = PreparedRouter(logits, routes, session=session, name='gguf-router', num_valid_tokens=valid)
        prepared = PreparedRoutedExperts(source, packed, routes.ids, routes.weights, output,
            workspace=workspace, session=session, name='gguf-routed')
        compiled = torch.compile(RoutedForward(prepared, router, logits), fullgraph=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=stream):
            compiled()
        try:
            for live in (17, 5, 0, 13):
                valid.fill_(live)
                logits.normal_()
                logits[0].zero_()
                graph.replay()
                torch.cuda.synchronize()
                chosen = 511 - logits.flip(-1).float().argsort(dim=-1, descending=True, stable=True)[:, :10]
                assert torch.equal(routes.ids[:live], chosen[:live].int())
                assert torch.equal(routes.ids[live:], torch.full_like(routes.ids[live:], -1))
                torch.testing.assert_close(routes.weights[:live], logits.float().gather(1, chosen)[:live].softmax(-1),
                                           rtol=2e-6, atol=1e-7)
                assert_moe(output, reference(source, routes.ids, routes.weights, expanded))
                assert torch.count_nonzero(output[live:]).item() == 0
                assert torch.count_nonzero(prepared.errors).item() == 0
        finally:
            stream.synchronize()
            graph.reset()
