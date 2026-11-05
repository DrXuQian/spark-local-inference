# SPDX-License-Identifier: Apache-2.0

"""Actual GGUF file -> prepared helpers -> one graph; no b12x/cuTile import."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')
pytest.importorskip('quactlize.serving')

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gguf_weights import GgufWeight
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.preparation.quactlize import QuactlizeHelpersSpec


@pytest.fixture(scope='module')
def session():
    path = os.environ.get('FLARE_QUACTLIZE_HELPERS_LIBRARY')
    if not path or not torch.cuda.is_available():
        pytest.skip('assign FLARE_QUACTLIZE_HELPERS_LIBRARY and an actual GPU/PPU')
    helpers = QuactlizeHelpersSpec.read(path).load('cuda:0')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield SimpleNamespace(quactlize_helpers=helpers, stream=stream)
        stream.synchronize()
        helpers.close()


def gguf_table(tmp_path, qtype, width):
    kind = gguf.GGMLQuantizationType(qtype)
    block, size = gguf.GGML_QUANT_SIZES[kind]
    raw = np.arange(7 * (width // block) * size, dtype=np.int64).astype(np.uint8).reshape(7, width // block, size)
    raw[..., :2] = np.array([.03125], dtype='<f2').view(np.uint8)
    path = tmp_path / (kind.name + '.gguf')
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('token_embd.weight', raw.reshape(7, -1), raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    table = gguf.GGUFReader(path).tensors[0]
    assert tuple(reversed(table.shape)) == (7, width) and table.tensor_type == kind
    expected = torch.from_numpy(gguf.dequantize(np.array(table.data).reshape(7, -1), kind)).bfloat16()
    return GgufWeight(torch.from_numpy(np.array(table.data)).cuda(), kind.name, qtype, (7, width)), expected


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('qtype', [8, 20])
def test_gguf_dense_preparation_uses_selected_native_decoder(session, tmp_path, monkeypatch, dtype, qtype):
    import sys
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.preparation.gguf import inspect_gguf_checkpoint

    gguf_table(tmp_path, qtype, 640)
    path = tmp_path / (gguf.GGMLQuantizationType(qtype).name + '.gguf')
    raw = np.array(gguf.GGUFReader(path).tensors[0].data)
    expected = torch.from_numpy(gguf.dequantize(raw, gguf.GGMLQuantizationType(qtype))).to(dtype)
    # Loading must use the selected external library, even if no debug decoder exists.
    monkeypatch.setitem(sys.modules, 'quactlize.cuda_compat', None)
    with torch.cuda.stream(torch.cuda.default_stream()), GgufWeightLoader(inspect_gguf_checkpoint(path), 'cuda:0', read_mode='pread',
            quactlize_helpers=session.quactlize_helpers) as loader:
        actual = loader.load_dequantized('token_embd.weight', rows=(1, 6), columns=(32, 608), dtype=dtype)
    torch.testing.assert_close(actual.cpu(), expected[1:6, 32:608], rtol=0, atol=0)


@pytest.mark.parametrize('tokens', [1, 2, 4, 8, 17])
@pytest.mark.parametrize('qtype,width', [(8, 2560), (20, 160)])
def test_file_lookup_routing_and_rounded_activation_share_one_graph(session, tmp_path, tokens, qtype, width):
    weight, table = gguf_table(tmp_path, qtype, width)
    ids = torch.zeros(tokens, dtype=torch.int64, device='cuda')
    count = torch.zeros(1, dtype=torch.int32, device='cuda')
    output = torch.empty(tokens, width, dtype=torch.bfloat16, device='cuda')
    lookup = PreparedEmbedding(weight, ids, output, num_rows=count, session=session, name='file.rows')
    logits = torch.zeros(tokens, 512, dtype=torch.bfloat16, device='cuda')
    next_logits = torch.zeros_like(logits)
    scratch = RouterScratch.allocate(tokens, 'cuda', experts=512, top_k=10)
    route = PreparedRouter(logits, scratch, num_valid_tokens=count, session=session, name='route')
    gate = torch.linspace(-5, 5, tokens * 640).view(tokens, 640).bfloat16()
    up = torch.full_like(gate, 1.5703125)
    gate_up = torch.cat((gate, up), dim=-1).cuda()
    activation = torch.empty(tokens, 640, dtype=torch.bfloat16, device='cuda')
    normalized = (gate.float() * torch.sigmoid(gate.float())).bfloat16()
    expected_activation = (normalized.float() * up.float()).bfloat16()

    def forward():
        lookup.device_error.zero_()
        route.error.zero_()
        lookup()
        route(logits)
        # The first call's error must survive a later successful MTP-style call.
        route(next_logits)
        session.quactlize_helpers.swiglu(gate_up, activation, round_silu=True)

    forward(); session.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=session.stream):
        forward()
    try:
        for live, bad_id, bad_logit in ((tokens, False, False), (0, False, False),
                (tokens, True, True), (tokens // 2, False, False), (tokens, False, False)):
            indices = torch.arange(tokens, dtype=torch.int64) % 7
            indices[0] = (1 << 40) if bad_id else -1
            ids.copy_(indices); count.fill_(live)
            logits.zero_()
            if bad_logit:
                logits[0, 0] = float('nan')
            if live < tokens:
                logits[live:] = float('nan')
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay(); session.stream.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            assert lookup.device_error.item() == int(bad_id and live > 0)
            assert route.error.item() == (2 if bad_logit and live > 0 else 0)
            valid = (indices >= 0) & (indices < 7) & (torch.arange(tokens) < live)
            expected = torch.zeros(tokens, width, dtype=torch.bfloat16)
            expected[valid] = table[indices[valid]]
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            selected = torch.arange(511, 501, -1, dtype=torch.int32).repeat(live, 1)
            assert torch.equal(scratch.ids[:live].cpu(), selected)
            assert torch.all(scratch.ids[live:] == -1) and torch.all(scratch.weights[live:] == 0)
            torch.testing.assert_close(scratch.weights[:live].cpu(), torch.full((live, 10), .1), rtol=1e-6, atol=0)
            # exp implementations may straddle one BF16 rounding midpoint.
            ulp = (torch.nextafter(expected_activation, torch.full_like(expected_activation, float('inf')))
                   .float() - expected_activation.float()).abs()
            assert torch.all((activation.cpu().float() - expected_activation.float()).abs() <= ulp)
    finally:
        session.stream.synchronize()
        graph.reset()
