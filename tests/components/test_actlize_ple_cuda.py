# SPDX-License-Identifier: Apache-2.0

"""Small IQ4_NL file -> external PLE graph; staged rows, no native I/O latency claim."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')
serving = pytest.importorskip('actlize_la.serving')
pytest.importorskip('quactlize.serving')

from ignition.components.blas_projection import BlasLibrary, PreparedBlas
from ignition.preparation.quactlize import QuactlizeHelpersSpec
from ignition.recipes.qwen38.gguf_ple import GgufPLECaps, GgufPLEGeometry, GgufPLELayout, GgufPLETable
from ignition.recipes.qwen38.ple import PLEMetadata, PLEScratch, PreparedPLE, allocate_convolution_state


@pytest.fixture(scope='module')
def session():
    names = ('FLARE_ACTLIZE_LIBRARY', 'FLARE_ACTLIZE_TARGET', 'FLARE_QUACTLIZE_HELPERS_LIBRARY', 'FLARE_TEST_BLAS_LIBRARY')
    path, target, helpers, blas = (os.environ.get(name) for name in names)
    if not all((path, target, helpers, blas)) or not torch.cuda.is_available():
        pytest.skip('assign actual actlizeLA, QUactlize and BLAS libraries with platform Torch/device')
    owner = QuactlizeHelpersSpec.read(helpers).load('cuda:0')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dense = PreparedBlas(BlasLibrary(blas, api=os.environ.get('FLARE_TEST_BLAS_API', 'acblas')),
            torch.empty(4 << 20, dtype=torch.uint8, device='cuda:0'))
        yield SimpleNamespace(actlize_helpers=serving.NativeLibrary(path, target=target),
            quactlize_helpers=owner, dense_blas=dense, stream=stream)
        stream.synchronize()
        owner.close()


def norm(x, delta):
    x = x.float().view(x.shape[0], 4, 2560)
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            * (1. + delta.view(4, 2560))).bfloat16().flatten(1)


def within_ulp(actual, expected):
    def ordered(value):
        bits = value.bfloat16().view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)
    actual = actual.cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert (ordered(actual) - ordered(expected)).abs().max() <= 1


def hashes(tokens, starts, history, slots, geometry):
    # Semantic CPU oracle: newest-first EOS-bounded windows per packed request.
    result = torch.full((len(tokens), 16), -1, dtype=torch.int64)
    requests = torch.full((len(tokens),), -1, dtype=torch.int64)
    for row, (begin, end) in enumerate(zip(starts, starts[1:])):
        if slots[row] < 0:
            continue
        prefix = list(history[row])
        for index in range(begin, end):
            prefix.append(tokens[index])
            previous = prefix[:-1]
            boundary = max((i + 1 for i, token in enumerate(previous) if token == 19), default=0)
            for head in range(16):
                window = [prefix[-1 - lag] if len(prefix) - 1 - lag >= boundary else 19
                          for lag in range(2 + head // 8)]
                code = 0
                for token, multiplier in zip(window, geometry.multipliers):
                    code ^= token * multiplier
                result[index, head] = code % geometry.prime_sizes[head] + geometry.table_offsets[head]
            requests[index] = row
    return result, requests


class ObservedNorm:
    """Retain actual intermediate BF16 values for per-boundary comparisons."""
    def __init__(self, operation, output):
        self.operation, self.output = operation, output
        self.saved = torch.empty_like(output)
        self.device_errors = operation.device_errors

    def __call__(self):
        self.operation()
        self.saved.copy_(self.output)


@pytest.mark.parametrize('mode,tokens', [('decode', 1), ('decode', 8), ('decode', 17), ('prefill', 17)])
@torch.no_grad()
def test_gguf_ple_hash_rows_contribution_and_history_share_one_graph(session, tmp_path, mode, tokens):
    from ignition.components.readiness import ReadinessTimings

    device, sequences = session.dense_blas.device, 3
    geometry = serving.ple_geometry(vocab_size=32, base_size=100)
    raw = np.arange(geometry.padded_vocab_size * 90, dtype=np.int64).astype(np.uint8).reshape(-1, 5, 18)
    raw[..., :2] = np.array([.03125], dtype='<f2').view(np.uint8)
    path = tmp_path / 'ple.gguf'
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('per_layer_token_embd.weight', raw.reshape(-1, 90), raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    stored = np.array(gguf.GGUFReader(path).tensors[0].data).reshape(-1, 90)
    decoded = torch.from_numpy(gguf.dequantize(stored, gguf.GGMLQuantizationType.IQ4_NL)).bfloat16()
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    ready = torch.cuda.Event(external=True); ready.record(session.stream)
    table = GgufPLETable(dict(vocab_size=32), GgufPLELayout(GgufPLECaps(device, 19), geometry),
        GgufPLEGeometry.allocate(geometry, device), torch.zeros(tokens * 16, 90, dtype=torch.uint8, device=device),
        torch.full((tokens, 16), -1, dtype=torch.int64, device=device),
        SimpleNamespace(device_view=torch.zeros(tokens * 16 + 1, dtype=torch.int64, device=device)), ready, (),
        sequence, ReadinessTimings.allocate(frame, 1))
    metadata = PLEMetadata.allocate(tokens, sequences, device, eos_token_id=19)
    scratch = PLEScratch.allocate(tokens, device)
    source = torch.zeros(tokens, 10240, dtype=torch.bfloat16, device=device)
    state = allocate_convolution_state(4, device, speculative_tokens=7)
    weights = torch.tensor([.125, -.25, .5, .25], dtype=torch.bfloat16, device=device).expand(10240, 4).contiguous()
    module = SimpleNamespace(hidden_size=2560, hc_count=4, short_conv_state_len=9,
        conv1d=SimpleNamespace(weight=weights[:, None, :], bias=None))
    for name, delta in (('norm_key', .125), ('norm_query', .25), ('norm_conv', -.125)):
        setattr(module, name, SimpleNamespace(weight=torch.full((10240,), delta, device=device), group_size=2560, eps=1e-6))
    eye = torch.eye(2560, dtype=torch.bfloat16, device=device)
    projections = dict(key_proj=eye.repeat(4, 1), value_proj=eye * .5)
    operation = PreparedPLE(module, projections, source, scratch, state, metadata, table,
        mode=mode, session=session, name='ple')
    assert type(operation).__name__ == 'PreparedActlizePLE'
    operation.key_norm = ObservedNorm(operation.key_norm, scratch.normalized)
    operation.query_norm = ObservedNorm(operation.query_norm, scratch.convolved)

    def forward():
        operation.embedding_lookup.publish()
        operation(source)

    forward(); session.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=session.stream):
        forward()
    try:
        lengths = [min(tokens, 1), min(tokens, 7), 0, tokens]
        for iteration, live in enumerate(lengths):
            starts = [0, min(live, 6), min(live, 12), live]
            slots = [0, 1, -1] if iteration % 2 == 0 else [1, 0, -1]
            history = [[5, 6], [19, 4], [2, 3]]
            token_ids = [19 if i % 3 == 1 else i % 17 for i in range(tokens)]
            expected_hash, expected_requests = hashes(token_ids, starts, history, slots, geometry)
            selected = expected_hash.flatten()
            delivered = torch.from_numpy(stored[selected.clamp_min(0).numpy()].copy())
            delivered[selected < 0] = 0
            table.weights.copy_(delivered)
            metadata.token_ids.copy_(torch.tensor(token_ids))
            metadata.committed_history.copy_(torch.tensor(history))
            metadata.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32))
            metadata.convolution_slots.copy_(torch.tensor(slots))
            metadata.num_tokens.fill_(live); metadata.num_seqs.fill_(sequences)
            metadata.fresh.fill_(True); metadata.accepted_tokens.fill_(1)
            initial = ((torch.arange(tokens * 10240).reshape(tokens, 10240) % 31).float() / 16 - .5).bfloat16()
            source.copy_(initial)
            state.fill_(.375)
            sequence.fill_(iteration + 1)
            ready.record(session.stream)
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay(); session.stream.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            torch.testing.assert_close(table.hashes.cpu(), expected_hash, rtol=0, atol=0)
            torch.testing.assert_close(operation.embedding_lookup.requests.cpu(), expected_requests, rtol=0, atol=0)
            torch.testing.assert_close(table.mailbox.device_view[:-1].cpu(), selected, rtol=0, atol=0)
            assert table.mailbox.device_view[-1].item() == iteration + 1
            torch.testing.assert_close(metadata.committed_history.cpu(), torch.tensor(history), rtol=0, atol=0)
            expected_embedding = decoded[selected.clamp_min(0)].clone()
            expected_embedding[selected < 0] = 0
            torch.testing.assert_close(scratch.embedding.cpu(), expected_embedding.reshape(tokens, 2560), rtol=0, atol=0)
            active = expected_requests >= 0
            expected_key = norm(scratch.key.cpu(), module.norm_key.weight.cpu())
            expected_query = norm(initial, module.norm_query.weight.cpu())
            expected_key[~active] = 0; expected_query[~active] = 0
            within_ulp(operation.key_norm.saved, expected_key)
            within_ulp(operation.query_norm.saved, expected_query)
            key = operation.key_norm.saved.cpu().view(tokens, 4, 2560)
            query = operation.query_norm.saved.cpu().view(tokens, 4, 2560)
            gate = (key * query).sum(-1, keepdim=True) / math.sqrt(2560)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gated = (gate.sigmoid() * scratch.value.cpu()[:, None]).flatten(1)
            gated[~active] = 0
            within_ulp(scratch.gated, gated)
            normalized = norm(scratch.gated.cpu(), module.norm_conv.weight.cpu())
            normalized[~active] = 0
            within_ulp(scratch.normalized, normalized)
            normalized = scratch.normalized.cpu()
            for row, (begin, end) in enumerate(zip(starts, starts[1:])):
                if begin == end or slots[row] < 0:
                    continue
                values = normalized[begin:end]
                padded = torch.cat((torch.zeros(9, 10240).bfloat16(), values))
                conv = torch.stack([sum(padded[i + tap * 3].float() * weights[:, tap].cpu().float()
                    for tap in range(4)) for i in range(end - begin)]).bfloat16()
                conv = torch.nn.functional.silu(conv.float()).bfloat16()
                within_ulp(scratch.convolved[begin:end], conv)
                expected_state = torch.zeros(10240, 16).bfloat16()
                if mode == 'prefill':
                    expected_state[:, :9] = padded[-9:].T
                else:
                    expected_state[:, :9] = padded[1:10].T
                    expected_state[:, 9:9 + end - begin - 1] = values[1:].T
                torch.testing.assert_close(state[slots[row]].cpu(), expected_state, rtol=0, atol=0)
            torch.testing.assert_close(scratch.output.cpu(), (scratch.gated.cpu() + scratch.convolved.cpu()).bfloat16(), rtol=0, atol=0)
            assert not scratch.output[~active.to(device)].any().item()
            assert (state[2:] == .375).all().item()
            assert all(not error.any().item() for error in operation.device_errors)
    finally:
        session.stream.synchronize(); graph.reset()
