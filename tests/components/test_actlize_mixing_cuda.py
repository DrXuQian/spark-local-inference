# SPDX-License-Identifier: Apache-2.0

"""Prepared HC and full-width MTP feedback using actlizeLA and platform BLAS."""

import os
from types import SimpleNamespace

import pytest
import torch

serving = pytest.importorskip('actlize_la.serving')

from ignition.components.blas_projection import BlasLibrary, PreparedBlas
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.actlize_feedback import PreparedActlizeMTPFeedback
from ignition.recipes.qwen38.hyperconnection import HyperconnectionScratch, PreparedHyperconnection


@pytest.fixture(scope='module')
def session():
    path, blas, target = (os.environ.get(name) for name in
        ('FLARE_ACTLIZE_LIBRARY', 'FLARE_TEST_BLAS_LIBRARY', 'FLARE_ACTLIZE_TARGET'))
    if not path or not blas or not target or not torch.cuda.is_available():
        pytest.skip('assign actual actlizeLA and BLAS libraries with platform Torch/device')
    library = serving.NativeLibrary(path, target=target)
    dense = PreparedBlas(BlasLibrary(blas, api=os.environ.get('FLARE_TEST_BLAS_API', 'acblas')),
                         torch.empty(4 << 20, dtype=torch.uint8, device='cuda:0'))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield SimpleNamespace(actlize_helpers=library, dense_blas=dense, stream=stream)
        stream.synchronize()


def norm(value, weight, groups, eps=1e-6):
    value = value.float().view(value.shape[0], groups, -1)
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            * (1. + weight.float().view(groups, -1))).bfloat16().flatten(1)


def within_bf16_ulp(actual, expected):
    actual, expected = actual.detach().cpu().bfloat16(), expected.bfloat16()
    def ordered(value):
        bits = value.view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert (ordered(actual) - ordered(expected)).abs().max() <= 1


@pytest.mark.parametrize('tokens', [1, 4, 17])
@torch.no_grad()
def test_feedback_normalizes_whole_hidden_then_projects_each_stream(session, tokens):
    device = session.dense_blas.device
    module = SimpleNamespace(config=SimpleNamespace(rms_norm_eps=1e-6),
        pre_fc_norm_embedding=SimpleNamespace(weight=torch.full((2560,), .125, dtype=torch.bfloat16, device=device)),
        pre_fc_norm_hidden=SimpleNamespace(weight=torch.full((10240,), .25, dtype=torch.bfloat16, device=device)),
        fc_embedding=SimpleNamespace(weight=torch.eye(2560, dtype=torch.bfloat16, device=device) * .25),
        fc_hidden=SimpleNamespace(weight=torch.eye(2560, dtype=torch.bfloat16, device=device) * .5))
    token = torch.ones(tokens, 2560, dtype=torch.bfloat16, device=device)
    hidden = torch.zeros(tokens, 10240, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(hidden)
    arena = ScratchArena(device, overlap=False)
    operator = PreparedActlizeMTPFeedback(module, token, hidden, output, arena, session=session, name='feedback')
    assert arena.external_nbytes == tokens * 30720
    arena.reserve(arena.nbytes * 2)
    arena.trim(0); arena.freeze(); arena.finalize()
    operator(); session.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=session.stream):
        operator()
    try:
        for scales in ([1., 2., 4., 8.], [16., 2., 4., 8.]):
            value = torch.tensor(scales).view(1, 4, 1).expand(tokens, 4, 2560).reshape(tokens, 10240).bfloat16()
            hidden.copy_(value)
            normalized = norm(value, module.pre_fc_norm_hidden.weight.cpu(), 1)
            embedding = norm(token.cpu(), module.pre_fc_norm_embedding.weight.cpu(), 1)
            expected = ((normalized * .5).view(tokens, 4, 2560)
                        + (embedding * .25)[:, None]).flatten(1)
            grouped = ((norm(value, module.pre_fc_norm_hidden.weight.cpu(), 4) * .5).view(tokens, 4, 2560)
                       + (embedding * .25)[:, None]).flatten(1)
            assert torch.count_nonzero(grouped != expected) > tokens * 2560
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay(); session.stream.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            torch.testing.assert_close(hidden.cpu(), value, rtol=0, atol=0)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            assert all(not error.any().item() for error in operator.device_errors)
    finally:
        session.stream.synchronize(); graph.reset()


def mixer(device, *, inject):
    module = torch.nn.Module()
    module.hc_count, module.hidden_size = 4, 2560
    module.hc_norm = torch.nn.Module()
    module.hc_norm.eps = 1e-6
    module.hc_norm.weight = torch.nn.Parameter(torch.full((10240,), .125 if inject else .25, device=device))
    module.input_mix_weight_down = torch.nn.Linear(10240, 320, bias=False, dtype=torch.bfloat16, device=device)
    module.input_mix_weight_up = torch.nn.Linear(320, 10240, bias=False, dtype=torch.bfloat16, device=device)
    module.block_inject_weight = (torch.nn.Linear(10240, 4, bias=False, dtype=torch.bfloat16, device=device)
                                 if inject else None)
    module.requires_grad_(False)
    module.input_mix_weight_down.weight.fill_(1 / 8192)
    module.input_mix_weight_up.weight.fill_(1 / 1024)
    if inject:
        module.block_inject_weight.weight.fill_(1 / 4096)
    return module


@pytest.mark.parametrize('tokens', [1, 8, 64])
@torch.no_grad()
def test_hc_mean_and_combined_normalization_preserve_residual_boundaries(session, tokens):
    device = session.dense_blas.device
    scratch = HyperconnectionScratch.allocate(tokens, device)
    residual = torch.zeros(tokens, 10240, dtype=torch.bfloat16, device=device)
    block = torch.full((tokens, 2560), .25, dtype=torch.bfloat16, device=device)
    first, last = mixer(device, inject=True), mixer(device, inject=False)
    operation = PreparedHyperconnection(first, scratch, priming_state=residual, session=session, name='hc.first')
    following = PreparedHyperconnection(last, scratch, priming_state=residual, session=session, name='hc.last')
    operation.prepare_combine_norm(following, priming_state=residual, session=session, name='hc.combine_norm')
    saved_norm, saved_gate = torch.empty_like(residual), torch.empty_like(residual)
    mixed = torch.empty_like(block)

    def forward():
        operation.mix(residual)
        saved_norm.copy_(scratch.normalized); saved_gate.copy_(scratch.gate_workspace)
        mixed.copy_(scratch.block_input)
        operation.combine_norm(residual, block)

    forward(); session.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=session.stream):
        forward()
    try:
        for shift in (0., .5):
            initial = ((torch.arange(tokens * 10240).reshape(tokens, 10240) % 29).float() / 32 + shift).bfloat16()
            residual.copy_(initial)
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay(); session.stream.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            within_bf16_ulp(saved_norm, norm(initial, first.hc_norm.weight.cpu(), 4))
            products = (saved_norm.cpu().view(tokens, 4, 2560)
                        * torch.sigmoid(saved_gate.cpu().view(tokens, 4, 2560))).bfloat16()
            within_bf16_ulp(mixed, products.float().mean(1).bfloat16())
            scale = 2. * torch.sigmoid(scratch.injection_logits.cpu().float() / 4.)
            combined = (initial.float().view(tokens, 4, 2560) + scale[:, :, None] * block.cpu().float()[:, None]).bfloat16().flatten(1)
            within_bf16_ulp(residual, combined)
            within_bf16_ulp(scratch.normalized, norm(residual.cpu(), last.hc_norm.weight.cpu(), 4))
            assert all(not error.any().item() for error in operation.device_errors)
    finally:
        session.stream.synchronize(); graph.reset()
