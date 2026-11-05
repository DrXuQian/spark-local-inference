# SPDX-License-Identifier: Apache-2.0

"""External convolution/GDN/norm/rollback in the actual Flare sequence adapter."""

import os
from types import SimpleNamespace

import pytest
import torch

serving = pytest.importorskip('actlize_la.serving')

from ignition.components.gated_delta import GDNScratch, GDNState, PreparedGDN
from ignition.components.recurrent_state import DeltaMetadata
from ignition.preparation.scratch import ScratchArena


@pytest.fixture(scope='module')
def library():
    path = os.environ.get('FLARE_ACTLIZE_LIBRARY')
    target = os.environ.get('FLARE_ACTLIZE_TARGET')
    if os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not target or not torch.cuda.is_available():
        pytest.skip('assign FLARE_RUN_CUDA_TESTS=1, FLARE_ACTLIZE_TARGET and a device')
    return serving.NativeLibrary(path, target=target)


@pytest.mark.parametrize('width', [1, 6])
@pytest.mark.parametrize('conv_slot', [0, 1])
def test_shortconv_norm_and_consumed_prefix_share_one_graph(library, width, conv_slot):
    device = torch.device('cuda:0')
    module = SimpleNamespace(num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
        hidden_size=2560, conv_dim=10240, conv_kernel_size=4, activation='silu',
        conv1d=SimpleNamespace(weight=torch.full((10240, 1, 4), .125, dtype=torch.bfloat16, device=device), bias=None),
        A_log=torch.full((48,), -2., device=device), dt_bias=torch.zeros(48, device=device),
        norm=SimpleNamespace(weight=torch.ones(128, device=device), variance_epsilon=1e-6, activation='silu'))
    scratch = GDNScratch.allocate(width, device)
    state = GDNState.allocate(3, device, speculative_tokens=7)
    metadata = DeltaMetadata.allocate(2, device, verification_columns=8)
    arena = ScratchArena(device, overlap=False)
    records = SimpleNamespace(delta=torch.empty((2, 48, 8, 128), device=device),
        keys=torch.empty((2, 16, 8, 128), device=device), decay=torch.empty((2, 48, 8), device=device))
    # Projections are qualified separately. Supply their actual row-strided
    # outputs to the same run_sequence entry used by prepared model forwards.
    weights = {name: torch.empty(shape, device='meta', dtype=torch.bfloat16) for name, shape in (
        ('in_proj_qkvz', (16384, 2560)), ('in_proj_ba', (96, 2560)), ('out_proj', (2560, 6144)))}
    op = PreparedGDN(module, weights, torch.zeros(width, 2560, dtype=torch.bfloat16, device=device),
        scratch, state, metadata, mode='decode', records=records, workspace=arena,
        prepare_projections=False, actlize_library=library, session=None, name='external.sequence')
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    commit = op.actlize.prepare_commit(counts)
    arena.trim(0); arena.freeze(); arena.finalize()
    scratch.projected_qkvz.normal_(std=.1)
    scratch.projected_ba.normal_(std=.1)
    initial = torch.randn_like(state.recurrent) * .001
    initial[0].zero_()
    history = torch.randn_like(state.convolution) * .1
    metadata.state_indices[:, 0].copy_(torch.tensor([1, 2], device=device))
    metadata.convolution_slots.copy_(torch.tensor([conv_slot, 2], device=device))
    metadata.accepted_tokens.fill_(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def forward():
        for flag in op.actlize.device_errors:
            flag.zero_()
        op.run_sequence()
        commit()

    with torch.cuda.stream(stream):
        forward(); stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        try:
            for consumed in range(width + 1):
                results = []
                for live in (width, consumed):
                    state.recurrent.copy_(initial); state.convolution.copy_(history)
                    metadata.query_start_loc.copy_(torch.tensor([0, live, live], dtype=torch.int32, device=device))
                    metadata.num_seqs.fill_(2); metadata.num_tokens.fill_(live)
                    counts[0].fill_(consumed)
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    graph.replay(); stream.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                    assert all(not flag.any().item() for flag in op.actlize.device_errors)
                    for untouched in ({0, 1, 2} - {conv_slot}):
                        assert torch.equal(state.convolution[untouched], history[untouched])
                    assert torch.equal(state.recurrent[0], initial[0])
                    assert torch.equal(state.recurrent[2], initial[2])
                    chronological = (state.convolution if consumed == 0 else
                                     state.convolution[conv_slot, :, consumed - 1:consumed + 2])
                    results.append((state.recurrent.cpu(), chronological.cpu()))
                for full, truncated in zip(*results, strict=True):
                    torch.testing.assert_close(full, truncated, rtol=0, atol=0)
                if consumed == 0:
                    assert torch.equal(state.convolution, history)
        finally:
            stream.synchronize()
            graph.reset()
