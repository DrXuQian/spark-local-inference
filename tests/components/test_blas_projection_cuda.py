# SPDX-License-Identifier: Apache-2.0

"""Run with a matching Torch/device and FLARE_TEST_BLAS_LIBRARY + FLARE_TEST_BLAS_API."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.blas_projection import BlasLibrary, PreparedBlas, PreparedBlasProjection
from ignition.components.linear import UnquantizedProjection, prepare_projection


@pytest.fixture
def owner():
    path = os.environ.get('FLARE_TEST_BLAS_LIBRARY')
    if not path or not torch.cuda.is_available():
        pytest.skip('requires the explicitly assigned GPU/PPU BLAS library and device')
    library = BlasLibrary(path, api=os.environ.get('FLARE_TEST_BLAS_API', 'cublas'))
    prepared = PreparedBlas(library, torch.empty(4 << 20, device='cuda', dtype=torch.uint8))
    yield prepared
    torch.cuda.synchronize(prepared.device)


@pytest.mark.parametrize('rows', [1, 2, 4, 8, 17])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_graph_replay_reads_changed_input_and_keeps_padded_output(owner, rows, dtype):
    device = owner.device
    source = torch.zeros(rows, 64, device=device, dtype=torch.bfloat16)[:, 3:35]
    weight = (torch.arange(19 * 32, device=device).reshape(19, 32).remainder(7).float() / 32).bfloat16()
    backing = torch.full((rows, 31), -37, device=device, dtype=dtype)
    output = backing[:, 5:24]
    op = prepare_projection(source, weight, output, session=SimpleNamespace(dense_blas=owner),
                            name='qualification.dense', workspace=None)
    op(source)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(source)
    for step in range(4):
        source.copy_((torch.arange(rows * 32, device=device).reshape(rows, 32).remainder(11).float() - step) / 32)
        graph.replay()
        expected = (source.cpu().double() @ weight.cpu().double().T).to(dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        assert torch.all(backing[:, :5] == -37) and torch.all(backing[:, 24:] == -37)


@pytest.mark.parametrize('compiler', ['none', 'inductor', 'flare'])
def test_two_projections_share_explicit_workspace_inside_one_graph(owner, compiler):
    source = torch.zeros(8, 2560, device=owner.device, dtype=torch.bfloat16)
    weight = torch.full((640, 2560), 1 / 256, device=owner.device, dtype=torch.bfloat16)
    second = torch.full((32, 640), 1 / 64, device=owner.device, dtype=torch.bfloat16)
    intermediate = torch.empty(8, 640, device=owner.device, dtype=torch.bfloat16)
    output = torch.empty(8, 32, device=owner.device, dtype=torch.float32)
    session = SimpleNamespace(dense_blas=owner)
    first_op = UnquantizedProjection(source, weight, intermediate, session=session, name='first')
    second_op = UnquantizedProjection(intermediate, second, output, session=session, name='second')

    def forward():
        return second_op(first_op(source))

    if compiler == 'inductor':
        forward = torch.compile(forward, fullgraph=True)
    elif compiler == 'flare':
        from ignition.preparation.compilation import compile_forward

        forward = compile_forward(forward)
    forward()
    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    before = torch.cuda.memory_allocated()
    for value in (1 / 8, -1 / 16, 0):
        source.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        assert torch.all(output == value * 100)


def test_rebinding_uses_explicit_workspace_instead_of_saved_address(owner):
    from ignition.components.blas_projection import projection

    source = torch.ones(2, 32, device=owner.device, dtype=torch.bfloat16)
    weight = torch.full((16, 32), 1 / 32, device=owner.device, dtype=torch.bfloat16)
    output = torch.empty(2, 16, device=owner.device, dtype=torch.float32)
    op = PreparedBlasProjection(source, weight, output, owner=owner)
    op(source)
    rebound = torch.empty_like(owner.workspace)
    assert rebound.data_ptr() != owner.workspace.data_ptr()
    projection(source, weight, output, rebound, owner.handle)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        projection(source, weight, output, rebound, owner.handle)
    for value in (2, -4, 0):
        source.fill_(value)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)
