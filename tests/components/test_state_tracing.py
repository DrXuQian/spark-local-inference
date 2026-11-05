# SPDX-License-Identifier: Apache-2.0

"""Trace the CUDA state-transfer branches with metadata, without executing kernels."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components import state


@pytest.mark.parametrize('operation', ['gather', 'gather_three', 'save_hidden', 'scatter'])
@pytest.mark.parametrize('tracer', ['dynamo', 'make_fx'])
def test_state_cuda_branches_remain_in_one_graph(monkeypatch, operation, tracer):
    from torch.utils import _triton

    monkeypatch.setattr(_triton, 'has_triton', lambda *args, **kwargs: True)
    with FakeTensorMode():
        source = torch.empty_strided((7, 8), (16, 1), dtype=torch.bfloat16, device='cuda:0')
        output = torch.empty_like(source)
        rows = torch.empty(5, dtype=torch.int32, device='cuda:0')
        active = torch.empty(5, dtype=torch.int32, device='cuda:0')
        sources = (source, source.clone(), source.clone())
        outputs = (output, output.clone(), output.clone())

    def forward():
        if operation == 'gather':
            state.gather_rows(source, rows, output.narrow(0, 0, 5))
        elif operation == 'gather_three':
            state.gather_rows_three(sources, rows, tuple(value.narrow(0, 0, 5) for value in outputs))
        elif operation == 'save_hidden':
            state.save_hidden(source.narrow(0, 0, 5), output, rows, active)
        else:
            state.scatter_rows(source.narrow(0, 0, 5), rows, output, active)

    if tracer == 'make_fx':
        with source.fake_mode:
            graph = make_fx(forward, tracing_mode='fake')()
    else:
        captured = []

        def record(module, examples):
            captured.append(module)
            return lambda *args: ()

        torch.compile(forward, backend=record, fullgraph=True, dynamic=False)()
        graph, = captured
    calls = [node for node in graph.graph.nodes
             if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation]
    assert len(calls) == 1
