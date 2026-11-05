# SPDX-License-Identifier: Apache-2.0

"""Real Dynamo launch tracing on a host without a device; no kernel execution."""

import pytest
import torch
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

from ignition.components import probability_normalization as normalization


@pytest.mark.parametrize('in_place', [False, True])
@pytest.mark.parametrize('vocabulary', [17, 65537])
def test_complete_normalization_traces_four_mutating_launches(monkeypatch, in_place, vocabulary):
    from torch.utils import _triton

    # Like the existing prepared-launch tests, admit the installed Triton JIT
    # class for tracing only. Arithmetic is checked by the separate device gate.
    monkeypatch.setattr(_triton, 'has_triton', lambda *args, **kwargs: True)
    probabilities = torch.empty(2, 3, vocabulary)
    output = probabilities if in_place else torch.empty_like(probabilities)
    params = torch.empty(2, 6)
    valid = torch.ones(2, 3, dtype=torch.bool)
    operation = normalization.PreparedProbabilityNormalization(output, valid)
    graphs = []
    def backend(module, examples):
        graphs.append(module)
        return lambda *args: ()
    def forward(probabilities, params):
        operation(probabilities, params)
    torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)(probabilities, params)
    assert len(graphs) == 1
    launches = [node for node in graphs[0].graph.nodes
                if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation]
    assert len(launches) == 4
    assert [kernel_side_table.get_kernel(node.kwargs['kernel_idx']) for node in launches] == [
        normalization._maxima, normalization._filter, normalization._totals, normalization._apply]
    assert launches[-1].kwargs['kwargs']['Output'] is launches[1].kwargs['kwargs']['Output']
    assert launches[-1].kwargs['kwargs']['Valid'] is launches[2].kwargs['kwargs']['Valid']
