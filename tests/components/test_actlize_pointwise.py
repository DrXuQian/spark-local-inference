# SPDX-License-Identifier: Apache-2.0

"""Compilation metadata for external pointwise bindings; no device arithmetic."""

import ctypes
import pickle

import pytest
import torch

pytest.importorskip('actlize_la.serving')

from ignition.components.actlize_pointwise import PreparedActlizePointwise
from ignition.preparation.compilation import ForwardTracer, _lower_storage


def test_rebound_pointwise_inputs_and_all_outputs_survive_storage_lowering():
    operation = PreparedActlizePointwise.__new__(PreparedActlizePointwise)
    residual = torch.empty(2, 4, 2560, dtype=torch.bfloat16)
    gate = torch.empty(2, 4, dtype=torch.bfloat16)
    block = torch.empty(2, 1, 2560, dtype=torch.bfloat16)
    output, combined = torch.empty_like(residual), torch.empty_like(residual)
    error = torch.empty(2, dtype=torch.int32)
    operation.handle = 912345
    operation.inputs = [residual, gate, block]
    operation.buffers = [output, combined, error]
    operation.positions = dict(x=0, gate=1, value=2)
    operation.device_errors = (error,)
    operation.owner = ctypes.c_void_p(1)
    restored = pickle.loads(pickle.dumps(operation))
    assert 'owner' not in vars(restored) and restored.handle == operation.handle
    replacement = torch.empty(2, 1, 5120, dtype=torch.bfloat16)[..., ::2]

    def forward():
        operation(value=replacement)
        residual.copy_(combined)
        return output

    graph = ForwardTracer()(forward)
    calls = _lower_storage(graph)
    assert len(calls) == 1 and calls[0].target == torch.ops.actlize_la.serving_out.default
    assert calls[0].leaves[-1] == operation.handle
    assert sum(mutable for position, mutable, index, view in calls[0].operands) == 3
    # The dynamic block argument must replace the preparation placeholder.
    view = next(view for position, mutable, index, view in calls[0].operands if position == 2)
    assert view[1:3] == (tuple(replacement.shape), replacement.stride())
    assert view[2] != block.stride()
