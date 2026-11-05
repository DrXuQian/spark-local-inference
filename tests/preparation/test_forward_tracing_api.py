# SPDX-License-Identifier: Apache-2.0

"""Trace real tensor mutations with current and older make_fx signatures."""

import pytest
import torch

from ignition.preparation import compilation


@pytest.mark.parametrize('older_signature', [False, True])
def test_forward_tracer_preserves_writes_and_aliases_without_optional_metadata_flag(monkeypatch, older_signature):
    original = compilation.make_fx
    if older_signature:
        def legacy(function, tracing_mode, _allow_non_fake_inputs):
            return original(function, tracing_mode=tracing_mode, _allow_non_fake_inputs=_allow_non_fake_inputs)
        monkeypatch.setattr(compilation, 'make_fx', legacy)
    source = torch.arange(8, dtype=torch.float32)
    output = torch.zeros_like(source)

    def forward():
        output.copy_(source * 3)
        return output, output[1:]

    graph = compilation.ForwardTracer()(forward)
    torch.testing.assert_close(source, torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
    assert not output.any().item()
    result, view = graph()
    torch.testing.assert_close(result, torch.arange(8, dtype=torch.float32) * 3, rtol=0, atol=0)
    torch.testing.assert_close(source, torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
    assert result.data_ptr() == output.data_ptr() and view.data_ptr() == output[1:].data_ptr()
