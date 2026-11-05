# SPDX-License-Identifier: Apache-2.0

"""Exercise Torch 2.9's real compiler/artifact API on CPU validation operands."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

from ignition.preparation.compilation import _INVOCATIONS, _Invocation, _storage_call
from ignition.preparation.inductor29 import StandaloneArtifact

pytestmark = pytest.mark.skipif(torch.__version__.split('+')[0].split('.')[:2] != ['2', '9'],
                               reason='requires actual Torch 2.9 for its artifact protocol')


@torch.library.custom_op('flare_test_torch29::scale_out', mutates_args=('output',))
def scale_out(source: torch.Tensor, output: torch.Tensor, scale: float) -> None:
    output.copy_(source * scale)


@scale_out.register_fake
def _fake_scale_out(source, output, scale):
    pass


def test_compiled_artifact_roundtrip_keeps_mutation_aliases_and_new_storage():
    def forward(storage):
        scale_out(storage[:16], storage[16:], 3.)
        return (storage[16:],)

    source = torch.arange(32, dtype=torch.float32)
    module = make_fx(forward, tracing_mode='fake', _allow_non_fake_inputs=True)(source)
    artifact = StandaloneArtifact.compile(module, [source],
        {'compile_threads': 1, 'triton.cudagraphs': False, 'emulate_precision_casts': True})
    first, = artifact(source)
    torch.testing.assert_close(first, source[:16] * 3, rtol=0, atol=0)
    assert first.data_ptr() == source[16:].data_ptr()
    encoded = artifact.serialize()
    assert artifact.artifact._artifacts is None
    assert artifact.serialize() == encoded
    restored = StandaloneArtifact.load(encoded)
    different = torch.arange(32, dtype=torch.float32) - 7
    output, = restored(different)
    torch.testing.assert_close(output, different[:16] * 3, rtol=0, atol=0)
    assert output.data_ptr() == different[16:].data_ptr()
    different[:16].fill_(13)
    again, = restored(different)
    assert torch.all(again == 39)


def test_restored_artifact_reads_current_storage_call_bindings():
    def forward(source, output):
        _storage_call([source], [output], 0)
        return (output,)

    def invocation(scale):
        leaves, tree = tree_flatten(((None, None, scale), {}))
        return _Invocation(torch.ops.flare_test_torch29.scale_out.default, leaves, tree,
                           [(0, False, 0, None), (1, True, 0, None)])

    source, output = torch.arange(16, dtype=torch.float32), torch.zeros(16)
    module = make_fx(forward, tracing_mode='fake', _allow_non_fake_inputs=True)(source, output)
    artifact = StandaloneArtifact.compile(module, [source, output],
        {'compile_threads': 1, 'triton.cudagraphs': False, 'emulate_precision_casts': True})
    token = _INVOCATIONS.set([invocation(3.)])
    try:
        artifact(source, output)
        torch.testing.assert_close(output, source * 3, rtol=0, atol=0)
    finally:
        _INVOCATIONS.reset(token)
    restored = StandaloneArtifact.load(artifact.serialize())
    token = _INVOCATIONS.set([invocation(7.)])
    try:
        new_source, new_output = torch.arange(16, dtype=torch.float32) - 4, torch.zeros(16)
        restored(new_source, new_output)
        torch.testing.assert_close(new_output, new_source * 7, rtol=0, atol=0)
        torch.testing.assert_close(output, source * 3, rtol=0, atol=0)
    finally:
        _INVOCATIONS.reset(token)
