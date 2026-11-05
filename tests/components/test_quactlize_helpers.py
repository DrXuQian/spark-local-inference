# SPDX-License-Identifier: Apache-2.0

"""Host metadata/compilation checks; these do not execute device kernels."""

import ctypes
import pickle
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

serving = pytest.importorskip('quactlize.serving')

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gguf_weights import GgufWeight
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.components.quactlize_helpers import PreparedQuactlizeHelpers, PreparedQuactlizeRouter
from ignition.preparation.compilation import ForwardTracer, _lower_storage


def metadata_session():
    # This is the same metadata-only state sent to compiler workers. Its key
    # deliberately has no registered DSO, so any attempted device call fails.
    helpers = PreparedQuactlizeHelpers.__new__(PreparedQuactlizeHelpers)
    helpers.handle = 987654
    return SimpleNamespace(quactlize_helpers=helpers)


def test_helpers_serialize_handle_without_native_owner():
    helpers = metadata_session().quactlize_helpers
    helpers.owner = ctypes.c_void_p(1)  # ctypes pointers cannot be pickled.
    restored = pickle.loads(pickle.dumps(helpers))
    assert vars(restored) == {'handle': helpers.handle}


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_router_dispatch_and_lowering_preserve_count_status_and_padded_strides(dtype):
    session = metadata_session()
    logits = torch.empty((4, 520), dtype=dtype)[:, :512]
    scratch = RouterScratch(torch.empty((4, 12))[:, :10],
        torch.empty((4, 12), dtype=torch.int32)[:, :10], torch.empty((4, 12))[:, :10])
    count = torch.zeros(1, dtype=torch.int32)
    route = PreparedRouter(logits, scratch, session=session, name='route', num_valid_tokens=count)
    assert isinstance(route, PreparedQuactlizeRouter) and route.plan is None
    assert route.device_errors == (route.error,)

    def forward():
        route(logits)
        route(logits)
        return scratch.weights

    graph = ForwardTracer()(forward)
    calls = _lower_storage(graph)
    assert [call.target for call in calls] == [torch.ops.quactlize_serving.router.default] * 2
    nodes = [n for n in graph.graph.nodes if n.target == torch.ops.flare.storage_call.default]
    # selected logits, IDs, weights and the sticky error are all explicit writes.
    assert len(nodes[0].args[1]) == 4
    assert nodes[0].args[1] == nodes[1].args[1]
    assert calls[0].leaves[-1] == session.quactlize_helpers.handle
    assert next(view for position, mutable, index, view in calls[0].operands if position == 0)[2] == (520, 1)
    assert {position for position, mutable, index, view in calls[0].operands if not mutable} == {0, 5}
    assert not serving._libraries  # Tracing did not initialize a CUDA context/DSO.


@pytest.mark.parametrize('option', [dict(score_func='sigmoid'), dict(renormalize=False),
    dict(correction_bias=torch.empty(512)), dict(image_mask=torch.empty(4, dtype=torch.bool)),
    dict(routed_scaling_factor=float('nan')), dict(routed_scaling_factor=1e100),
    dict(num_valid_tokens=torch.zeros(1, dtype=torch.int64))])
def test_router_rejects_unsupported_arithmetic_and_metadata(option):
    logits = torch.empty(4, 512, dtype=torch.bfloat16)
    scratch = RouterScratch.allocate(4, 'cpu', experts=512, top_k=10)
    with pytest.raises(ValueError, match='QUactlize'):
        PreparedRouter(logits, scratch, session=metadata_session(), name='route', **option)


@pytest.mark.parametrize('rounded', [True, False])
def test_swiglu_retains_explicit_rounding_and_mutation(rounded):
    helpers = metadata_session().quactlize_helpers
    gate_up = torch.empty(10, 1280, dtype=torch.bfloat16)
    result = torch.empty(10, 640, dtype=torch.bfloat16)
    graph = ForwardTracer()(lambda: helpers.swiglu(gate_up, result, round_silu=rounded))
    calls = _lower_storage(graph)
    assert len(calls) == 1 and calls[0].target == torch.ops.quactlize_serving.swiglu.default
    assert calls[0].leaves[3:] == [rounded, helpers.handle]
    assert {position for position, mutable, index, view in calls[0].operands if mutable} == {2}


@pytest.mark.parametrize('qtype,width,size', [(8, 2560, 34), (20, 160, 18)])
def test_gguf_embedding_dispatch_keeps_raw_bytes_and_dynamic_ids(qtype, width, size):
    session = metadata_session()
    with FakeTensorMode():
        raw = torch.empty(7 * width // 32 * size, dtype=torch.uint8, device='cuda:0')
        weight = GgufWeight(raw, 'Q8_0' if qtype == 8 else 'IQ4_NL', qtype, (7, width))
        ids = torch.empty(4, dtype=torch.int64, device='cuda:0')
        output = torch.empty(4, width, dtype=torch.bfloat16, device='cuda:0')
        count = torch.empty(1, dtype=torch.int32, device='cuda:0')
        operation = PreparedEmbedding(weight, ids, output, num_rows=count, session=session, name='gguf.rows')
        assert operation.raw is raw and operation.ids is ids and operation.num_rows is count
        assert operation() is output
        assert operation.device_errors == (operation.device_error,)
    assert not serving._libraries
