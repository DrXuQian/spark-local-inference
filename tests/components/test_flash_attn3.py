# SPDX-License-Identifier: Apache-2.0

"""FA3 storage validation and compiler metadata, without device execution."""

import ctypes

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ignition.components.flash_attn3 import (
    PreparedFlashAttention3,
    PreparedPagedKVWriter,
)
from ignition.preparation.compilation import ForwardTracer, _lower_storage
from ignition.preparation.forward_metadata import dumps, loads
from ignition.preparation.scratch import ScratchArena


def operands():
    device = 'cuda:0'
    query = torch.empty(7, 24, 256, dtype=torch.bfloat16, device=device)
    kv = [torch.empty(3, 64, 2, 256, dtype=torch.bfloat16, device=device) for _ in range(2)]
    pages = torch.empty(2, 4, dtype=torch.int32, device=device)
    lengths, starts = (torch.empty(n, dtype=torch.int32, device=device) for n in (2, 3))
    return query, *kv, pages, lengths, starts, torch.empty_like(query)


def test_real_constructor_declares_copy_then_rebinds_same_page_geometry():
    with FakeTensorMode():
        query, key, value, pages, lengths, starts, output = operands()
        fresh = [torch.empty_strided((7, 2, 256), (13312, 256, 1), dtype=key.dtype, device=key.device)
                 for _ in range(2)]
        slots = torch.empty(7, dtype=torch.int64, device=key.device)
        arena = ScratchArena(key.device, overlap=False)
        writer = PreparedPagedKVWriter(*fresh, key, value, slots, workspace=arena, max_cache_pages=9)
        attention = PreparedFlashAttention3(object(), query, key, value, pages, lengths, starts, output, width=7)
        assert writer.compiled is None and len(arena._finalizers) == 1
        assert not writer.prepared
        with pytest.raises(RuntimeError, match='finalize paged KV'):
            writer()
        assert writer.device_errors == (writer.error,)
        bigger = [torch.empty(9, 64, 2, 256, dtype=key.dtype, device=key.device) for _ in range(2)]
        for operator in (writer, attention):
            operator.arguments.update(k_cache=bigger[0], v_cache=bigger[1])
            operator.rebind()
        assert writer._arguments()[-1] == 9 * 64
        writer.arguments['k_cache'] = torch.empty(10, 64, 2, 256, dtype=key.dtype, device=key.device)
        with pytest.raises(ValueError, match='page ABI and capacity'):
            writer.rebind()


def test_writer_dynamo_guards_do_not_bind_compiler_process_resources(monkeypatch):
    from torch.utils import _triton
    from ignition.preparation import triton as launches

    # The only stand-ins are device admission and native compilation. Trace the
    # real writer with Dynamo and check that resource replacement does not
    # trigger a second compilation; this is not a device arithmetic test.
    monkeypatch.setattr(_triton, 'has_triton', lambda *args, **kwargs: True)
    native = ctypes.CDLL(None)
    monkeypatch.setattr(launches, 'warmup_triton', lambda *args, **kwargs: native)
    with FakeTensorMode():
        _, key, value, *_ = operands()
        fresh = torch.empty_strided((7, 2, 256), (13312, 256, 1), dtype=key.dtype, device=key.device)
        slots = torch.empty(7, dtype=torch.int64, device=key.device)
        arena = ScratchArena(key.device, overlap=False)
        writer = PreparedPagedKVWriter(fresh, fresh, key, value, slots, workspace=arena)
        arena.freeze(); arena.finalize()
    assert writer.prepared and writer.compiled is native
    writer.compiled = loads(dumps(native))
    graphs = []

    def record(module, examples):
        graphs.append(module)
        return lambda *args: ()

    def forward(owner):
        owner()

    compiled = torch.compile(forward, backend=record, fullgraph=True, dynamic=False)
    compiled(writer)
    writer.compiled = native
    compiled(writer)
    assert len(graphs) == 1
    def failed_compile(*args, **kwargs):
        raise RuntimeError('native compilation failed')

    monkeypatch.setattr(launches, 'warmup_triton', failed_compile)
    with pytest.raises(RuntimeError, match='native compilation failed'):
        writer.prepare()
    assert not writer.prepared
    with pytest.raises(RuntimeError, match='finalize paged KV'):
        writer()


def test_fa3_rejects_output_alias_and_incompatible_metadata():
    with FakeTensorMode():
        query, key, value, pages, lengths, starts, output = operands()
        with pytest.raises(ValueError, match='disjoint'):
            PreparedFlashAttention3(object(), query, key, value, pages, lengths, starts, query, width=7)
        with pytest.raises(ValueError, match='int32'):
            PreparedFlashAttention3(object(), query, key, value, pages, lengths.long(), starts, output, width=7)
        wrong_page = torch.empty(3, 32, 2, 256, dtype=key.dtype, device=key.device)
        with pytest.raises(ValueError, match='page64/D256'):
            PreparedFlashAttention3(object(), query, wrong_page, wrong_page, pages, lengths, starts, output, width=7)


def test_writer_rejects_cache_alias_and_zero_stride_slots():
    with FakeTensorMode():
        _, key, value, *_ = operands()
        fresh = torch.empty(7, 2, 256, dtype=key.dtype, device=key.device)
        slots = torch.empty(7, dtype=torch.int64, device=key.device)
        arena = ScratchArena(key.device, overlap=False)
        with pytest.raises(ValueError, match='disjoint'):
            PreparedPagedKVWriter(fresh, fresh, key, key, slots, workspace=arena)
        broadcast_slots = torch.empty_strided((7,), (0,), dtype=torch.int64, device=key.device)
        with pytest.raises(ValueError, match='int64'):
            PreparedPagedKVWriter(fresh, fresh, key, value, broadcast_slots, workspace=arena)


def test_fa3_opaque_call_preserves_mutation_and_strides_through_storage_lowering():
    # CPU operands are only serialization/tracing fixtures, not FA3 execution.
    operation = PreparedFlashAttention3.__new__(PreparedFlashAttention3)
    query = torch.empty(7, 24, 512, dtype=torch.bfloat16)[..., :256]
    cache = torch.empty(3, 64, 2, 256, dtype=torch.bfloat16)
    operation.library = ctypes.CDLL(None)
    operation.width, operation.splits = 7, 0
    operation.arguments = {'q': query, 'k_cache': cache, 'v_cache': torch.empty_like(cache),
        'page_table': torch.empty(2, 4, dtype=torch.int32), 'cache_seqlens': torch.empty(2, dtype=torch.int32),
        'cu_seqlens_q': torch.empty(3, dtype=torch.int32), 'output': torch.empty(7, 24, 256, dtype=torch.bfloat16)}
    restored = loads(dumps(operation))
    assert 'library' not in vars(restored)
    assert restored.arguments['q'].stride() == query.stride()
    graph = ForwardTracer()(operation)
    calls = _lower_storage(graph)
    assert [call.target for call in calls] == [torch.ops.aten.zero_.default, torch.ops.flare.flash_attn3_paged.default]
    assert sum(mutable for position, mutable, index, view in calls[1].operands) == 1
    assert calls[1].leaves[-2:] == [7, 0]
    view = next(view for position, mutable, index, view in calls[1].operands if position == 0)
    assert view[2] == query.stride()
