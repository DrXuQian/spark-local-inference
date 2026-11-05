# SPDX-License-Identifier: Apache-2.0

"""Real producer/consumer host contracts and prepared packed-weight metadata."""

from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip('quactlize.compat.native')

from ignition.components.gguf_weights import GgufPlanes, GgufWeight
from ignition.components.ppu_gguf_projection import _buffers, _disjoint, matmul_packed, projection_sizes
from ignition.preparation.gguf_packing import GgufPacker, arrangement_descriptor
from ignition.preparation.forward_metadata import dumps, loads
from ignition.preparation.rebuild_manifest import declare_weights, declared_views
from tests.components.test_ppu_gguf_projection import contract, operands  # noqa: F401


def packed_weight(contract, raw):
    descriptor, sizes = GgufPacker(contract).describe(raw)
    planes = tuple(torch.empty(count, dtype=torch.uint8, device=raw.device) if count else None
                   for count in (sizes.low_bytes, sizes.high_bytes, sizes.units_bytes))
    values = tuple(getattr(descriptor, name) for name, _ in descriptor._fields_)
    return replace(raw, raw=None, packed=GgufPlanes(*planes, values)), sizes


@pytest.mark.parametrize('qtype,n,k', [(7, 2560, 640), (8, 2560, 640), (8, 1280, 2560), (12, 1280, 2560)])
@pytest.mark.parametrize('m', [1, 8, 9, 128])
def test_model_expert_layouts_use_exact_producer_descriptors_and_byte_budgets(contract, qtype, n, k, m):
    with FakeTensorMode():
        source, raw, output, ids = operands(m, qtype=qtype, n=n, k=k, channels=10, padding=7)
        packed, producer = packed_weight(contract, raw)
        problem, consumer = projection_sizes(contract, source, packed, output, ids=ids)
        assert problem.layout == 1 and packed.raw is None
        assert producer.raw_bytes == raw.raw.nbytes
        assert (producer.low_bytes, producer.high_bytes, producer.units_bytes) == (
            consumer.low_bytes, consumer.high_bytes, consumer.units_bytes)
        assert consumer.low_bytes + consumer.high_bytes + consumer.units_bytes == raw.raw.nbytes
        assert consumer.raw_bytes == 0
        assert consumer.route == ((2 if m <= 8 else 3) if qtype == 7 else (0 if m <= 8 else 1))
        if qtype == 7:
            assert consumer.expanded_weight_bytes == consumer.grouped_input_bytes == consumer.grouped_output_bytes == 0
        if qtype == 8:
            assert packed.packed.high is None


@pytest.mark.parametrize('fault', ['identity', 'missing', 'short', 'qtype'])
def test_packed_mismatches_fail_in_the_real_native_query(contract, fault):
    with FakeTensorMode():
        source, raw, output, ids = operands(9, qtype=7, n=2560, k=640)
        packed, _ = packed_weight(contract, raw)
        if fault == 'identity':
            descriptor = (*packed.packed.arrangement[:-1], packed.packed.arrangement[-1] ^ 1)
            packed = replace(packed, packed=replace(packed.packed, arrangement=descriptor))
        elif fault == 'missing':
            packed = replace(packed, packed=replace(packed.packed, arrangement=()))
        elif fault == 'short':
            plane = torch.empty(packed.packed.low.numel() - 1, dtype=torch.uint8, device=source.device)
            packed = replace(packed, packed=replace(packed.packed, low=plane))
        else:
            packed = replace(packed, qtype=8, encoding='Q8_0')
        with pytest.raises(ValueError, match='producer|arrangement|extents'):
            projection_sizes(contract, source, packed, output, ids=ids)


def test_packed_record_metadata_and_export_declare_planes_without_raw_copy(contract):
    raw = GgufWeight(torch.empty(16 * 128 // 32 * 24, dtype=torch.uint8), 'Q5_1', 7, (16, 128))
    packed, _ = packed_weight(contract, raw)
    declaration = declare_weights({'weights': packed}, components={'weights': 'expert'})
    views = list(declared_views(declaration))
    assert {view[0] for view in views} == {'weights.packed.low', 'weights.packed.high', 'weights.packed.units'}
    assert sum(view[2].nbytes for view in views) == raw.raw.nbytes
    restored = loads(dumps(packed))
    assert restored.raw is None and restored.packed.arrangement == packed.packed.arrangement
    assert bytes(arrangement_descriptor(restored)) == bytes(arrangement_descriptor(packed))
    assert restored.packed.low.shape == packed.packed.low.shape
    with pytest.raises(ValueError, match='exactly one'):
        replace(packed, raw=raw.raw)


@pytest.mark.parametrize('gate_type,down_type', [(12, 7), (8, 8)])
def test_target_and_mtp_pack_each_expert_weight_before_loading_the_next(contract, gate_type, down_type):
    import gguf
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader

    # Exercise the real recipe ordering with metadata-only sources. GPU packing
    # and bytes are tested separately by test_gguf_packing_cuda.py.
    loader = QwenGgufWeightLoader.__new__(QwenGgufWeightLoader)
    loader.shard = TensorParallelShard(0, 1)
    loader.manifest = SimpleNamespace(quactlize=SimpleNamespace(target='ppu15'), tensors={
        'blk.48.ffn_down_exps.weight': SimpleNamespace(shape=(512, 2560, 640), encoding=gguf.GGMLQuantizationType(down_type).name),
        'blk.48.ffn_down_shexp.weight': SimpleNamespace(shape=(2560, 640), encoding='Q8_0')})
    events, raw_owners = [], []
    def weight(qtype, shape):
        if raw_owners:
            assert raw_owners[-1]() is None, 'previous raw expert weight survived packing'
        kind = gguf.GGMLQuantizationType(qtype)
        block, width = gguf.GGML_QUANT_SIZES[kind]
        raw = torch.empty(512 * shape[-2] * shape[-1] // block * width, device='meta', dtype=torch.uint8)
        raw_owners.append(weakref.ref(raw))
        events.append(('read', qtype))
        return GgufWeight(raw, kind.name, qtype, shape)
    def pack(raw):
        events.append(('pack', raw.qtype))
        return packed_weight(contract, raw)[0]
    loader.load_quantized_group = lambda *args, **kwargs: weight(gate_type, (512, 1280, 2560))
    loader.load_quantized = lambda *args, **kwargs: weight(down_type, (512, 2560, 640))
    loader.load_dense = lambda name, **kwargs: torch.empty(
        (2560, 640) if 'down' in name else (640, 2560), device='meta', dtype=torch.bfloat16)
    loader.load_tensor = lambda name: torch.empty((2560,) if 'shexp' in name else (512, 2560), device='meta')
    loader.quactlize_packing = pack
    result = loader.load_moe(48, cache_shared=True)
    assert events == [('read', gate_type), ('pack', gate_type), ('read', down_type), ('pack', down_type)]
    assert all(owner() is None for owner in raw_owners)
    assert result.experts.gate_up.packed is not None and result.experts.down.packed is not None
    loader.quactlize_packing = None
    with pytest.raises(ValueError, match='canonical weight producer'):
        loader.load_moe(48, cache_shared=True)


def test_packed_abi_owns_only_planes_and_keeps_write_spans_disjoint(contract):
    raw = GgufWeight(torch.empty(16 * 128 // 32 * 24, dtype=torch.uint8), 'Q5_1', 7, (16, 128))
    weight, _ = packed_weight(contract, raw)
    planes = (weight.packed.low, weight.packed.high, weight.packed.units)
    source, output = torch.empty((2, 128), dtype=torch.bfloat16), torch.empty((2, 16), dtype=torch.bfloat16)
    workspace, error = torch.empty(0, dtype=torch.uint8), torch.zeros((), dtype=torch.int32)
    buffers = _buffers(source, None, None, output, workspace, error, 0x123456789abc, planes=planes)
    assert buffers.raw is None and buffers.raw_bytes == 0
    assert (buffers.low, buffers.high, buffers.units) == tuple(value.data_ptr() for value in planes)
    assert buffers.stream == 0x123456789abc
    _disjoint((source, *planes, None, output, workspace, error), writes_from=5)
    with pytest.raises(ValueError, match='disjoint'):
        _disjoint((source, *planes, None, planes[0], workspace, error), writes_from=5)


@pytest.mark.parametrize('high_present', [False, True])
def test_packed_custom_op_export_preserves_readonly_planes_and_three_mutations(high_present):
    class Forward(torch.nn.Module):
        def forward(self, source, low, high, units, ids, output, workspace, error):
            matmul_packed(source, low, high, units, ids, output, workspace, error, 987654321)
            return output

    args = (torch.empty(2, 128, dtype=torch.bfloat16), torch.empty(32, dtype=torch.uint8),
        torch.empty(8, dtype=torch.uint8) if high_present else None, torch.empty(8, dtype=torch.uint8),
        torch.empty(2, 2, dtype=torch.int32), torch.empty(2, 2, 16, dtype=torch.bfloat16),
        torch.empty(256, dtype=torch.uint8), torch.empty((), dtype=torch.int32))
    exported = torch.export.export(Forward(), args, strict=True).run_decompositions({})
    mutations = {spec.target for spec in exported.graph_signature.output_specs if spec.kind.name == 'USER_INPUT_MUTATION'}
    assert mutations == {'output', 'workspace', 'error'}
    with pytest.raises(RuntimeError, match='retain the prepared packed projection'):
        exported.module()(*args)
