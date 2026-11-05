# SPDX-License-Identifier: Apache-2.0

"""Explicit GGUF operator selection and its physical KV/snapshot budget."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ignition.preparation.actlize import ActlizeLibrarySpec
from ignition.preparation.quactlize import QuactlizeLibrarySpec
from ignition.preparation.checkpoint import CheckpointError
from ignition.preparation.qwen38_gguf import GgufInputs
from ignition.recipes.qwen38.recipe import gguf_cache_costs, gguf_recipe


def bundle(tmp_path, *, actlize=False, quactlize=False, helpers=False):
    (tmp_path / 'main.gguf').write_bytes(b'header fixture')
    (tmp_path / 'mtp.gguf').write_bytes(b'draft header fixture')
    (tmp_path / 'frontend').mkdir(exist_ok=True)
    value = dict(schema='qwen38-gguf/1', main='main.gguf', mtp='mtp.gguf',
                 frontend='frontend', read_mode='pread')
    if actlize:
        (tmp_path / 'operator.so').write_bytes(b'operator identity fixture')
        value['actlize_library'] = 'operator.so'
    if quactlize:
        (tmp_path / 'matmul.so').write_bytes(b'matmul identity fixture')
        value['quactlize_abi_library'] = 'matmul.so'
    if helpers:
        (tmp_path / 'helpers.so').write_bytes(b'helpers identity fixture')
        value['quactlize_helpers_library'] = 'helpers.so'
    (tmp_path / 'manifest.json').write_text(json.dumps(value))
    return value


def test_input_bundle_preserves_default_and_binds_selected_library(tmp_path):
    bundle(tmp_path)
    assert GgufInputs.read(tmp_path).actlize is None
    bundle(tmp_path, actlize=True)
    inputs = GgufInputs.read(tmp_path)
    assert inputs.actlize.path == tmp_path / 'operator.so'
    copy = tmp_path / 'same-operator.so'
    copy.write_bytes(inputs.actlize.path.read_bytes())
    assert ActlizeLibrarySpec.read(copy).identity() == inputs.actlize.identity()
    assert inputs.actlize.identity()['kv_dtype'] == 'bfloat16'
    inputs.actlize.path.write_bytes(b'changed operator fixture')
    with pytest.raises(ValueError, match='changed after input inspection'):
        inputs.actlize.load('cuda:0')


def test_helpers_are_independent_and_changed_binary_is_rejected(tmp_path):
    from ignition.preparation.quactlize import QuactlizeHelpersSpec
    from ignition.recipes.qwen38.recipe import RECIPE, prepare_gguf

    bundle(tmp_path, helpers=True)
    inputs = GgufInputs.read(tmp_path)
    assert inputs.quactlize is None and inputs.actlize is None
    copy = tmp_path / 'same-helpers.so'
    copy.write_bytes(inputs.quactlize_helpers.path.read_bytes())
    assert QuactlizeHelpersSpec.read(copy).identity() == inputs.quactlize_helpers.identity()
    assert inputs.operator_identity()['quactlize_helpers']['router'] == 'v2'
    selected = replace(RECIPE, name='qwen38_gguf', quactlize_helpers=inputs.quactlize_helpers)
    inputs.quactlize_helpers.path.write_bytes(b'changed helpers')
    with pytest.raises(ValueError, match='changed after input inspection'):
        inputs.load_operators('cuda:0')
    with pytest.raises(ValueError, match='QUactlize helpers changed after memory planning'):
        prepare_gguf(tmp_path, SimpleNamespace(recipe=selected))


@pytest.mark.parametrize('value', [None, '', 1, {}, []])
@pytest.mark.parametrize('key', ['actlize_library', 'quactlize_abi_library', 'quactlize_helpers_library'])
def test_invalid_library_selection_fails_during_input_read(tmp_path, value, key):
    fields = bundle(tmp_path)
    fields[key] = value
    (tmp_path / 'manifest.json').write_text(json.dumps(fields))
    with pytest.raises(CheckpointError, match='native library path'):
        GgufInputs.read(tmp_path)


@pytest.mark.parametrize('speculating', [False, True])
def test_bf16_planning_matches_actual_cache_and_snapshot_layout(speculating):
    import torch
    from ignition.recipes.qwen38.attention_state import QSAState
    from ignition.recipes.qwen38.state_layout import HybridStateLayout

    for dtype, bf16 in ((torch.float8_e4m3fn, False), (torch.bfloat16, True)):
        costs = gguf_cache_costs(speculating=speculating, bf16=bf16)
        layout = HybridStateLayout.build(speculative_tokens=7 if speculating else 0, kv_dtype=dtype)
        page = QSAState.allocate(pages=1, slots=1, device='meta', kv_dtype=dtype)
        layer_bytes = sum(getattr(page, field).nbytes for field in ('main_k_cache', 'main_v_cache', 'compressed_k_cache'))
        assert costs.state_row_bytes == layout.row_bytes
        assert costs.page_bytes == (12 + int(speculating)) * layer_bytes


def test_preparation_uses_bf16_and_rejects_fp8_snapshot_budget(tmp_path):
    import torch
    from ignition.recipes.qwen38.config import PreparationConfig

    bundle(tmp_path, actlize=True)
    inputs = GgufInputs.read(tmp_path)
    config = PreparationConfig(gguf_inputs=inputs, max_active=1, context=1024, prefill_chunk_size=64)
    assert config.kv_dtype == torch.bfloat16 and config.image_concurrency == 0
    assert replace(config, gguf_inputs=replace(inputs, actlize=None)).kv_dtype == torch.float8_e4m3fn
    fp8 = gguf_cache_costs(speculating=False, bf16=False)
    with pytest.raises(ValueError, match='snapshot budget'):
        replace(config, snapshot_budget_bytes=fp8.state_row_bytes)
    with pytest.raises(ValueError, match='mixed prefill/decode partitions'):
        replace(config, max_active=2, mixed_prefill=True)


@pytest.mark.parametrize('actlize,quactlize,helpers', [
    (True, False, False), (False, True, False), (True, True, False),
    (False, False, True), (True, True, True)])
def test_resolved_recipe_reports_operator_identity_and_sizes_pages(tmp_path, actlize, quactlize, helpers):
    from ignition import planning

    bundle(tmp_path, actlize=actlize, quactlize=quactlize, helpers=helpers)
    inputs = GgufInputs.read(tmp_path)
    # The planner consumes inspected sizes; this fixture does not claim a full checkpoint.
    tensor = SimpleNamespace(nbytes=2 << 30, dtype='F16', shape=(1024, 1024), component='weights')
    shard = SimpleNamespace(file_size=2 << 30)
    draft = SimpleNamespace(tensors={'blk.48.projection': tensor}, shards=(shard,))
    manifest = SimpleNamespace(draft=draft, shards=(shard,), headers_sha256='a' * 64,
        tensors={'per_layer_token_embd.weight': replace_namespace(tensor, component='ple_tables'),
                 'token_embd.weight': tensor}, actlize=inputs.actlize, quactlize=inputs.quactlize,
        dense_blas=None, quactlize_helpers=inputs.quactlize_helpers)
    selected = gguf_recipe(manifest, tmp_path)
    baseline = gguf_recipe(replace_namespace(manifest, actlize=None, quactlize=None, quactlize_helpers=None), tmp_path)
    assert selected.checkpoints[0].alias == baseline.checkpoints[0].alias
    assert selected.checkpoints[0].revision != baseline.checkpoints[0].revision
    for name in ('actlize', 'quactlize', 'quactlize_helpers'):
        spec = getattr(inputs, name)
        if spec is not None:
            changed = replace(spec, sha256='b' * 64)
            assert gguf_recipe(replace_namespace(manifest, **{name: changed}), tmp_path).checkpoints[0].revision != selected.checkpoints[0].revision
    gpu = planning.Gpu(0, 'RTX qualification', 'GPU-test', 96 << 30, 95 << 30, (12, 0))
    plan = planning.plan(selected, selected.checkpoints[0], (gpu,), selected=(0,),
        concurrency=1, reserve_bytes=0, speculation=selected.speculation, prefill_chunk=64)
    costs = gguf_cache_costs(speculating=True, bf16=actlize)
    assert plan.costs == costs
    record = plan.to_dict()
    assert record['operators'] == inputs.operator_identity()
    assert record['page_bytes'] == plan.pages * costs.page_bytes
    assert record['snapshot_virtual_bytes'] == plan.snapshots * costs.state_row_bytes
    other = replace(gpu, capability=(12, 1))
    if actlize or quactlize:
        with pytest.raises(planning.CapacityError, match='SM120'):
            planning.plan(selected, selected.checkpoints[0], (other,), selected=(0,),
                concurrency=1, reserve_bytes=0, speculation=selected.speculation, prefill_chunk=64)
    else:
        # Serving helpers do not select the SM120-only debugging matmul/GDN library.
        planning.plan(selected, selected.checkpoints[0], (other,), selected=(0,),
            concurrency=1, reserve_bytes=0, speculation=selected.speculation, prefill_chunk=64)
    automatic = planning.plan(selected, selected.checkpoints[0], (other, replace(gpu, index=1)), selected=None,
        concurrency=1, reserve_bytes=0, speculation=selected.speculation, prefill_chunk=64)
    assert automatic.gpus[0].index == int(actlize or quactlize)


def test_quactlize_input_preserves_precision_and_rejects_changed_binary(tmp_path, monkeypatch):
    import torch
    from ignition.recipes.qwen38.config import PreparationConfig

    bundle(tmp_path, quactlize=True)
    inputs = GgufInputs.read(tmp_path)
    copy = tmp_path / 'same-matmul.so'
    copy.write_bytes(inputs.quactlize.path.read_bytes())
    assert QuactlizeLibrarySpec.read(copy).identity() == inputs.quactlize.identity()
    config = PreparationConfig(gguf_inputs=inputs, max_active=1, context=1024, prefill_chunk_size=64)
    assert config.kv_dtype == torch.float8_e4m3fn
    assert inputs.operator_identity()['matmul']['target'] == 'nvidia'
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 1))
    with pytest.raises(ValueError, match='SM120'):
        inputs.load_operators('cuda:0')
    inputs.quactlize.path.write_bytes(b'changed matmul fixture')
    with pytest.raises(ValueError, match='changed after input inspection'):
        inputs.load_operators('cuda:0')


@pytest.mark.parametrize('name', ['actlize', 'quactlize'])
def test_prepare_rejects_operator_changed_after_planning(tmp_path, name):
    from ignition.recipes.qwen38.recipe import RECIPE, prepare_gguf

    bundle(tmp_path, actlize=True, quactlize=True)
    inputs = GgufInputs.read(tmp_path)
    recipe = replace(RECIPE, name='qwen38_gguf', actlize=inputs.actlize, quactlize=inputs.quactlize)
    getattr(inputs, name).path.write_bytes(b'changed operator after planning')
    with pytest.raises(ValueError, match='changed after memory planning'):
        prepare_gguf(tmp_path, SimpleNamespace(recipe=recipe))


def replace_namespace(value, **fields):
    return SimpleNamespace(**(vars(value) | fields))
