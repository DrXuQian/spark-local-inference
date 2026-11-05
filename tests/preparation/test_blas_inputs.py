# SPDX-License-Identifier: Apache-2.0

"""BLAS input identity, memory accounting and preparation selection guards."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ignition.preparation.blas import BlasLibrarySpec
from ignition.preparation.checkpoint import CheckpointError
from ignition.preparation.qwen38_gguf import GgufInputs
from ignition.recipes.qwen38.recipe import RECIPE, gguf_cache_costs, gguf_recipe, prepare_gguf


def bundle(directory, *, api='cublas', workspace=4 << 20):
    (directory / 'main.gguf').write_bytes(b'main header fixture')
    (directory / 'mtp.gguf').write_bytes(b'mtp header fixture')
    (directory / 'frontend').mkdir()
    (directory / 'blas.so').write_bytes(b'BLAS identity fixture, not executable')
    value = dict(schema='qwen38-gguf/1', main='main.gguf', mtp='mtp.gguf', frontend='frontend',
                 read_mode='pread', dense_blas=dict(library='blas.so', api=api, workspace_bytes=workspace))
    (directory / 'manifest.json').write_text(json.dumps(value))
    return value


@pytest.mark.parametrize('api', ['cublas', 'acblas'])
def test_manifest_binds_blas_content_api_and_explicit_workspace(tmp_path, api):
    bundle(tmp_path, api=api)
    inputs = GgufInputs.read(tmp_path)
    assert inputs.dense_blas.path == tmp_path / 'blas.so'
    assert inputs.operator_identity()['dense_blas'] == inputs.dense_blas.identity()
    assert inputs.operator_identity()['kv_dtype'] == 'float8_e4m3fn'
    copied = tmp_path / 'copy.so'
    copied.write_bytes(inputs.dense_blas.path.read_bytes())
    assert BlasLibrarySpec.read(tmp_path, dict(library='copy.so', api=api,
        workspace_bytes=4 << 20)).identity() == inputs.dense_blas.identity()
    inputs.dense_blas.path.write_bytes(b'changed BLAS identity')
    with pytest.raises(ValueError, match='changed after input inspection'):
        inputs.dense_blas.load()


@pytest.mark.parametrize('value', [None, '', {}, {'library': 'blas.so'},
    dict(library='blas.so', api='unknown', workspace_bytes=256),
    dict(library='blas.so', api='acblas', workspace_bytes=0),
    dict(library='blas.so', api='acblas', workspace_bytes=257),
    dict(library='blas.so', api='acblas', workspace_bytes=True),
    dict(library='blas.so', api='acblas', workspace_bytes=1 << 64)])
def test_manifest_rejects_ambiguous_or_invalid_blas_resources(tmp_path, value):
    fields = bundle(tmp_path)
    fields['dense_blas'] = value
    (tmp_path / 'manifest.json').write_text(json.dumps(fields))
    with pytest.raises(CheckpointError, match='dense_blas'):
        GgufInputs.read(tmp_path)


def test_planning_records_blas_and_counts_shared_workspace_once(tmp_path):
    from ignition import planning

    bundle(tmp_path)
    inputs = GgufInputs.read(tmp_path)
    tensor = SimpleNamespace(nbytes=2 << 30, dtype='F16', shape=(1024, 1024), component='weights')
    shard = SimpleNamespace(file_size=2 << 30)
    draft = SimpleNamespace(tensors={'blk.48.weight': tensor}, shards=(shard,))
    manifest = SimpleNamespace(draft=draft, shards=(shard,), headers_sha256='a' * 64,
        tensors={'per_layer_token_embd.weight': SimpleNamespace(**(vars(tensor) | {'component':'ple_tables'})),
                 'token_embd.weight': tensor}, actlize=None, quactlize=None, dense_blas=inputs.dense_blas,
        quactlize_helpers=None, flash_attention3=None)
    selected = gguf_recipe(manifest, tmp_path)
    baseline = gguf_recipe(SimpleNamespace(**(vars(manifest) | {'dense_blas':None})), tmp_path)
    assert selected.checkpoints[0].revision != baseline.checkpoints[0].revision
    for mapping in ('ranks', 'plain_ranks'):
        actual, previous = getattr(selected, mapping)[1], getattr(baseline, mapping)[1]
        assert actual.graph_bytes == previous.graph_bytes + inputs.dense_blas.workspace_bytes
        assert (actual.state_row_bytes, actual.page_bytes) == (previous.state_row_bytes, previous.page_bytes)
    gpu = planning.Gpu(0, 'RTX qualification', 'GPU-test', 96 << 30, 95 << 30, (12, 0))
    plan = planning.plan(selected, selected.checkpoints[0], (gpu,), selected=(0,),
        concurrency=1, reserve_bytes=0, speculation=selected.speculation, prefill_chunk=64)
    assert plan.to_dict()['operators'] == inputs.operator_identity()
    assert plan.costs == gguf_cache_costs(speculating=True, bf16=False, blas_workspace_bytes=4 << 20)
    for spec in (replace(inputs.dense_blas, api='acblas'),
                 replace(inputs.dense_blas, workspace_bytes=8 << 20)):
        altered = gguf_recipe(SimpleNamespace(**(vars(manifest) | {'dense_blas':spec})), tmp_path)
        assert altered.checkpoints[0].revision != selected.checkpoints[0].revision


def test_blas_change_after_planning_is_rejected_before_preparation(tmp_path):
    bundle(tmp_path)
    inputs = GgufInputs.read(tmp_path)
    recipe = replace(RECIPE, name='qwen38_gguf', dense_blas=inputs.dense_blas)
    inputs.dense_blas.path.write_bytes(b'changed after planning')
    with pytest.raises(ValueError, match='dense BLAS selection changed after memory planning'):
        prepare_gguf(tmp_path, SimpleNamespace(recipe=recipe))
