# SPDX-License-Identifier: Apache-2.0

"""Framework package identity through actlizeLA's real file-only verifier."""

import ctypes
import hashlib
import json
import subprocess

import pytest
import torch

discovery = pytest.importorskip('actlize_la._library')

from ignition.preparation.actlize import ActlizeLibrarySpec, resolve_actlize
from ignition.preparation.qwen38_gguf import GgufInputs
from tests.preparation.test_qwen38_gguf_inputs import bundle


def seal(package, manifest):
    path = package / '_native/manifest.json'
    path.write_text(json.dumps(manifest))
    (package / '_native/wheel.json').write_text(json.dumps(dict(
        schema=1, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


@pytest.fixture
def installed(tmp_path, monkeypatch):
    package = tmp_path / 'installed/actlize_la'
    names = (*discovery.REQUIRED_BINDINGS,
             *(f'_native/ppu10/{name}' for name in (*discovery.HEADERS, 'serving.so', 'graph')))
    for name in names:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'file identity fixture, not executable device code\n')
    manifest = dict(schema=1, version=discovery.__version__, source_revision='a' * 40,
        source_dirty=False, source_identity={}, default_target='ppu10',
        targets={'ppu10': dict(library='_native/ppu10/serving.so', native_graph_example='_native/ppu10/graph',
            compatibility=dict(compiler_version='fixture'), abi=dict(version=discovery.ABI_VERSION,
                target=discovery.TARGET_IDS['ppu10'], exports=discovery.EXPORTS,
                struct_sizes={name: 1 for name in discovery.STRUCT_NAMES}))},
        resources={name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names})
    seal(package, manifest)
    monkeypatch.setattr(discovery, '__file__', str(package / '_library.py'))
    return package, manifest


def test_gguf_manifest_discovers_installed_actlize_without_a_device(installed, tmp_path, monkeypatch):
    package, _ = installed
    value = bundle(tmp_path, quactlize=True, helpers=True, fa3=True)
    value['operator_target'] = 'ppu10'
    (tmp_path / 'manifest.json').write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        pytest.fail('package selection loaded a DSO, initialized a device or launched a child')
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    inputs = GgufInputs.read(tmp_path)
    assert inputs.actlize.path == package / '_native/ppu10/serving.so'
    digest = hashlib.sha256((package / '_native/manifest.json').read_bytes()).hexdigest()
    assert inputs.actlize.package_sha256 == inputs.operator_identity()['package_manifest'] == digest
    assert inputs.operator_identity()['kv_dtype'] == 'bfloat16'
    inputs.actlize.validate()


def test_changed_bindings_invalidate_prepared_identity_even_with_the_same_dso(installed, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from ignition.recipes.qwen38.recipe import RECIPE, prepare_gguf

    package, manifest = installed
    value = bundle(tmp_path, quactlize=True, helpers=True)
    value['operator_target'] = 'ppu10'
    (tmp_path / 'manifest.json').write_text(json.dumps(value))
    before = GgufInputs.read(tmp_path).actlize
    recipe = replace(RECIPE, name='qwen38_gguf', actlize=before)
    name = 'serving/flare.py'
    (package / name).write_bytes(b'updated preparation binding fixture')
    manifest['resources'][name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    seal(package, manifest)
    after = resolve_actlize(target='ppu10')
    assert before.path == after.path and before.sha256 == after.sha256
    assert before.identity() != after.identity()
    with pytest.raises(ValueError, match='changed after input inspection'):
        before.load('cuda:0')
    with pytest.raises(ValueError, match='actlizeLA selection changed after memory planning'):
        prepare_gguf(tmp_path, SimpleNamespace(recipe=recipe))
    after.validate()


@pytest.mark.parametrize('fault', ['missing', 'native', 'binding', 'extra', 'target'])
def test_broken_installed_artifact_cannot_fall_back(installed, fault):
    package, _ = installed
    spec = ActlizeLibrarySpec.installed(target='ppu10')
    if fault == 'missing':
        (package / '_native/ppu10/serving.so').unlink()
    elif fault == 'native':
        (package / '_native/ppu10/serving.so').write_bytes(b'corrupt')
    elif fault == 'binding':
        (package / 'serving/flare.py').write_bytes(b'corrupt')
    elif fault == 'extra':
        (package / 'serving/extra.py').write_bytes(b'unbound module')
    with pytest.raises(discovery.LibraryError, match='resource|hash mismatch|unavailable'):
        if fault == 'target':
            resolve_actlize(target='ppu15')
        else:
            spec.validate()


def test_explicit_target_path_and_nvidia_default_preserve_existing_selection(tmp_path, monkeypatch):
    import actlize_la
    def forbidden(*args, **kwargs):
        pytest.fail('an explicit path or NVIDIA default queried the installed PPU wheel')
    monkeypatch.setattr(actlize_la, 'get_library_info', forbidden)
    path = tmp_path / 'explicit.so'
    path.write_bytes(b'explicit operator identity fixture')
    spec = resolve_actlize(path, target='ppu15')
    assert spec.path == path and spec.target == 'ppu15' and spec.package_sha256 is None
    assert 'package_manifest' not in spec.identity()
    spec.validate()
    assert resolve_actlize(target='cuda_sm120') is None
