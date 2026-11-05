# SPDX-License-Identifier: Apache-2.0

"""Real package discovery over file fixtures; no native execution is simulated."""

import ctypes
import hashlib
import json
import subprocess

import pytest
import torch

discovery = pytest.importorskip('quactlize._library')

from ignition.preparation.quactlize import QuactlizeLibrarySpec, QuactlizeHelpersSpec, resolve_quactlize
from ignition.preparation.qwen38_gguf import GgufInputs
from tests.preparation.test_qwen38_gguf_inputs import bundle


@pytest.fixture
def installed(tmp_path, monkeypatch):
    # The actual QUactlize verifier checks every required resource and its hash.
    # These files deliberately are not executable DSOs; GPU tests use real wheels.
    package = tmp_path / 'installed/quactlize'
    root = package / '_native'
    for name in discovery.REQUIRED_PAYLOADS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'framework package-discovery fixture, not device code\n')
    source = package / 'compat/native.py'
    source.parent.mkdir()
    source.write_text('# binding identity fixture\n')
    build = dict(source_hashes={'quactlize/compat/native.py': discovery.file_sha256(source)})
    manifest = dict(schema='quactlize.ppu-library.v1', library=discovery.PUBLIC_LIBRARY, build_info=build,
                    payloads={name: discovery.file_sha256(root / name) for name in discovery.REQUIRED_PAYLOADS})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'wheel.json').write_text(json.dumps(dict(schema='quactlize.ppu-wheel.v1',
        version=discovery.__version__, supported_targets=['ppu10'], default_target='ppu10', build_info=build,
        verified_exports=discovery.PUBLIC_EXPORTS, manifest_sha256=discovery.file_sha256(root / 'manifest.json'))))
    monkeypatch.setattr(discovery, '__file__', str(package / '_library.py'))
    return root


def test_manifest_resolves_both_wheel_roles_without_loading_a_device(installed, tmp_path, monkeypatch):
    value = bundle(tmp_path, actlize=True, fa3=True)
    value['operator_target'] = 'ppu10'
    (tmp_path / 'manifest.json').write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        pytest.fail('package selection initialized a device, loaded a DSO or launched a child')
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    inputs = GgufInputs.read(tmp_path)
    assert inputs.quactlize.path == inputs.quactlize_helpers.path == installed / discovery.PUBLIC_LIBRARY
    expected = hashlib.sha256((installed / 'manifest.json').read_bytes()).hexdigest()
    assert inputs.quactlize.package_sha256 == inputs.quactlize_helpers.package_sha256 == expected
    assert inputs.operator_identity()['matmul']['package_manifest'] == expected
    assert inputs.operator_identity()['quactlize_helpers']['package_manifest'] == expected
    inputs.quactlize.validate()
    inputs.quactlize_helpers.validate()


@pytest.mark.parametrize('spec_class', [QuactlizeLibrarySpec, QuactlizeHelpersSpec])
def test_private_runtime_updates_change_identity_and_reject_stale_bindings(installed, spec_class):
    before = spec_class.installed(target='ppu10')
    name = 'quactlize-runtime/libquactlize_ppu_execution.so'
    (installed / name).write_bytes(b'updated private execution fixture')
    manifest = json.loads((installed / 'manifest.json').read_text())
    manifest['payloads'][name] = discovery.file_sha256(installed / name)
    (installed / 'manifest.json').write_text(json.dumps(manifest))
    wheel = json.loads((installed / 'wheel.json').read_text())
    wheel['manifest_sha256'] = discovery.file_sha256(installed / 'manifest.json')
    (installed / 'wheel.json').write_text(json.dumps(wheel))
    after = spec_class.installed(target='ppu10')
    assert before.path == after.path and before.sha256 == after.sha256
    assert before.identity() != after.identity()
    with pytest.raises(ValueError, match='changed after input inspection'):
        before.load('cuda:0')
    after.validate()


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'python', 'target'])
def test_installed_package_failure_does_not_fall_back(installed, fault):
    old = QuactlizeLibrarySpec.installed(target='ppu10')
    path = installed / 'quactlize-runtime/libquactlize_ppu_execution.so'
    if fault == 'missing':
        path.unlink()
    elif fault == 'corrupt':
        path.write_bytes(b'corrupt')
    elif fault == 'python':
        (installed.parent / 'compat/native.py').write_text('# different bindings\n')
    with pytest.raises(discovery.LibraryError, match='resource|hash mismatch|supports'):
        if fault == 'target':
            resolve_quactlize(target='ppu15')
        else:
            old.validate()


def test_explicit_paths_and_nvidia_default_do_not_discover_a_wheel(tmp_path, monkeypatch):
    import quactlize
    def forbidden(*args, **kwargs):
        pytest.fail('an explicit path or NVIDIA default queried the installed PPU wheel')
    monkeypatch.setattr(quactlize, 'get_library_info', forbidden)
    path = tmp_path / 'explicit.so'
    path.write_bytes(b'explicit operator file')
    projection, helpers = resolve_quactlize(path, path, target='ppu15')
    assert projection.path == helpers.path == path
    assert projection.package_sha256 is helpers.package_sha256 is None
    assert 'package_manifest' not in projection.identity()
    assert resolve_quactlize(target='nvidia') == (None, None)
    projection.validate()
    helpers.validate()


@pytest.mark.parametrize('role', ['matmul', 'helpers'])
def test_each_explicit_role_overrides_only_its_wheel_default(installed, tmp_path, role):
    path = tmp_path / 'explicit.so'
    path.write_bytes(b'explicit override')
    projection, helpers = resolve_quactlize(path if role == 'matmul' else None,
                                           path if role == 'helpers' else None, target='ppu10')
    override, default = (projection, helpers) if role == 'matmul' else (helpers, projection)
    assert override.path == path and override.package_sha256 is None
    assert default.path == installed / discovery.PUBLIC_LIBRARY and default.package_sha256 is not None
