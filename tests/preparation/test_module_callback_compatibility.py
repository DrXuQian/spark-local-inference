# SPDX-License-Identifier: Apache-2.0

"""Real native recorder calls across a host C ABI; no GPU delivery is claimed."""

import ctypes as C
from pathlib import Path
import shutil
import subprocess

import pytest

from ignition.preparation.module_images import ModuleImages, check_callback_names


@pytest.fixture
def cupti(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('host C compiler required')
    path = tmp_path / 'libcupti-test.so'
    source = Path(__file__).with_name('fixtures') / 'recorder_cupti.c'
    subprocess.run([compiler, '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                    str(source), '-o', str(path)], check=True, capture_output=True)
    library = C.CDLL(str(path))
    library.configure.argtypes, library.configure.restype = [C.c_uint], None
    library.observed.argtypes, library.observed.restype = [C.c_uint], C.c_uint
    library.emit_module.argtypes, library.emit_module.restype = [C.c_size_t, C.c_int], None
    return path, library


@pytest.mark.parametrize('alias', [0, 1], ids=['cuda', 'hgpti'])
def test_name_check_is_independent_of_subscription_and_start_enables_fatbinary(cupti, alias):
    path, library = cupti
    library.configure(alias)
    check_callback_names(path)
    assert library.observed(0) == (66 if alias else 47)
    assert [library.observed(i) for i in (1, 2, 3, 4)] == [0, 0, 0, 0]
    with ModuleImages(path):
        assert ModuleImages.active == 1
        assert [library.observed(i) for i in (1, 2, 3, 4)] == [1, 69 if alias else 50, 0, 1]
    assert library.observed(3) == 1
    assert ModuleImages.active == 0


@pytest.mark.parametrize('fault, message', [
    (2, 'CUPTI 1, no callback name'),
    (8, 'unexpected callback name'),
    (16, 'CUPTI 0, no callback name'),
])
def test_missing_wrong_and_null_callback_names_reject_before_subscribing(cupti, fault, message):
    path, library = cupti
    library.configure(fault | 1)
    with pytest.raises(RuntimeError) as caught:
        with ModuleImages(path):
            pytest.fail('unsupported callback table was accepted')
    error = str(caught.value)
    assert message in error and 'cuFuncLoad (domain=1, id=742)' in error
    assert str(path) in error
    assert library.observed(0) == 66
    assert [library.observed(i) for i in (1, 2, 3)] == [0, 0, 0]
    assert ModuleImages.active == 0


def test_all_missing_callbacks_are_reported_together(cupti):
    path, library = cupti
    library.configure(2 | 4)
    with pytest.raises(RuntimeError) as caught:
        check_callback_names(path)
    error = str(caught.value)
    assert 'cuFuncLoad (domain=1, id=742)' in error
    assert 'cuModuleLoad (domain=1, id=18)' in error
    assert library.observed(0) == 47 and library.observed(1) == 0


def test_successful_name_lookup_does_not_hide_enable_failure_or_leak_subscription(cupti):
    path, library = cupti
    library.configure(32)
    check_callback_names(path)
    with pytest.raises(RuntimeError, match=r'id=742, name=cuFuncLoad.*CUPTI 27'):
        with ModuleImages(path):
            pytest.fail('failed activation was accepted')
    assert [library.observed(i) for i in (1, 3)] == [1, 1]
    assert ModuleImages.active == 0
    library.configure(1)
    with ModuleImages(path):
        assert ModuleImages.active == 1
    assert library.observed(3) == 1 and ModuleImages.active == 0


def test_ppu_export_rejects_callback_table_before_compiling_or_creating_output(cupti, tmp_path, monkeypatch):
    from ignition.preparation import module_images
    from qualify.ppu_export import main

    path, library = cupti
    library.configure(2)
    monkeypatch.setattr(module_images, 'cupti_library', lambda library=None: path)
    output = tmp_path / 'export'
    with pytest.raises(RuntimeError, match=r'cuFuncLoad \(domain=1, id=742\)'):
        main(['--target', 'ppu10', '--compiler', '/must-not-run', '--output', str(output)])
    assert not output.exists()
    assert library.observed(1) == 0


@pytest.mark.parametrize('abi', ['cupti', 'hgpti-bare'])
def test_module_payloads_cross_the_native_callback_boundary_without_a_crash(cupti, abi):
    path, library = cupti
    library.configure(1)
    assert ModuleImages.active_resource_abi == 'cupti'
    with ModuleImages(path, module_resource_abi=abi) as images:
        assert ModuleImages.active_resource_abi == abi
        library.emit_module(0x1000, abi == 'hgpti-bare')
        images.native.check_error()
    assert ModuleImages.active == 0 and ModuleImages.active_resource_abi == 'cupti'
    assert [library.observed(i) for i in (1, 2, 3)] == [1, 69, 1]


def test_bare_context_failure_unsubscribes_and_restores_the_parent_scope(cupti):
    path, library = cupti
    library.configure(1)
    with pytest.raises(RuntimeError, match='no owning driver context'):
        with ModuleImages(path, module_resource_abi='hgpti-bare') as images:
            library.emit_module(0, True)
            images.native.check_error()
    assert ModuleImages.active == 0 and ModuleImages.active_resource_abi == 'cupti'
    assert library.observed(3) == 1


def test_bare_mode_rejects_a_non_hgpti_provider_before_subscribing(cupti):
    path, library = cupti
    library.configure(0)
    with pytest.raises(RuntimeError, match='hgpti-bare requires HGPTI callback names'):
        with ModuleImages(path, module_resource_abi='hgpti-bare'):
            pytest.fail('bare module mode accepted a non-HGPTI provider')
    assert library.observed(1) == 0
    assert ModuleImages.active_resource_abi == 'cupti'


@pytest.mark.parametrize('fault, message', [(64, 'no callback name'), (128, 'CUPTI 27')])
def test_hg_graph_history_requires_both_name_and_activation(cupti, fault, message):
    path, library = cupti
    library.configure(1 | fault)
    with pytest.raises(RuntimeError, match=message):
        with ModuleImages(path):
            pytest.fail('incomplete graph-history recording was accepted')
    assert ModuleImages.active == 0
    assert library.observed(1) == library.observed(3)
