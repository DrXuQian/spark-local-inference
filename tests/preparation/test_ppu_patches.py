# SPDX-License-Identifier: Apache-2.0

"""Apply the PPU patch set to actual Torch 2.9 sources without touching the installation."""

import ast
from importlib.metadata import distribution
from importlib.resources import files
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ignition import patches
from ignition.errors import PreparationError


@pytest.fixture
def installation(tmp_path, monkeypatch):
    installed = {name: distribution(name) for name in ('torch', 'triton')}
    if any(installed[name].version.partition('+')[0] != version for name, version in patches._PPU_VERSIONS.items()):
        pytest.skip('requires the inspected Torch 2.9.0/Triton 3.5.0 sources')
    paths = ('torch/_inductor/runtime/triton_heuristics.py', 'triton/runtime/build.py')
    original = {}
    for relative in paths:
        source = Path(installed[relative.split('/')[0]].locate_file(relative)).read_bytes()
        # A box may already have applied these fixes. Normalize only the test
        # copy so both first application and idempotence still execute here.
        for patch in patches._PPU_PATCHES:
            for target, before, after in patches._hunks(files(patches).joinpath(patch).read_text()):
                if target == relative and source.count(after.encode()) == 1:
                    source = source.replace(after.encode(), before.encode(), 1)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        original[relative] = source
        monkeypatch.delitem(sys.modules, relative.removesuffix('.py').replace('/', '.'), raising=False)
    packages = {name: SimpleNamespace(version=version + '+platform', locate_file=lambda path: tmp_path / path)
                for name, version in patches._PPU_VERSIONS.items()}
    monkeypatch.setattr(patches, 'distribution', packages.__getitem__)
    return tmp_path, packages, original


@pytest.mark.parametrize('target', ['ppu10', 'ppu15'])
def test_platform_patch_set_handles_long_triton_cache_names(installation, target):
    root, _, _ = installation
    changed = patches.apply(operator_target=target)
    assert {path.relative_to(root).as_posix() for path in changed} == {
        'torch/_inductor/runtime/triton_heuristics.py', 'triton/runtime/build.py'}
    assert patches.apply(operator_target=target) == ()
    # Execute the patched real reload method against Triton's 150-character filename.
    tree = ast.parse((root / 'torch/_inductor/runtime/triton_heuristics.py').read_text())
    definition, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StaticTritonCompileResult']
    method, = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == 'reload_cubin_path']
    namespace = dict(os=os, triton_cache_dir=lambda device: str(root), triton_hash_to_path_key=lambda value: value)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<patched-reload>', 'exec'), namespace)
    name = 'triton_' + 'long_fused_operation_' * 16
    binary = root / 'hash' / (name[:150] + '.cubin')
    binary.parent.mkdir()
    binary.write_bytes(b'opaque device executable fixture')
    kernel = SimpleNamespace(name=name, hash='hash', cubin_raw=None, cubin_path=None)
    namespace['reload_cubin_path'](SimpleNamespace(kernel=kernel, compile_meta={'device': 0}))
    assert kernel.cubin_path == str(binary)


@pytest.mark.parametrize('name,version', [('torch', '2.8.0'), ('torch', '2.9.1'), ('triton', '3.6.0')])
def test_wrong_platform_version_fails_before_any_write(installation, name, version):
    root, packages, original = installation
    packages[name].version = version
    with pytest.raises(PreparationError, match=f'require {name}=='):
        patches.apply(operator_target='ppu15')
    assert all((root / path).read_bytes() == value for path, value in original.items())


def test_platform_source_mismatch_never_partially_patches_the_installation(installation):
    root, _, original = installation
    path = root / 'triton/runtime/build.py'
    path.write_text(path.read_text().replace('from platform import machine, system, architecture', 'pass'))
    original['triton/runtime/build.py'] = path.read_bytes()
    with pytest.raises(PreparationError, match='preserve the installed artifacts'):
        patches.apply(operator_target='ppu15')
    assert all((root / path).read_bytes() == value for path, value in original.items())
