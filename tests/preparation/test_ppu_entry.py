# SPDX-License-Identifier: Apache-2.0

"""Explicit PPU admission preserves library identity, memory and device visibility."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli.preparation import Host
from ignition.errors import CapacityError, UsageError
from ignition.preparation.actlize import ActlizeLibrarySpec
from ignition.preparation.blas import BlasLibrarySpec
from ignition.preparation.flash_attn3 import FlashAttention3Spec
from ignition.preparation.gguf import GgufCheckpoint
from ignition.preparation.quactlize import QuactlizeHelpersSpec, QuactlizeLibrarySpec
from ignition.preparation.qwen38_gguf import QwenGgufCheckpoint
from ignition.recipes.qwen38.recipe import gguf_recipe


def checkpoint(target='ppu15'):
    path, digest = Path('/unopened-fixture.so'), 'a' * 64
    tensor = SimpleNamespace(nbytes=1 << 30, dtype='F16', shape=(1024, 1024), component='weights')
    draft = GgufCheckpoint(Path('/fixture'), {'draft.weight': tensor}, (), {}, 248320)
    return QwenGgufCheckpoint(Path('/fixture'), {'token_embd.weight': tensor,
        'per_layer_token_embd.weight': SimpleNamespace(**(vars(tensor) | {'component': 'ple_tables'}))},
        (), {}, 248320, draft, Path('/frontend'), {}, 'b' * 64, 'pread',
        ActlizeLibrarySpec(path, digest, target), QuactlizeLibrarySpec(path, digest, target),
        BlasLibrarySpec(path, digest, 'acblas', 4 << 20), QuactlizeHelpersSpec(path, digest, target),
        FlashAttention3Spec(path, digest))


def derive(manifest, *, free=95 << 30):
    recipe = gguf_recipe(manifest, Path('/fixture'))
    gpu = planning.Gpu(0, 'PPU qualification', 'GPU-00000000-0000-0000-0000-000000000001',
                       96 << 30, free, (1, 5))
    return planning.plan(recipe, recipe.checkpoints[0], (gpu,), selected=(0,),
        concurrency=1, reserve_bytes=0, speculation=recipe.speculation, prefill_chunk=64)


@pytest.mark.parametrize('target', ['ppu10', 'ppu15'])
def test_complete_ppu_libraries_use_the_same_memory_and_snapshot_planner(target):
    manifest = checkpoint(target)
    result = derive(manifest)
    assert result.operator_target == target and result.fits
    assert result.to_dict()['operators']['target'] == target
    assert result.costs.page_bytes > 0 and result.snapshots > 0
    assert result.weights_bytes == 2 << 30
    with pytest.raises(CapacityError, match='GiB|GPU|memory'):
        derive(manifest, free=2 << 30)
    with pytest.raises(UsageError, match='one device'):
        replace(result, gpus=result.gpus * 2)
    with pytest.raises(UsageError, match='single-device'):
        replace(result, tuning_gpus=result.gpus)
    with pytest.raises(UsageError, match='BF16'):
        replace(result, native_dense_activations=True)


@pytest.mark.parametrize('missing', ['actlize', 'quactlize', 'quactlize_helpers', 'dense_blas', 'flash_attention3'])
def test_missing_or_mixed_ppu_libraries_fail_before_device_discovery(monkeypatch, missing):
    def forbidden():
        pytest.fail('invalid native libraries reached device discovery')

    monkeypatch.setattr(planning, 'discover_ppus', forbidden)
    manifest = checkpoint()
    recipe = gguf_recipe(replace(manifest, **{missing: None}), Path('/fixture'))
    with pytest.raises(UsageError, match=missing):
        Host.discover(recipe=recipe)
    with pytest.raises(UsageError, match='matching target'):
        derive(replace(manifest, quactlize=replace(manifest.quactlize, target='ppu10')))


def test_ppu_weight_admission_keeps_memory_and_single_device_checks(monkeypatch):
    import torch
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38 import decoder, gguf_weights

    manifest, loaded = checkpoint(), object()
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (1, 5))
    monkeypatch.setattr(decoder, 'device_memory_info', lambda device: (95 << 30, 96 << 30))
    monkeypatch.setattr(gguf_weights, 'load_model', lambda selected, device, **kwargs:
                        loaded if selected is manifest else pytest.fail('checkpoint changed'))
    assert decoder.ModelWeights.load(manifest, 'cuda:0') is loaded
    with pytest.raises(ValueError, match='one CUDA device'):
        decoder.ModelWeights.load(manifest, 'cuda:0', shard=TensorParallelShard(0, 2))
    with pytest.raises(ValueError, match='flash_attention3'):
        decoder.ModelWeights.load(replace(manifest, flash_attention3=None), 'cuda:0')
    monkeypatch.setattr(decoder, 'device_memory_info', lambda device: (8 << 30, 96 << 30))
    with pytest.raises(MemoryError, match='weight loading'):
        decoder.ModelWeights.load(manifest, 'cuda:0')


def test_ppu_weight_reader_requires_explicit_transport_and_native_decoder():
    from ignition.components.gguf_weights import GgufWeightLoader

    manifest = checkpoint()
    with pytest.raises(ValueError, match='pread and prepared QUactlize helpers'):
        GgufWeightLoader(manifest, 'cuda:0', read_mode='pread')
    with pytest.raises(ValueError, match='pread and prepared QUactlize helpers'):
        GgufWeightLoader(manifest, 'cuda:0', read_mode='auto', quactlize_helpers=object())
    helper = object()
    assert GgufWeightLoader(manifest, 'cuda:0', read_mode='pread', quactlize_helpers=helper).quactlize_helpers is helper


@pytest.fixture
def platform_package(tmp_path, monkeypatch):
    """A CUDA property provider, used only to exercise the real child protocol."""
    package = tmp_path / 'torch'
    package.mkdir()
    (package / '__init__.py').write_text('''
import os
from types import SimpleNamespace
from uuid import UUID
def properties(index):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,0'
    return SimpleNamespace(name='PPU-' + str(index), major=1, minor=5, is_integrated=False,
        uuid=SimpleNamespace(bytes=UUID(int=(3, 1)[index]).bytes))
cuda = SimpleNamespace(device_count=lambda: 2, get_device_properties=properties,
    mem_get_info=lambda index: (80 << 30, 96 << 30))
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,0')
    return package


def test_platform_discovery_retains_runtime_ordinals_without_initializing_parent(platform_package):
    script = '''
import json, sys
from types import SimpleNamespace
from ignition.cli.preparation import Host
library = SimpleNamespace(target='ppu15')
recipe = SimpleNamespace(actlize=library, quactlize=library, quactlize_helpers=library,
                         dense_blas=object(), flash_attention3=object())
host = Host.discover(recipe=recipe)
assert 'torch' not in sys.modules
assert 'b12x' not in sys.modules
assert host.restriction == '2,0'
print(json.dumps([vars(gpu) for gpu in host.gpus]))
'''
    result = subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
    devices = json.loads(result.stdout)
    assert [device['index'] for device in devices] == [0, 1]
    assert [device['uuid'][-1] for device in devices] == ['3', '1']
    assert devices[0]['free_bytes'] == 80 << 30 and devices[1]['total_bytes'] == 96 << 30


def test_platform_discovery_rejects_ambiguous_device_identity(platform_package):
    source = platform_package / '__init__.py'
    source.write_text(source.read_text().replace('(3, 1)[index]', '0'))
    with pytest.raises(CapacityError, match='distinct nonzero UUIDs'):
        planning.discover_ppus()
