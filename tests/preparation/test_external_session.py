# SPDX-License-Identifier: Apache-2.0

"""External operator preparation preserves scratch, capture and drain ordering."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ignition.preparation.external_session import ExternalPreparationSession
from ignition.preparation.forward_metadata import _Unavailable, dumps, loads
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.tuning import external_operators, preparation_session


def session():
    return ExternalPreparationSession(device='cuda:0', checkpoint='test-revision', namespace={'operators': 'test'})


def scratch():
    with FakeTensorMode():
        arena = ScratchArena('cuda:0', overlap=False)
        arena.reserve(512)
    return arena


def test_finalize_complete_scratch_before_capture_and_drain_before_release(monkeypatch):
    owner, arena, bound = session(), scratch(), []
    before = arena.tensor.untyped_storage()._cdata
    arena.register_finalizer(lambda: bound.append(arena.views(((96,), torch.uint8), ((32,), torch.int32))))
    with pytest.raises(RuntimeError, match='finalized'):
        with owner.capture():
            pass
    owner.register_scratch(arena)
    report = owner.tune()
    assert report['external_binding_finalizers'] == 1 and report['scratch_bytes'] == 512
    assert owner.state == 'ready' and arena.frozen
    assert bound[0][0].untyped_storage()._cdata == before
    with pytest.raises(RuntimeError, match='frozen'):
        arena.reserve(1024)
    with owner.capture():
        with pytest.raises(RuntimeError, match='capture'):
            owner.close()
        with pytest.raises(RuntimeError, match='capture'):
            owner.freeze()
    owner.freeze()
    native = owner.dense_blas = object()

    def drain(device):
        assert device == torch.device('cuda:0')
        assert owner.dense_blas is native and owner._scratch_arena is arena

    monkeypatch.setattr(torch.cuda, 'synchronize', drain)
    owner.close()
    assert owner.state == 'closed' and owner._scratch_arena is None and owner.dense_blas is None
    owner.close()


def test_failed_binding_or_failed_drain_does_not_release_live_resources(monkeypatch):
    owner, arena = session(), scratch()

    def fail():
        raise RuntimeError('native binding failed')

    arena.register_finalizer(fail)
    owner.register_scratch(arena)
    with pytest.raises(RuntimeError, match='native binding failed'):
        owner.tune()
    assert owner.state == 'failed'
    with pytest.raises(RuntimeError, match='finalized'):
        with owner.capture():
            pass
    native = owner.dense_blas = object()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: fail())
    with pytest.raises(RuntimeError, match='native binding failed'):
        owner.close()
    assert owner.dense_blas is native and owner._scratch_arena is arena and arena._finalizers == [fail]
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    owner.close()
    assert owner.state == 'closed' and not arena._finalizers


def test_external_session_never_enters_compiler_worker_metadata():
    restored = loads(dumps(session()))
    assert isinstance(restored, _Unavailable)
    with pytest.raises(RuntimeError, match='process-local'):
        restored.tune()


def test_external_session_rejects_unresolved_parallel_plans():
    owner, arena = session(), scratch()
    arena.overlap = True
    arena.register_parallel(object(), object())
    owner.register_scratch(arena)
    with pytest.raises(ValueError, match='disjoint scratch'):
        owner.tune()
    assert owner.state == 'failed' and not arena.frozen


def test_explicit_ppu_selection_requires_the_complete_external_operator_set():
    values = {name: SimpleNamespace(target='ppu15') for name in
              ('actlize', 'quactlize', 'dense_blas', 'quactlize_helpers', 'flash_attention3')}
    inputs = SimpleNamespace(**values)
    assert external_operators(inputs)
    owner = preparation_session(device='cuda:0', checkpoint='test', gguf_inputs=inputs)
    assert isinstance(owner, ExternalPreparationSession)
    assert owner.namespace.to_dict()['numerical_policy'] == 'gguf-bf16-fp32-accum'
    inputs.flash_attention3 = None
    with pytest.raises(ValueError, match='flash_attention3'):
        external_operators(inputs)
    assert external_operators(None) is False


def test_text_bootstrap_and_single_rank_copies_do_not_import_unused_operator_packages():
    script = '''
import importlib.abc, sys, torch
class RejectUnused(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('b12x', 'cutlass', 'cuda.tile', 'flashinfer', 'transformers')):
            raise AssertionError('unused dependency imported: ' + fullname)
sys.meta_path.insert(0, RejectUnused())
from ignition.recipes.qwen38 import bootstrap
from ignition.recipes.qwen38 import mixed_state
from qualify import qwen38_gguf
from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedBroadcast
source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
for kind in (PreparedAllGather, PreparedAllReduce):
    output = torch.empty_like(source)
    operation = kind(source, output, runtime=None, session=None, coordinator=None, name='single')
    operation(source)
    torch.testing.assert_close(source, output, rtol=0, atol=0)
PreparedBroadcast((source,), runtime=None, session=None, coordinator=None, name='single')
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
