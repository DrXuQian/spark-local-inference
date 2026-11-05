# SPDX-License-Identifier: Apache-2.0

"""Real Torch 2.9 AOT guards and bundled compilation on CPU validation operands."""

import ctypes
import io
import pickle
from dataclasses import replace

import pytest
import torch

from ignition.preparation.aot import (
    BindingMismatch, BoundForward, _guard_code_cache, _load_inductor, aot_forward,
)
from ignition.preparation.torch_aot_api import TORCH29

pytestmark = pytest.mark.skipif(not TORCH29, reason='requires the platform Torch 2.9 artifact protocol')


@torch.library.custom_op('flare_test_aot29::write', mutates_args=('output',))
def write(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    output.copy_(source * handle)


@write.register_fake
def _write_fake(source, output, handle):
    pass


class Forward:
    def __init__(self, handle=3):
        storage = torch.arange(64, dtype=torch.float32)
        self.source, self.output = storage[:32], storage[32:]
        self.handle = handle
        self.negate = False
        self.retained_library = ctypes.CDLL(None)

    def __call__(self):
        write(self.source, self.output, self.handle)
        self.output.add_(self.source)
        if self.negate:
            self.output.neg_()
        return self.output


@pytest.mark.parametrize('metadata_worker', [False, True])
def test_real_guard_and_bundle_reload_rebinds_handles_and_preserves_alias_contract(tmp_path, monkeypatch,
                                                                                 metadata_worker):
    from ignition.preparation.compilation import _artifact_kernel_table
    from ignition.preparation.forward_metadata import dumps, loads

    original = Forward()
    template = loads(dumps(original)) if metadata_worker else original
    compiled = aot_forward(template, options={'compile_threads': 1, 'triton.cudagraphs': False})
    original() if metadata_worker else compiled(original)
    torch.testing.assert_close(original.output, original.source * 4, rtol=0, atol=0)
    backend = compiled._artifacts.compiled_fn
    with _artifact_kernel_table():
        data = backend.artifact.serialize()
        backend.artifact, backend._serialized_artifact = _load_inductor(data, torch.device('cpu'))
    assert backend.artifact._flare_graph.current_callable is not None
    # Rebuild the real Dynamo guard manager from its persisted state. The full
    # outer StorageBackend cache intentionally requires CUDA; a device test
    # covers that boundary without adding CPU serving to the framework.
    with _guard_code_cache(tmp_path / 'guards'):
        restored = type(compiled)(replace(compiled._artifacts, guard_manager=None))

    def no_compile(*args, **kwargs):
        raise AssertionError('binding or executing the saved forward must not recompile')

    monkeypatch.setattr(torch, 'compile', no_compile)
    fresh = Forward(handle=7)
    fresh.source.add_(5)
    assert restored.guard_check(fresh), str(restored._artifacts.guard_manager.check_verbose({'forward': fresh}))
    restored(fresh)
    torch.testing.assert_close(fresh.output, fresh.source * 8, rtol=0, atol=0)
    torch.testing.assert_close(original.output, original.source * 4, rtol=0, atol=0)
    fresh.handle = 11
    restored(fresh)
    torch.testing.assert_close(fresh.output, fresh.source * 12, rtol=0, atol=0)

    changed = Forward()
    changed.source = changed.source.clone()
    with pytest.raises((BindingMismatch, RuntimeError), match='aliases|GuardManager'):
        restored(changed)
    changed = Forward()
    changed.source = changed.source[:16]
    assert not restored.guard_check(changed)
    changed = Forward()
    changed.negate = True
    assert not restored.guard_check(changed)
    bound = BoundForward(restored, fresh, cache_hit=True)
    bound.prepare_capture()
    bound()
    fresh.negate = True
    with pytest.raises(BindingMismatch, match='guards'):
        bound()


def test_legacy_guard_resources_remain_unavailable_instead_of_serializing_native_handles(tmp_path):
    from ignition.preparation.aot import _GuardPickler
    from ignition.preparation.forward_metadata import _Unavailable

    forward = Forward()
    buffer = io.BytesIO()
    _GuardPickler(buffer).dump(forward)
    with _guard_code_cache(tmp_path):
        restored = pickle.loads(buffer.getvalue())
    assert isinstance(restored.retained_library, _Unavailable)
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.retained_library.some_function
    assert restored.handle == 3 and restored.source.device.type == 'cpu'


def test_guard_metadata_preserves_original_tensor_dispatch_keys(tmp_path):
    from ignition.preparation.aot import _GuardPickler
    from ignition.preparation.forward_metadata import dumps, loads

    with torch.inference_mode():
        original = torch.empty(8)
    metadata = loads(dumps(original))
    buffer = io.BytesIO()
    _GuardPickler(buffer).dump(metadata)
    with _guard_code_cache(tmp_path):
        restored = pickle.loads(buffer.getvalue())
    assert restored.dispatch_keys == torch._C._dispatch_keys(original)


def test_deferred_guards_bind_metadata_then_real_storage(tmp_path):
    from ignition.preparation.aot import _DEFERRED_BINDINGS
    from ignition.preparation.forward_metadata import dumps, loads

    # The second metadata serialization corresponds to a compiler worker
    # receiving a forward whose persistent allocations are still deferred.
    deferred = loads(dumps(loads(dumps(Forward()))))
    compiled = aot_forward(deferred, options={'compile_threads': 1, 'triton.cudagraphs': False})
    for metadata in (True, False):
        token = _DEFERRED_BINDINGS.set(metadata)
        try:
            with _guard_code_cache(tmp_path):
                bound = type(compiled)(replace(compiled._artifacts, guard_manager=None))
            owner = deferred if metadata else Forward()
            assert bound.guard_check(owner), str(bound._artifacts.guard_manager.check_verbose({'forward': owner}))
        finally:
            _DEFERRED_BINDINGS.reset(token)


def test_saved_tuning_selection_requires_one_matching_static_result():
    from types import SimpleNamespace
    from triton import Config
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, StaticTritonCompileResult
    from ignition.preparation.inductor_tuning import _selected

    configs = [Config({'XBLOCK': 128}, num_warps=4), Config({'XBLOCK': 256}, num_warps=8)]
    results = [StaticTritonCompileResult(SimpleNamespace(cubin_raw=bytes([index])), config, {}, {})
               for index, config in enumerate(configs)]
    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.inductor_meta = {'kernel_name': 'selection_metadata_fixture'}
    tuner.compile_results = list(results)
    tuner.launchers = [SimpleNamespace(config=configs[1])]
    chosen = _selected(tuner)
    assert chosen.kernel.cubin_raw == bytes([1]) and chosen.config is configs[1]
    assert tuner.compile_results == [results[1]]
    for invalid in ([results[0]], [results[1], results[1]]):
        tuner.compile_results = invalid
        with pytest.raises(ValueError, match='exactly one compiled config'):
            _selected(tuner)


def test_unfinished_reduction_search_cannot_be_frozen():
    from types import SimpleNamespace
    from triton import Config
    from torch._inductor.runtime.hints import HeuristicType
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, StaticTritonCompileResult
    from ignition.preparation.device_binding import _freeze_tuning_winners

    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.heuristic_type = HeuristicType.REDUCTION
    tuner.inductor_meta = {'coordinate_descent_tuning': True}
    config = Config({'XBLOCK': 1, 'R0_BLOCK': 256})
    tuner.compile_results = [StaticTritonCompileResult(None, config, {}, {})]
    bundle = SimpleNamespace(static_autotuners=[
        SimpleNamespace(cache_key='selection', kernel_name='reduction', kernel=tuner)])
    with pytest.raises(ValueError, match='pending coordinate-descent'):
        _freeze_tuning_winners(bundle)
    config.found_by_coordesc = True
    _freeze_tuning_winners(bundle)
    assert tuner.inductor_meta['coordinate_descent_tuning'] is False
    assert tuner.inductor_meta['dynamic_scale_rblock'] is False
