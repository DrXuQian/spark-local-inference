# SPDX-License-Identifier: Apache-2.0

"""The b12x-free compiler catalog: worker, disk reload, guards and graph replay."""

import os

import pytest
import torch

from ignition.preparation.aot import BindingMismatch, ForwardCache
from ignition.preparation.execution import GraphSignature

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                               reason='requires an assigned device and FLARE_RUN_CUDA_TESTS=1')


@torch.library.custom_op('flare_test_external_aot::write', mutates_args=('output',))
def write(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    output.copy_(source * handle)


@write.register_fake
def _write_fake(source, output, handle):
    pass


class Forward:
    def __init__(self, handle):
        storage = torch.ones((2, 128, 4096), dtype=torch.float32, device='cuda:0')
        self.source, self.output = storage[0], storage[1]
        self.sums = torch.empty(128, device='cuda:0')
        self.count = torch.full((), 128, dtype=torch.int32, device='cuda:0')
        self.handle = handle
        self.negate = False

    def __call__(self):
        write(self.source, self.output, self.handle)
        self.output.add_(self.source)
        values = self.output.sum(dim=1)
        self.sums.copy_(torch.where(torch.arange(128, device=self.source.device) < self.count, values, 0))
        if self.negate:
            self.sums.neg_()


@torch.no_grad()
def test_worker_catalog_restores_frozen_kernels_and_replays_new_bindings(tmp_path, monkeypatch):
    import triton
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ignition.preparation import aot
    from ignition.preparation.compile_pool import CompilePool

    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        cache = ForwardCache(namespace={'qualification': 'external-aot'}, device='cuda:0',
                             root=tmp_path, compile_workers=1)
        signatures = {'forward': GraphSignature('prefill', 128, 1)}
        original = Forward(3)
        compiled = cache.prepare_many({'forward': original}, signatures)['forward']
        assert not compiled.cache_hit
        torch.testing.assert_close(original.sums, torch.full_like(original.sums, 4 * 4096), rtol=0, atol=0)
        backend = compiled.artifact._artifacts.compiled_fn
        assert backend._tuning_frozen
        tuners = backend.artifact._flare_graph._triton_bundle.static_autotuners
        assert tuners and all(len(tuner.kernel.compile_results) == 1 for tuner in tuners)
        CompiledTritonKernels.cache_clear()
        PyCodeCache.cache_clear()

        def forbidden(*args, **kwargs):
            raise AssertionError('a frozen cached forward must not compile or benchmark')

        fresh = Forward(7)
        with monkeypatch.context() as loading:
            loading.setattr(aot, 'aot_forward', forbidden)
            loading.setattr(CompilePool, 'submit_payload', forbidden)
            loading.setattr(triton, 'compile', forbidden)
            loading.setattr(CachingAutotuner, 'benchmark_all_configs', forbidden)
            restored = cache.prepare_many({'forward': fresh}, signatures)['forward']
        assert restored.cache_hit and restored.artifact.guard_check(fresh)
        torch.testing.assert_close(fresh.sums, torch.full_like(fresh.sums, 8 * 4096), rtol=0, atol=0)

        changed = Forward(7)
        changed.source = changed.source.clone()
        with pytest.raises((BindingMismatch, RuntimeError), match='aliases|GuardManager|guards'):
            restored.artifact(changed)
        changed = Forward(7)
        changed.negate = True
        assert not restored.artifact.guard_check(changed)
        restored.prepare_capture()
        restored(); stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            restored()
        try:
            for value, live in ((2, 128), (-3, 71), (4, 0), (1, 128)):
                fresh.source.fill_(value)
                fresh.count.fill_(live)
                fresh.output.fill_(float('nan'))
                fresh.sums.fill_(float('nan'))
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); stream.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                expected = torch.zeros(128)
                expected[:live] = value * 8 * 4096
                torch.testing.assert_close(fresh.sums.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(fresh.output, fresh.source * 8, rtol=0, atol=0)
                torch.testing.assert_close(original.sums, torch.full_like(original.sums, 4 * 4096), rtol=0, atol=0)
        finally:
            stream.synchronize(); graph.reset()
