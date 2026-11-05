# SPDX-License-Identifier: Apache-2.0

"""Installed QSA, real Flare pages, worker rebinding and graph replay; no b12x."""

from contextlib import nullcontext
import os

import pytest
import torch

pytest.importorskip('actlize_la.serving')

from ignition.preparation.actlize import ActlizeLibrarySpec
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.external_session import ExternalPreparationSession
from ignition.preparation.scratch import ScratchArena
from qualify.actlize_qsa_export import Fixture


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='requires FLARE_RUN_CUDA_TESTS=1 and an assigned PPU/CUDA device')


@pytest.mark.parametrize('compilation', ['none', 'catalog'])
@pytest.mark.parametrize('rows', [1, 7, 128])
@torch.no_grad()
def test_qsa_pages_prefix_recovery_and_catalog_rebinding(tmp_path, monkeypatch, rows, compilation):
    from ignition.preparation import aot
    from ignition.preparation.compile_pool import CompilePool

    target = os.environ.get('FLARE_ACTLIZE_TARGET')
    if target is None:
        pytest.skip('select FLARE_ACTLIZE_TARGET matching the installed library')
    path = os.environ.get('FLARE_ACTLIZE_LIBRARY')
    spec = (ActlizeLibrarySpec.installed(target=target) if path is None
            else ActlizeLibrarySpec.read(path, target=target))
    stream = torch.cuda.Stream(device=0)
    previous_handle = None
    def forbidden(*args, **kwargs):
        raise AssertionError('a restored QSA catalog must not compile again')

    for generation in range(2 if compilation == 'catalog' else 1):
        graphs, compiled = {}, {}
        library = spec.load('cuda:0')
        f = Fixture(library, rows=rows)
        assert f.operation.handle != previous_handle
        previous_handle = f.operation.handle
        try:
            with ExternalPreparationSession(device='cuda:0', checkpoint='qsa-component',
                    namespace={'qualification': 'qsa-pages-v1', 'operator': spec.identity()},
                    compile_workers=1) as session:
                session.register_scratch(ScratchArena('cuda:0', overlap=False))
                session.tune()
                with monkeypatch.context() if generation else nullcontext() as cached:
                    if generation:
                        cached.setattr(aot, 'aot_forward', forbidden)
                        cached.setattr(CompilePool, 'submit_payload', forbidden)
                    graphs, compiled = capture_forwards({'qsa': f.forward},
                        {'qsa': GraphSignature('prefill', 2, rows)}, session=session, device='cuda:0',
                        stream=stream, pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path,
                        compile_decoder=compilation == 'catalog', retain_graphs=True, report_catalog=False)
                if compiled:
                    assert compiled['qsa'].cache_hit == bool(generation)
                session.freeze()

                def replay_and_check():
                    before = {name: getattr(f.state, name).cpu().clone() for name in f.state.__dataclass_fields__}
                    inputs = {name: value.cpu().clone() for name, value in f.inputs.items()}
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    graphs['qsa'].replay(); stream.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                    f.check(before)
                    for name, value in inputs.items():
                        torch.testing.assert_close(f.inputs[name].cpu(), value, rtol=0, atol=0)
                    assert (f.work['selected'][f.width:] == -1).all()

                with torch.cuda.stream(stream):
                    for case, context in enumerate((0, 3, 63, 64, 2051)):
                        f.seed(context, 1107 + case)
                        f.interval(context, rows, 1)
                        replay_and_check()
                    # Use the same captured graph for a shorter replacement
                    # interval after accepting each possible consumed prefix.
                    width = min(rows, 7)
                    for accepted in range(width + 1):
                        f.seed(63, 1200 + accepted)
                        f.interval(63, width, 1)
                        replay_and_check()
                        start = 63 + accepted
                        for name in ('query', 'key', 'value', 'raw', 'index_query'):
                            f.data[name][start:start + width].mul_(-1)
                        f.interval(start, width, accepted)
                        replay_and_check()
                    f.seed(63, 1300)
                    f.interval(63, 0, 1)
                    replay_and_check()
        finally:
            stream.synchronize()
            for graph in graphs.values():
                graph.reset()
            compiled.clear()
