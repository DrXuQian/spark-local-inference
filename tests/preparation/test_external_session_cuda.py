# SPDX-License-Identifier: Apache-2.0

"""External preparation through the real compiler catalog and state-transfer graphs."""

import os

import pytest
import torch

from ignition.components import state
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.external_session import ExternalPreparationSession
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                               reason='requires an assigned device and FLARE_RUN_CUDA_TESTS=1')


class StateForward:
    def __init__(self, rows, sources, hidden, stored, workspace):
        self.sources, self.hidden, self.stored = sources, hidden, stored
        self.rows = torch.arange(rows, dtype=torch.int64, device='cuda:0')
        self.active = torch.ones(rows, dtype=torch.int32, device='cuda:0')
        self.observed = torch.empty((rows, 35), dtype=torch.bfloat16, device='cuda:0')
        self.shapes = tuple(((rows, 35), value.dtype) for value in sources)
        self.prepared = False
        workspace.require_views(*self.shapes)

        def bind():
            self.gathered = workspace.views(*self.shapes)
            self.prepared = True

        workspace.register_finalizer(bind)

    def __call__(self):
        if not self.prepared:
            raise RuntimeError('state-transfer scratch was not finalized')
        state.gather_rows_three(self.sources, self.rows, self.gathered)
        state.save_hidden(self.gathered[0], self.hidden, self.rows, self.active)
        state.scatter_rows(self.gathered[1], self.rows, self.stored, self.active)
        state.gather_rows(self.hidden, self.rows, self.observed)


@torch.no_grad()
def test_external_session_catalog_replays_alternating_state_transfers(tmp_path):
    stream = torch.cuda.Stream(device=0)
    graphs, compiled = {}, {}
    with ExternalPreparationSession(device='cuda:0', checkpoint='state-transfer-qualification',
            namespace={'qualification': 'external-state-catalog'}, compile_workers=1) as session:
        workspace = ScratchArena('cuda:0', overlap=False)
        sources = tuple(torch.empty_strided((9, 35), (48, 1), dtype=dtype, device='cuda:0')
                        for dtype in (torch.bfloat16, torch.float32, torch.int32))
        for source in sources:
            source.zero_()
        hidden = torch.zeros_like(sources[0])
        stored = torch.zeros_like(sources[1])
        forwards = {f'decode-{rows}': StateForward(rows, sources, hidden, stored, workspace) for rows in (1, 7)}
        signatures = {name: GraphSignature('decode', forward.rows.numel(), 1) for name, forward in forwards.items()}
        session.register_scratch(workspace)
        session.tune()
        assert workspace.frozen and all(forward.prepared for forward in forwards.values())
        try:
            graphs, compiled = capture_forwards(forwards, signatures, session=session,
                device='cuda:0', stream=stream, pool=torch.cuda.graph_pool_handle(),
                cache_root=tmp_path, retain_graphs=True, report_catalog=False)
            assert set(compiled) == set(forwards)
            session.freeze()
            indices = torch.tensor([8, 3, 0, 1, 4, 5, 2], dtype=torch.int64)
            expected_hidden = torch.zeros((9, 35), dtype=torch.bfloat16)
            expected_stored = torch.zeros((9, 35), dtype=torch.float32)
            with torch.cuda.stream(stream):
                hidden.zero_(); stored.zero_()
                for step, (rows, live) in enumerate(((7, 7), (1, 0), (7, 3), (1, 1), (7, 0), (7, 7))):
                    forward = forwards[f'decode-{rows}']
                    chosen = indices.roll(step)[:rows]
                    active = torch.arange(rows) < live
                    values = (torch.arange(9 * 35).reshape(9, 35) % 113 + step).to(torch.bfloat16)
                    for index, source in enumerate(sources):
                        source.copy_((values + index).to(source.dtype))
                    forward.rows.copy_(chosen)
                    forward.active.copy_(active.to(torch.int32))
                    forward.observed.fill_(float('nan'))
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    graphs[f'decode-{rows}'].replay()
                    stream.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                    selected = chosen[active]
                    # Hidden checkpoints reserve slot zero. Generic row scatter
                    # also serves request-indexed arrays, where row zero is valid.
                    saved = selected[selected > 0]
                    expected_hidden[saved] = values[saved]
                    expected_stored[selected] = values.float()[selected] + 1
                    torch.testing.assert_close(hidden.cpu(), expected_hidden, rtol=0, atol=0)
                    torch.testing.assert_close(stored.cpu(), expected_stored, rtol=0, atol=0)
                    torch.testing.assert_close(forward.observed.cpu(), expected_hidden[chosen], rtol=0, atol=0)
                    for index, gathered in enumerate(forward.gathered):
                        torch.testing.assert_close(gathered.cpu(), (values[chosen] + index).to(gathered.dtype),
                                                   rtol=0, atol=0)
        finally:
            stream.synchronize()
            for graph in graphs.values():
                graph.reset()
