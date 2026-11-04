# SPDX-License-Identifier: Apache-2.0

"""Bind external QSA to prepared projections and BF16 Flare cache storage."""

import torch

from ignition.preparation.scratch import ScratchArena


class PreparedActlizeQSA:
    def __init__(self, library, *, scratch, state, metadata, rotary, live_rows,
                 query_weight, key_weight, epsilon, workspace):
        from actlize_la.serving import qsa_workspace_specs

        if not isinstance(workspace, ScratchArena):
            raise TypeError('actlizeLA QSA requires a shared preparation scratch arena')
        self.library, self.scratch, self.state = library, scratch, state
        self.metadata, self.rotary, self.live_rows = metadata, rotary, live_rows
        self.query_weight, self.key_weight, self.epsilon = query_weight, key_weight, epsilon
        self.workspace = workspace
        tokens, heads = scratch.query.shape[:2]
        rows = metadata.main_block_table.shape[0]
        specs = qsa_workspace_specs(tokens, rows, heads)
        self.work = dict(output=scratch.attended, selected=scratch.selected_positions,
            errors=torch.zeros(rows, dtype=torch.int32, device=scratch.query.device))
        self.transient = {name: spec for name, spec in specs.items() if name not in self.work}
        workspace.views(*self.transient.values())
        self.device_errors = (self.work['errors'],)
        self.operation = None
        workspace.register_finalizer(self.bind)

    def bind(self):
        from actlize_la.serving.flare import bind_qsa

        s = self.scratch
        tokens, kv_heads = s.key.shape[:2]
        self.work.update(zip(self.transient, self.workspace.views(*self.transient.values()), strict=True))
        self.operation = bind_qsa(self.library, query=s.query, key=s.key,
            value=s.projected_v.view(tokens, kv_heads, 256),
            index_query=s.projected_index[:, :512].view(tokens, 4, 128),
            raw_index_key=s.projected_index[:, 512:], index_q_norm_weight=self.query_weight,
            index_k_norm_weight=self.key_weight, rotary=self.rotary, state=self.state,
            metadata=self.metadata, live_rows=self.live_rows, workspace=self.work,
            norm_epsilon=self.epsilon)

    def rebind(self, state):
        self.state = state
        if self.operation is not None:
            self.bind()

    def __call__(self):
        if self.operation is None:
            raise RuntimeError('finalize actlizeLA scratch bindings before QSA warmup')
        self.operation()
        return self.work['output']
