# SPDX-License-Identifier: Apache-2.0

"""Prepare actlizeLA recurrence around Flare's convolution and state contract."""

import torch

from ignition.preparation.scratch import ScratchArena


class PreparedActlizeGDN:
    def __init__(self, library, *, module, scratch, state, metadata, mode, records, workspace):
        from actlize_la.serving import gdn_workspace_specs

        if not isinstance(workspace, ScratchArena):
            raise TypeError('actlizeLA GDN requires a shared preparation scratch arena')
        self.library, self.state, self.metadata = library, state, metadata
        self.mode = 'verify' if records is not None else mode
        rows = metadata.query_start_loc.numel() - 1
        width = (records.delta.shape[2] if records is not None else
                 scratch.convolved_qkv.shape[0] if mode == 'prefill' else 1)
        specs = gdn_workspace_specs(rows, width, module.num_k_heads, module.num_v_heads,
                                    mode=self.mode, target=library.target)
        device = state.recurrent.device
        self.work = {'errors': torch.zeros(rows, dtype=torch.int32, device=device)}
        if records is not None:
            self.work.update(delta=records.delta[:rows], keys=records.keys[:rows], decays=records.decay[:rows],
                recorded_lengths=torch.zeros(rows, dtype=torch.int32, device=device),
                recorded_slots=torch.full((rows,), -1, dtype=torch.int64, device=device))
        transient = {name: spec for name, spec in specs.items() if name not in self.work}
        workspace.views(*transient.values())
        self.operation = None
        self.snapshot = self.commit = None
        self.device_errors = (self.work['errors'],)

        def bind():
            from actlize_la.serving.flare import bind_gdn

            self.work.update(zip(transient, workspace.views(*transient.values()), strict=True))
            self.operation = bind_gdn(library, mixed_qkv=scratch.convolved_qkv,
                a=scratch.a, b=scratch.b, A_log=module.A_log, dt_bias=module.dt_bias,
                state=state.recurrent, metadata=metadata, output=scratch.recurrence,
                workspace=self.work, mode=self.mode, width=width,
                key_heads=module.num_k_heads, value_heads=module.num_v_heads)

        workspace.register_finalizer(bind)

    def prepare_commit(self, counts):
        from actlize_la.serving import PreparedConvHistory
        from actlize_la.serving.flare import bind_gdn_commit

        if self.mode != 'verify' or self.commit is not None:
            raise ValueError('actlizeLA verification binds its consumed-prefix commit exactly once')
        rows = self.metadata.query_start_loc.numel() - 1
        device = self.state.recurrent.device
        errors = tuple(torch.zeros(rows, dtype=torch.int32, device=device) for _ in range(3))
        snapshot = torch.zeros((rows, *self.state.convolution.shape[1:]),
                               dtype=torch.bfloat16, device=device)
        stamps = torch.full((rows,), -1, dtype=torch.int64, device=device)
        history = dict(state=self.state.convolution, snapshot=snapshot,
            slots=self.metadata.convolution_slots, recorded_slots=stamps,
            live=self.metadata.num_seqs, consumed_prefix=counts)
        self.snapshot = PreparedConvHistory(self.library, **history, errors=errors[0], restore=False)
        self.commit = ActlizeGDNCommit(
            bind_gdn_commit(self.library, state=self.state.recurrent, metadata=self.metadata,
                records=self.work, consumed_prefix=counts, errors=errors[1]),
            PreparedConvHistory(self.library, **history, errors=errors[2], restore=True))
        self.device_errors += errors
        return self.commit

    def before_convolution(self):
        if self.mode == 'verify':
            if self.snapshot is None:
                raise RuntimeError('prepare the verification commit before GDN warmup')
            self.snapshot()

    def __call__(self):
        if self.operation is None:
            raise RuntimeError('finalize actlizeLA scratch bindings before GDN warmup')
        self.operation()


class ActlizeGDNCommit:
    def __init__(self, recurrence, convolution):
        self.recurrence, self.convolution = recurrence, convolution

    def __call__(self):
        self.recurrence()
        # Positive prefixes remain encoded for the next ShortConv call. Only
        # zero consumption restores the history from before this verification.
        self.convolution()


def prepare_layer_commits(backing, states, records, metadata, counts, operations):
    """Retain the fused Flare commit or bind every selected external layer."""
    from .gdn_recovery import PreparedGDNCommit

    if set(operations) != set(states) or set(states) != set(records.layers):
        raise ValueError('GDN commit requires an operator and records for every retained layer')
    external = [operations[layer].actlize for layer in records.layers]
    if any(operation is not None for operation in external):
        if not all(operation is not None for operation in external):
            raise ValueError('one verification interval must use one GDN record implementation')
        for layer, operation in zip(records.layers, external, strict=True):
            if operation.state.recurrent is not states[layer]:
                raise ValueError('external GDN commit must retain its verification state')
        return tuple(operation.prepare_commit(counts) for operation in external)
    heads = records.delta.shape[2]
    return (PreparedGDNCommit(backing, states, records, metadata, counts,
                             key_heads=heads // 3, value_heads=heads),)
