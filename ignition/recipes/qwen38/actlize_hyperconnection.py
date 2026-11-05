# SPDX-License-Identifier: Apache-2.0

"""Qwen four-stream mixing using external actlizeLA helpers and dense projections."""

import torch

from ignition.components.actlize_pointwise import PreparedActlizePointwise
from ignition.components.linear import UnquantizedProjection


class PreparedActlizeHyperconnection:
    def __init__(self, module, scratch, *, priming_state, session, name, runtime=None, coordinator=None):
        from .hyperconnection import packed_down_projection

        if module.hc_count != 4 or module.hidden_size != 2560:
            raise ValueError('actlizeLA Qwen mixer requires four streams of width 2560')
        if runtime is not None or module.input_mix_weight_down.weight.shape[0] != 320:
            raise ValueError('actlizeLA Qwen mixer currently requires unsharded GGUF weights')
        self.library, self.scratch = session.actlize_helpers, scratch
        self.tokens = scratch.normalized.shape[0]
        self.eps = module.hc_norm.eps
        self.norm_gain = (1. + module.hc_norm.weight.detach().float()).contiguous()
        if self.norm_gain.shape != (10240,) or priming_state.shape != (self.tokens, 10240):
            raise ValueError('actlizeLA Qwen mixer requires full 10240-wide residual and norm weights')
        self.norm = self._pointwise('grouped_norm', x=priming_state,
            output=scratch.normalized, weight=self.norm_gain, groups=4, epsilon=self.eps)
        self.activation = self._pointwise('hc_scaled_silu', x=scratch.projected_down,
            output=scratch.bottleneck, groups=1)
        self.mean = self._pointwise('hc_gate_mean', x=scratch.normalized,
            gate=scratch.gate_workspace, output=scratch.block_input, groups=4)
        self.injection = None if module.block_inject_weight is None else scratch.injection_logits
        self.down = UnquantizedProjection(scratch.normalized, packed_down_projection(module),
            scratch.projected_down if self.injection is None else scratch.down_and_injection,
            session=session, name=name + '.down')
        self.up = UnquantizedProjection(scratch.bottleneck, module.input_mix_weight_up.weight,
            scratch.gate_workspace, session=session, name=name + '.up')
        self.combine_op = None if self.injection is None else self._pointwise('hc_combine',
            x=priming_state, gate=self.injection, value=scratch.block_input,
            output=scratch.gate_workspace, groups=4)
        self.combine_norm_op = None

    def _pointwise(self, operation, **kwargs):
        error = torch.zeros(self.tokens, dtype=torch.int32, device=self.scratch.normalized.device)
        return PreparedActlizePointwise(self.library, operation, errors=error, **kwargs)

    def prepare_combine_norm(self, following, *, priming_state, session, name):
        if self.injection is None:
            raise ValueError('the final hyperconnection mixer has no residual injection')
        self.following = following
        self.combine_norm_op = self._pointwise('hc_combine_norm', x=priming_state,
            gate=self.injection, value=self.scratch.block_input, groups=4,
            weight=following.norm_gain, epsilon=following.eps,
            combined=self.scratch.gate_workspace, output=following.scratch.normalized)

    def mix(self, residual, *, pre_normalized=False):
        s = self.scratch
        if not pre_normalized:
            self.norm(x=residual.view(self.tokens, 4, 2560))
        self.down(s.normalized)
        self.activation()
        self.up(s.bottleneck)
        self.mean()
        return s.block_input

    def combine(self, residual, block_output):
        if self.combine_op is None:
            raise ValueError('the final hyperconnection mixer has no residual injection')
        self.combine_op(x=residual.view(self.tokens, 4, 2560), value=block_output.view(self.tokens, 1, 2560))
        # The external operation requires disjoint writes. Reuse expired gate
        # scratch, then update the residual before the following mix.
        residual.copy_(self.scratch.gate_workspace)
        return residual

    def combine_norm(self, residual, block_output):
        if self.combine_norm_op is None:
            raise RuntimeError('prepare the following mixer before combined normalization')
        self.combine_norm_op(x=residual.view(self.tokens, 4, 2560), value=block_output.view(self.tokens, 1, 2560))
        residual.copy_(self.scratch.gate_workspace)
        return residual

    def trace_intermediates(self):
        fields = ('normalized', 'projected_down', 'bottleneck')
        return {field: getattr(self.scratch, field) for field in
                (*fields, *(() if self.injection is None else ('injection_logits',)))}

    @property
    def device_errors(self):
        operators = (self.norm, self.activation, self.mean, self.combine_op, self.combine_norm_op)
        return tuple(flag for op in operators if op is not None for flag in op.device_errors)
