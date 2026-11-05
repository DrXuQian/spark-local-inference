# SPDX-License-Identifier: Apache-2.0

"""PLE contribution using external helpers over the existing staged row protocol."""

import torch

from ignition.components.actlize_pointwise import PreparedActlizePointwise
from ignition.components.linear import UnquantizedProjection


class PreparedActlizePLE:
    def __init__(self, module, projections, source, scratch, convolution_state, metadata, table, *,
                 mode, session, name, workspace=None, partitions=None, runtime=None, coordinator=None):
        from actlize_la.serving.flare import auxiliary_rows, bind_shortconv
        from ignition.components.ngram_delivery import PreparedPLELookup
        from .ple import _validate_operands

        sequences = _validate_operands(module, projections, source, scratch, convolution_state, metadata, table, mode)
        if partitions is not None or runtime is not None:
            raise ValueError('external PLE currently requires unpartitioned single-device GGUF execution')
        if table.encoding != 'GGUF_IQ4_NL' or getattr(session, 'quactlize_helpers', None) is None:
            raise ValueError('external PLE requires original GGUF IQ4_NL rows and QUactlize helpers')
        if module.conv1d.weight.dtype != torch.bfloat16:
            raise ValueError('external PLE convolution requires exactly represented BF16 checkpoint weights')
        self.source, self.scratch, self.metadata = source, scratch, metadata
        self.library, self.tokens = session.actlize_helpers, source.shape[0]
        self.rows = auxiliary_rows(metadata, self.tokens)
        self.embedding_lookup = PreparedPLELookup(table, metadata, scratch.embedding, table.layout.caps,
            session=session, name=name + '.embedding', runtime=runtime, coordinator=coordinator)
        self.embedding_output = scratch.embedding
        self.key_projection = UnquantizedProjection(scratch.embedding, projections['key_proj'], scratch.key,
            session=session, name=name + '.key')
        self.value_projection = UnquantizedProjection(scratch.embedding, projections['value_proj'], scratch.value,
            session=session, name=name + '.value')
        # Key/query norms expire at the gate; their storage then serves norm_conv
        # and the convolution result. No value survives across these aliases.
        self.key_norm = self._norm(module.norm_key, scratch.key, scratch.normalized)
        self.query_norm = self._norm(module.norm_query, source.view(self.tokens, 10240), scratch.convolved)
        self.gate = self._pointwise('ple_gate', x=scratch.normalized, gate=scratch.convolved,
            value=scratch.value, output=scratch.gated)
        self.conv_norm = self._norm(module.norm_conv, scratch.gated, scratch.normalized)
        self.convolution_errors = torch.zeros(sequences, dtype=torch.int32, device=source.device)
        self.convolution = bind_shortconv(self.library, x=scratch.normalized,
            weight=module.conv1d.weight.squeeze(1), state=convolution_state, metadata=metadata,
            output=scratch.convolved, errors=self.convolution_errors, dilation=3, mode=mode)
        self.contribution = self._pointwise('ple_contribution', x=scratch.gated,
            value=scratch.convolved, output=scratch.output,
            rows=auxiliary_rows(metadata, self.tokens, upstream_errors=self.convolution_errors))

    def _pointwise(self, operation, **kwargs):
        error = torch.zeros(self.tokens, dtype=torch.int32, device=self.source.device)
        kwargs.setdefault('rows', self.rows)
        return PreparedActlizePointwise(self.library, operation, groups=4, errors=error, **kwargs)

    def _norm(self, module, source, output):
        return self._pointwise('grouped_norm', x=source, output=output,
            weight=module.weight, epsilon=module.eps, zero_centered=True)

    def __call__(self, source):
        if source is not self.source and (source.data_ptr() != self.source.data_ptr() or source.shape != self.source.shape):
            raise ValueError('prepared PLE requires its fixed source storage')
        self.embedding_lookup(source)
        self.run_contribution()
        return self.scratch.output

    def run_contribution(self):
        self.key_projection(self.embedding_output)
        self.value_projection(self.embedding_output)
        self.key_norm()
        self.query_norm()
        self.gate()
        self.conv_norm()
        self.convolution()
        self.contribution()

    @property
    def device_errors(self):
        from ignition.components.operator_errors import device_errors

        return (self.convolution_errors, *device_errors(self.embedding_lookup,
            self.key_norm, self.query_norm, self.gate, self.conv_norm, self.contribution))
