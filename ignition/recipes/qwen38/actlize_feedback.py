# SPDX-License-Identifier: Apache-2.0

"""Full-width MTP normalization and BF16 projections with prepared external operators."""

import torch

from ignition.components.actlize_pointwise import PreparedActlizePointwise
from ignition.components.linear import UnquantizedProjection
from ignition.preparation.scratch import ScratchArena


class PreparedActlizeMTPFeedback:
    def __init__(self, module, token_embedding, target_hidden, output, workspace, *, session, name):
        if not isinstance(workspace, ScratchArena) or getattr(session, 'dense_blas', None) is None:
            raise ValueError('external MTP feedback requires shared scratch and explicitly prepared dense BLAS')
        tokens, device = token_embedding.shape[0], token_embedding.device
        for tensor, shape in ((token_embedding, (tokens, 2560)), (target_hidden, (tokens, 10240)),
                              (output, (tokens, 10240))):
            if (tensor.shape != shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
                    or tensor.device != device or device.type != 'cuda'):
                raise ValueError('MTP feedback requires contiguous device BF16 token/residual storage')
        begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
        for tensor in (token_embedding, target_hidden):
            if begin < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < end:
                raise ValueError('MTP feedback output must not alias token or target-hidden input')
        self.output, self.tokens = output, tokens
        self.token_error = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.hidden_error = torch.zeros_like(self.token_error)
        self.device_errors = (self.token_error, self.hidden_error)
        specs = tuple(((tokens, width), torch.bfloat16) for width in (2560, 10240, 2560))
        workspace.require_views(*specs)
        self.token_norm = self.hidden_norm = self.token_projection = self.hidden_projection = None

        def bind():
            self.normalized_token, self.normalized_hidden, self.projected_token = workspace.views(*specs)
            common = dict(groups=1, zero_centered=True, epsilon=module.config.rms_norm_eps)
            self.token_norm = PreparedActlizePointwise(session.actlize_helpers, 'grouped_norm',
                x=token_embedding, weight=module.pre_fc_norm_embedding.weight,
                output=self.normalized_token, errors=self.token_error, **common)
            # Normalize ALL 10240 elements before viewing the four projection rows.
            self.hidden_norm = PreparedActlizePointwise(session.actlize_helpers, 'grouped_norm',
                x=target_hidden, weight=module.pre_fc_norm_hidden.weight,
                output=self.normalized_hidden, errors=self.hidden_error, **common)
            self.token_projection = UnquantizedProjection(self.normalized_token, module.fc_embedding.weight,
                self.projected_token, session=session, name=name + '.embedding_fc')
            self.hidden_projection = UnquantizedProjection(self.normalized_hidden.view(-1, 2560), module.fc_hidden.weight,
                output.view(-1, 2560), session=session, name=name + '.hidden_fc')

        workspace.register_finalizer(bind)

    def __call__(self):
        if self.token_norm is None:
            raise RuntimeError('finalize MTP feedback scratch before warmup')
        self.token_norm()
        self.hidden_norm()
        self.token_projection(self.normalized_token)
        self.hidden_projection(self.normalized_hidden.view(-1, 2560))
        self.output.view(self.tokens, 4, 2560).add_(self.projected_token[:, None, :])
        return self.output
