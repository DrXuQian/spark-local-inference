# SPDX-License-Identifier: Apache-2.0

"""Prepared QUactlize serving helpers with caller-owned output and status."""

import math
import weakref

import torch


class PreparedQuactlizeHelpers:
    """Keep the DSO in the capture process and rebind its handle in AOT workers."""

    def __init__(self, owner):
        self.owner = owner
        self.handle = owner.key
        self._close = weakref.finalize(self, owner.close)

    def __getstate__(self):
        return {'handle': self.handle}

    def close(self):
        # PreparedModel closes owners after draining work and resetting graphs.
        self._close()

    def swiglu(self, gate_up, output, *, round_silu=True):
        from quactlize.serving import swiglu

        swiglu(gate_up, None, output, round_silu, self.handle)

    def gather_rows(self, raw, ids, output, count, error, qtype, table_rows):
        from quactlize.serving import gather_rows

        gather_rows(raw, ids, output, count, error, qtype, table_rows, self.handle)


class PreparedQuactlizeRouter:
    def __init__(self, logits, scratch, *, session, name, score_func='softmax',
                 renormalize=True, routed_scaling_factor=1., correction_bias=None,
                 image_correction_bias=None, image_mask=None, num_valid_tokens=None):
        if (score_func != 'softmax' or not renormalize or correction_bias is not None
                or image_correction_bias is not None or image_mask is not None):
            raise ValueError('QUactlize routing requires selected-logit softmax without correction biases')
        if (not math.isfinite(routed_scaling_factor)
                or abs(routed_scaling_factor) > torch.finfo(torch.float32).max):
            raise ValueError('QUactlize route multiplier must be finite FP32')
        if (logits.ndim != 2 or logits.dtype not in (torch.bfloat16, torch.float32)
                or logits.stride(1) != 1 or logits.stride(0) < logits.shape[1]):
            raise ValueError('QUactlize router requires row-strided BF16/FP32 logits')
        shape = scratch.ids.shape
        if (len(shape) != 2 or shape[0] != logits.shape[0]
                or not 0 < shape[1] <= min(32, logits.shape[1]) or logits.shape[1] > 1024):
            raise ValueError('QUactlize router requires matching rows, top_k <= 32 and experts <= 1024')
        for value, dtype in ((scratch.selected_logits, torch.float32),
                             (scratch.ids, torch.int32), (scratch.weights, torch.float32)):
            if (value.shape != shape or value.dtype != dtype or value.device != logits.device
                    or value.stride(1) != 1 or value.stride(0) < shape[1]):
                raise ValueError('QUactlize router output geometry, dtype or device differs')
        if num_valid_tokens is not None and (num_valid_tokens.numel() != 1
                or num_valid_tokens.dtype != torch.int32 or num_valid_tokens.device != logits.device):
            raise ValueError('QUactlize router live count requires one device int32')
        self.helpers, self.scratch = session.quactlize_helpers, scratch
        self.shape, self.stride, self.dtype, self.device = logits.shape, logits.stride(), logits.dtype, logits.device
        self.count, self.multiplier = num_valid_tokens, float(routed_scaling_factor)
        self.error = torch.zeros((), dtype=torch.int32, device=logits.device)
        self.device_errors = (self.error,)
        # GGUF experts consume explicit IDs/weights, with no b12x plan or overlap binding.
        self.plan = None

    def __call__(self, logits):
        from quactlize.serving import router

        if (logits.shape != self.shape or logits.stride() != self.stride
                or logits.dtype != self.dtype or logits.device != self.device):
            raise ValueError('router logits must preserve the prepared geometry, dtype and device')
        router(logits, self.scratch.selected_logits, self.scratch.ids, self.scratch.weights,
               self.error, self.count, self.multiplier, self.helpers.handle)
