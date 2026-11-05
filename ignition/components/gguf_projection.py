# SPDX-License-Identifier: Apache-2.0

"""Prepared GGUF projections using QUactlize's external NVIDIA operators."""

import ctypes

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor
from .gguf_weights import GgufWeight


def projection_workspace_bytes(source, weight: GgufWeight, *, ids=None):
    """Query bounded scratch without expanding weights or creating a CUDA handle."""
    if weight.packed is not None:
        raise ValueError('canonical GGUF weights require the selected native PPU projection library')
    from quactlize.cuda_compat import Problem, query

    indexed = ids is not None
    if (len(weight.shape) != (3 if indexed else 2) or source.ndim not in ((2, 3) if indexed else (2,))
            or source.shape[-1] != weight.in_features or source.shape[0] < 1):
        raise ValueError("GGUF projection requires matching input and weight geometry")
    if (source.device.type != 'cuda' or source.dtype not in (torch.bfloat16, torch.float16)
            or not source.is_contiguous()):
        raise ValueError("GGUF projection requires contiguous CUDA BF16 or FP16 input")
    if (weight.raw.device != source.device or weight.raw.dtype != torch.uint8
            or not weight.raw.is_contiguous()):
        raise ValueError("GGUF weight requires contiguous raw bytes on the input device")
    if indexed and (ids.ndim != 2 or ids.shape[0] != source.shape[0] or ids.dtype != torch.int32
                    or ids.device != source.device or not ids.is_contiguous()
                    or (source.ndim == 3 and source.shape[1] != ids.shape[1])):
        raise ValueError('indexed GGUF projection requires contiguous int32 IDs [M,topk]')
    topk = ids.shape[1] if indexed else 1
    problem = Problem(1, ctypes.sizeof(Problem), weight.qtype, 0, int(indexed),
                      int(source.dtype == torch.bfloat16), source.shape[0],
                      weight.out_features, weight.in_features, weight.experts, topk,
                      source.shape[1] if source.ndim == 3 else 1, source.device.index)
    sizes = query(problem)
    if weight.raw.nbytes != sizes.raw_bytes:
        raise ValueError("GGUF raw weight size differs from its logical geometry")
    return sizes.workspace_bytes


class PreparedGgufProjection:
    """Keep weights packed and expand one projection into caller-owned scratch.

    Sequential projections may share the same fixed workspace. Concurrent
    projections require disjoint workspaces. Allocate and bind the whole shape
    catalog before warmup/capture, and retain this owner until replay drains.
    This compatibility component does not expose a fictitious b12x plan: its
    opaque QUactlize handle has its own ABI and cannot enter b12x overlap helpers.
    """

    def __init__(self, source, weight: GgufWeight, output, *, workspace, ids=None):
        from quactlize.cuda_compat import PreparedMatmul

        needed = projection_workspace_bytes(source, weight, ids=ids)
        if (not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8
                or workspace.device != source.device or not workspace.is_contiguous()
                or workspace.nbytes < needed):
            raise ValueError(f"GGUF projection requires at least {needed} caller-owned workspace bytes")
        self.raw, self.output, self.workspace = weight.raw, output, workspace
        self.ids = ids
        self.input_shape, self.input_dtype, self.input_device = tuple(source.shape), source.dtype, source.device
        self.owner = PreparedMatmul(qtype=weight.qtype, n=weight.out_features,
            k=weight.in_features, input=source, output=output, raw=weight.raw, workspace=workspace,
            ids=ids, experts=weight.experts)
        # Flare binds attributes named `handle` as process-local integers when
        # compiling forwards in metadata-only workers and restoring artifacts.
        self.handle = self.owner.native_handle
        self.device_error = self.owner.device_error
        self.device_errors = (self.device_error,)
        record_numeric_tensor(self.raw)

    def __getstate__(self):
        # Workers trace only the opaque operation, tensors and integer binding.
        # The owning rank retains the native handle through graph export/replay.
        return {key: value for key, value in vars(self).items() if key != 'owner'}

    def __call__(self, source, *, ids=None, output=None, device_error=None):
        from quactlize.cuda_compat import matmul

        if (tuple(source.shape) != self.input_shape or source.dtype != self.input_dtype
                or source.device != self.input_device or not source.is_contiguous()):
            raise ValueError('input differs from prepared GGUF projection geometry, dtype or device')
        ids = self.ids if ids is None else ids
        output = self.output if output is None else output
        device_error = self.device_error if device_error is None else device_error
        if ((ids is None) != (self.ids is None) or (ids is not None and
                (ids.shape != self.ids.shape or ids.dtype != torch.int32
                 or ids.device != self.input_device or not ids.is_contiguous()))):
            raise ValueError('routing IDs differ from prepared GGUF projection geometry')
        if (output.shape != self.output.shape or output.dtype != self.input_dtype
                or output.device != self.input_device or not output.is_contiguous()):
            raise ValueError('output differs from prepared GGUF projection geometry')
        if (device_error.numel() != 1 or device_error.dtype != torch.int32
                or device_error.device != self.input_device or not device_error.is_contiguous()):
            raise ValueError('device error requires one contiguous CUDA int32 element')
        matmul(source, self.raw, None, None, None, ids, output,
               self.workspace, device_error, self.handle)
        return output
