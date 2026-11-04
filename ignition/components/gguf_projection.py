# SPDX-License-Identifier: Apache-2.0

"""Prepared GGUF projections using QUactlize's external NVIDIA operators."""

import ctypes

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor
from .gguf_weights import GgufWeight


def projection_workspace_bytes(source, weight: GgufWeight):
    """Query bounded scratch without expanding weights or creating a CUDA handle."""
    from quactlize.cuda_compat import Problem, query

    if (len(weight.shape) != 2 or source.ndim != 2
            or source.shape[1] != weight.in_features or source.shape[0] < 1):
        raise ValueError("GGUF dense projection requires input [M,K] and weight [N,K]")
    if (source.device.type != 'cuda' or source.dtype not in (torch.bfloat16, torch.float16)
            or not source.is_contiguous()):
        raise ValueError("GGUF projection requires contiguous CUDA BF16 or FP16 input")
    if (weight.raw.device != source.device or weight.raw.dtype != torch.uint8
            or not weight.raw.is_contiguous()):
        raise ValueError("GGUF weight requires contiguous raw bytes on the input device")
    problem = Problem(1, ctypes.sizeof(Problem), weight.qtype, 0, 0,
                      int(source.dtype == torch.bfloat16), source.shape[0],
                      weight.out_features, weight.in_features, 1, 1, 1, source.device.index)
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

    def __init__(self, source, weight: GgufWeight, output, *, workspace):
        from quactlize.cuda_compat import PreparedMatmul

        needed = projection_workspace_bytes(source, weight)
        if (not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8
                or workspace.device != source.device or not workspace.is_contiguous()
                or workspace.nbytes < needed):
            raise ValueError(f"GGUF projection requires at least {needed} caller-owned workspace bytes")
        self.raw, self.output, self.workspace = weight.raw, output, workspace
        self.input_shape, self.input_dtype, self.input_device = tuple(source.shape), source.dtype, source.device
        self.owner = PreparedMatmul(qtype=weight.qtype, n=weight.out_features,
            k=weight.in_features, input=source, output=output, raw=weight.raw, workspace=workspace)
        # Flare binds attributes named `handle` as process-local integers when
        # compiling forwards in metadata-only workers and restoring artifacts.
        self.handle = self.owner.native_handle
        self.device_error = self.owner.device_error
        record_numeric_tensor(self.raw)

    def __getstate__(self):
        # Workers trace only the opaque operation, tensors and integer binding.
        # The owning rank retains the native handle through graph export/replay.
        return {key: value for key, value in vars(self).items() if key != 'owner'}

    def __call__(self, source):
        from quactlize.cuda_compat import matmul

        if (tuple(source.shape) != self.input_shape or source.dtype != self.input_dtype
                or source.device != self.input_device or not source.is_contiguous()):
            raise ValueError('input differs from prepared GGUF projection geometry, dtype or device')
        matmul(source, self.raw, None, None, None, None, self.output,
               self.workspace, self.device_error, self.handle)
        return self.output
