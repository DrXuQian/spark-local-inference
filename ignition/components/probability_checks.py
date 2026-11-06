# SPDX-License-Identifier: Apache-2.0

"""Exact FP32 classification for captured sampling validation."""

import torch


def finite_fp32(values):
    # The PPU LLVM13 backend cannot lower vector llvm.is.fpclass emitted by
    # newer Triton for isfinite. Integer classification stays in the fused
    # forward and preserves all FP32 classes, including signed zero and NaNs.
    if values.dtype != torch.float32:
        raise ValueError('sampling classification requires FP32 values')
    return (values.view(torch.int32) & 0x7f800000) != 0x7f800000


def finite_nonnegative_fp32(values):
    finite = finite_fp32(values)
    bits = values.view(torch.int32)
    # Both zero signs are nonnegative; negative subnormals are not.
    return finite & ((bits >= 0) | (bits == -2147483648))
