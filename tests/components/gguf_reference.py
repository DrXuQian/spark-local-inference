# SPDX-License-Identifier: Apache-2.0

"""Test-only decoding, including the llama.cpp PPU extension absent upstream."""

import gguf
import numpy as np


def dequantize(raw, kind):
    if int(kind) != 42:
        return gguf.dequantize(raw, kind)
    # DrXuQian/llama.cpp 356a231c1, dequantize_row_q2_0: FP16 scale,
    # then sixteen bytes holding four adjacent two-bit codes each.
    blocks = raw.reshape(-1, 18)
    scales = blocks[:, :2].copy().view('<f2').astype(np.float32)
    codes = (blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    values = (codes.reshape(-1, 64).astype(np.float32) - 1) * scales
    return values.reshape(*raw.shape[:-1], raw.shape[-1] // 18 * 64)
