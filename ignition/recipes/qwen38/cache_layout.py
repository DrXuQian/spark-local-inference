# SPDX-License-Identifier: Apache-2.0

"""Physical Qwen QSA cache geometry shared by loading, snapshots and operators."""

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class CacheRequirements:
    kv_dtype: torch.dtype
    main_k_page_shape: tuple[int, int, int]
    main_v_page_shape: tuple[int, int, int]
    compressed_page_shape: tuple[int, int]
    raw_k_ring_shape: tuple[int, int]
    raw_logical_positions_shape: tuple[int]
    raw_rope_positions_shape: tuple[int, int]
    raw_ring_capacity: int
    selection_width: int = 2051

    @property
    def page_nbytes(self):
        return (2 * prod(self.main_k_page_shape) * self.kv_dtype.itemsize
                + prod(self.compressed_page_shape) * torch.bfloat16.itemsize)


def cache_requirements(*, main_page_size, kv_heads=2, position_axes=3,
                       max_speculative_tokens=0, kv_dtype=torch.bfloat16):
    """Retain ratio-four/D256 cache storage without loading b12x kernels.

    The raw ring must cover a compressed group plus the rejected draft horizon,
    and divide a physical page. Tags and rotary positions remain separate state
    fields; this function does not alias their storage with compressed keys.
    """
    if (type(main_page_size) is not int or main_page_size <= 0 or main_page_size % 4
            or type(kv_heads) is not int or kv_heads <= 0):
        raise ValueError('Qwen cache needs positive KV heads and a page size divisible by four')
    if type(position_axes) is not int or position_axes not in (1, 3):
        raise ValueError('Qwen rotary metadata has one or three position axes')
    if type(max_speculative_tokens) is not int or max_speculative_tokens < 0:
        raise ValueError('Qwen cache draft capacity must be a nonnegative integer')
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError('Qwen main KV cache requires BF16 or FP8 E4M3FN')
    minimum = ((4 + max_speculative_tokens + 3) // 4) * 4
    ring = next((value for value in range(minimum, main_page_size + 1, 4)
                 if main_page_size % value == 0), None)
    if ring is None:
        raise ValueError('Qwen raw ring must divide its page and cover speculative history')
    main_shape = (main_page_size, kv_heads, 256)
    return CacheRequirements(kv_dtype, main_shape, main_shape, (main_page_size // 4, 128),
                             (ring, 128), (ring,), (ring, position_axes), ring)
