# SPDX-License-Identifier: Apache-2.0

"""Copy native-precision rows into Flare's paged cache; -1 leaves storage untouched."""

import triton
import triton.language as tl


@triton.jit
def write_rows(Key, Value, Slots, KeyCache, ValueCache, Error,
               key_row: tl.constexpr, key_head: tl.constexpr, value_row: tl.constexpr, value_head: tl.constexpr,
               slot_stride: tl.constexpr, kp: tl.constexpr, kt: tl.constexpr, kh: tl.constexpr,
               vp: tl.constexpr, vt: tl.constexpr, vh: tl.constexpr,
               HEADS: tl.constexpr, DIM: tl.constexpr, PAGE: tl.constexpr, CAPACITY: tl.constexpr,
               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row * slot_stride).to(tl.int64)
    if slot != -1:
        if (slot < 0) | (slot >= CAPACITY):
            tl.atomic_or(Error, 1)
        else:
            index = tl.arange(0, BLOCK)
            head, column = index // DIM, index % DIM
            key = tl.load(Key + row * key_row + head * key_head + column, index < HEADS * DIM, other=0)
            value = tl.load(Value + row * value_row + head * value_head + column, index < HEADS * DIM, other=0)
            page, offset = slot // PAGE, slot % PAGE
            tl.store(KeyCache + page * kp + offset * kt + head * kh + column, key, index < HEADS * DIM)
            tl.store(ValueCache + page * vp + offset * vt + head * vh + column, value, index < HEADS * DIM)
