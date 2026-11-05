# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded row transfers shared by local and remote checkpoint readers."""

from .checkpoint_source import BLOCK_BYTES


def chunks(offset, width, rows, source_stride, destination, destination_stride, expand):
    """Partition rows without transferring source gaps or changing destination offsets."""
    if any(type(value) is not int or value < 0 or value >= 1 << 63
           for value in (offset, width, rows, source_stride, destination, destination_stride)):
        raise ValueError("invalid routed checkpoint geometry")
    if rows < 1 or expand not in (0, 1) or (expand and width % 2):
        raise ValueError("invalid routed checkpoint expansion")
    if (offset + (rows - 1) * source_stride + width >= 1 << 63 or
            destination + (rows - 1) * destination_stride + width * (1 + expand) >= 1 << 63):
        raise ValueError("routed checkpoint extent overflows signed bounds")
    if rows > 1 and destination_stride < width * (1 + expand):
        raise ValueError("routed checkpoint destinations overlap")
    for column in range(0, width, BLOCK_BYTES):
        take = min(BLOCK_BYTES, width - column)
        count = 0
        while count < rows:
            capacity = min(BLOCK_BYTES // take,
                           (BLOCK_BYTES - take) // source_stride + 1 if source_stride else rows)
            batch = min(rows - count, capacity)
            yield (offset + count * source_stride + column, take, batch, source_stride,
                   destination + count * destination_stride + column * (1 + expand), destination_stride)
            count += batch


def destinations_overlap(left, right):
    """Intersect routed row bytes without treating strided holes as writes."""
    a, aw, ac, ast = left
    b, bw, bc, bst = right
    i = j = 0
    while i < ac and j < bc:
        start_a, start_b = a + i * ast, b + j * bst
        if start_a + aw <= start_b:
            if not ast:
                return False
            i = max(i + 1, (start_b - a - aw) // ast + 1)
        elif start_b + bw <= start_a:
            if not bst:
                return False
            j = max(j + 1, (start_a - b - bw) // bst + 1)
        else:
            return True
    return False
