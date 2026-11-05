# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen rotary tables, paged caches and metadata without operator imports."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_layout import cache_requirements


@dataclass(frozen=True)
class RotaryTables:
    cos: torch.Tensor
    sin: torch.Tensor
    axes: torch.Tensor

    @classmethod
    def allocate(cls, config, capacity: int, device: torch.device) -> RotaryTables:
        if capacity <= 0:
            raise ValueError("rotary position capacity must be positive")
        rope = config.rope_parameters
        if (config.head_dim != 256 or rope["partial_rotary_factor"] != .25
                or tuple(rope["mrope_section"]) != (11, 11, 10)
                or rope["rope_theta"] != 10000000):
            raise ValueError("Qwen rotary geometry differs from the checkpoint contract")
        inverse = (1.0 / (rope["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(device)
        frequency = torch.arange(capacity, device=device, dtype=torch.float32)[:, None] * inverse
        axes = torch.zeros(32, device=device, dtype=torch.int64)
        axes[1:33:3] = 1
        axes[2:30:3] = 2
        return cls(frequency.cos(), frequency.sin(), axes)

    def gather(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Negative metadata marks inactive rows; their arithmetic uses position zero.
        rows = positions.clamp_min(0).index_select(1, self.axes)
        columns = torch.arange(32, device=positions.device)
        cos, sin = self.cos[rows, columns], self.sin[rows, columns]
        return torch.cat((cos, cos), -1).bfloat16(), torch.cat((sin, sin), -1).bfloat16()


@dataclass(frozen=True)
class QSAState:
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    compressed_k_cache: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_rope_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor

    @classmethod
    def allocate(cls, *, pages: int, slots: int, device: torch.device,
                 page_size: int = 64, draft_tokens: int = 0, kv_heads: int = 2,
                 kv_dtype: torch.dtype = torch.float8_e4m3fn) -> QSAState:
        if pages <= 0 or slots <= 0:
            raise ValueError("QSA page and state capacities must be positive")
        geometry = cache_requirements(main_page_size=page_size, position_axes=3, kv_heads=kv_heads,
                                         max_speculative_tokens=draft_tokens, kv_dtype=kv_dtype)
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, device=device, dtype=dtype)
        def invalid(shape):
            return torch.full(shape, -1, device=device, dtype=torch.int64)
        return cls(zeros((pages, *geometry.main_k_page_shape), geometry.kv_dtype),
                   zeros((pages, *geometry.main_v_page_shape), geometry.kv_dtype),
                   zeros((pages, *geometry.compressed_page_shape)),
                   zeros((slots, *geometry.raw_k_ring_shape)),
                   invalid((slots, *geometry.raw_logical_positions_shape)),
                   invalid((slots, *geometry.raw_rope_positions_shape)), invalid((slots,)))

    def resize_pages_for_preparation(self, pages: int, *, metadata_mode=None, storage=None, prefix="", backed_pages=None) -> QSAState:
        """Allocate final page pools while retaining fixed raw selector slots.

        No graph referencing the returned page pools may have been handed to
        native execution. Rebind each prepared attention operator and recapture
        its graphs before use. Page contents are empty after this operation.
        """
        if type(pages) is not int or pages <= 0:
            raise ValueError("QSA page capacity must be positive")
        def pages_like(value, name):
            if metadata_mode is not None:
                from ignition.preparation.deferred import deferred_tensor

                return deferred_tensor((pages, *value.shape[1:]), dtype=value.dtype,
                                       device=value.device, mode=metadata_mode)
            if storage is not None:
                return storage.device_allocate(prefix + name, (pages, *value.shape[1:]), value.dtype,
                    group="KV_PAGES", maximum=pages, units=backed_pages)
            return torch.zeros((pages, *value.shape[1:]), dtype=value.dtype, device=value.device)
        return type(self)(pages_like(self.main_k_cache, "key"), pages_like(self.main_v_cache, "value"),
            pages_like(self.compressed_k_cache, "selector"), self.raw_k_ring, self.raw_logical_positions,
            self.raw_rope_positions, self.raw_interval_start_positions)


@dataclass(frozen=True)
class QSAMetadata:
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    request_ids: torch.Tensor
    query_positions: torch.Tensor
    rope_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    is_prefilling: torch.Tensor
    slot_mapping: torch.Tensor

    @classmethod
    def allocate(cls, *, tokens: int, batch: int, context: int,
                 device: torch.device, page_size: int = 64) -> QSAMetadata:
        if tokens < batch or min(batch, context) <= 0 or context % page_size:
            raise ValueError("QSA requires positive, page-aligned capacity and tokens >= batch")
        def full(shape, value=-1, dtype=torch.int64):
            return torch.full(shape, value, dtype=dtype, device=device)
        return cls(full((batch, context // page_size), dtype=torch.int32),
                   full((batch, context // page_size), dtype=torch.int32), full((batch,)),
                   full((tokens,)), full((tokens,)), full((tokens, 3)),
                   full((batch,), 0, torch.int32), full((batch + 1,), 0, torch.int32),
                   full((batch,), 0, torch.int32), full((batch,), False, torch.bool), full((tokens,)))
