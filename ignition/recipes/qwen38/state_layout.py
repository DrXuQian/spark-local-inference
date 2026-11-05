# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen continuation byte layouts independent of kernel libraries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.state_layout import StateLayout
from .cache_layout import cache_requirements

GDN_LAYERS = tuple(layer for layer in range(48) if layer % 4 != 3)
QSA_LAYERS = tuple(range(3, 48, 4))


@dataclass(frozen=True)
class HybridStateLayout(StateLayout):
    """Byte layout for one complete consumed-prefix continuation state."""

    speculative_tokens: int = 0
    draft_ring_capacity: int = 0

    @classmethod
    def build(cls, *, speculative_tokens: int = 0, world_size: int = 1,
              kv_dtype: torch.dtype = torch.float8_e4m3fn) -> HybridStateLayout:
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("hybrid state supports zero through seven speculative tokens")
        from ignition.components.parallel import TensorParallelShard

        TensorParallelShard(0, world_size)
        specifications = []

        def add(name, shape, dtype):
            specifications.append((name, shape, dtype))

        for layer in GDN_LAYERS:
            add(f"gdn.{layer}.recurrent", (48 // world_size, 128, 128), torch.float32)
            add(f"gdn.{layer}.convolution", (10240 // world_size, 3 + speculative_tokens), torch.bfloat16)
        add("ple_convolution", (10240, 9 + speculative_tokens), torch.bfloat16)
        add("last_hidden", (2560,), torch.bfloat16)
        add("consumed_tokens", (1,), torch.int64)
        add("ple_history", (2,), torch.int64)
        add("rope_delta", (1,), torch.int64)
        geometry = cache_requirements(main_page_size=64, position_axes=3, kv_heads=max(1, 2 // world_size),
                                         max_speculative_tokens=speculative_tokens, kv_dtype=kv_dtype)
        for layer in QSA_LAYERS:
            add(f"qsa.{layer}.raw_keys", geometry.raw_k_ring_shape, torch.bfloat16)
            add(f"qsa.{layer}.raw_positions", geometry.raw_logical_positions_shape, torch.int64)
            add(f"qsa.{layer}.raw_rope", geometry.raw_rope_positions_shape, torch.int64)
            add(f"qsa.{layer}.raw_anchor", (1,), torch.int64)
            add(f"qsa.{layer}.boundary_key", geometry.main_k_page_shape, geometry.kv_dtype)
            add(f"qsa.{layer}.boundary_value", geometry.main_v_page_shape, geometry.kv_dtype)
            add(f"qsa.{layer}.boundary_selector", geometry.compressed_page_shape, torch.bfloat16)
        ring = 0
        if speculative_tokens:
            ring = geometry.raw_ring_capacity
            add("target_residual", (10240,), torch.bfloat16)
            add("target_residual_position", (3,), torch.int64)
            add("draft_raw_keys", geometry.raw_k_ring_shape, torch.bfloat16)
            add("draft_raw_positions", geometry.raw_logical_positions_shape, torch.int64)
            add("draft_raw_rope", geometry.raw_rope_positions_shape, torch.int64)
            add("draft_raw_anchor", (1,), torch.int64)
            add("draft_boundary_key", geometry.main_k_page_shape, geometry.kv_dtype)
            add("draft_boundary_value", geometry.main_v_page_shape, geometry.kv_dtype)
            add("draft_boundary_selector", geometry.compressed_page_shape, torch.bfloat16)
            add("draft_boundary_key_extra", geometry.main_k_page_shape, geometry.kv_dtype)
            add("draft_boundary_value_extra", geometry.main_v_page_shape, geometry.kv_dtype)
            add("draft_boundary_selector_extra", geometry.compressed_page_shape, torch.bfloat16)
        layout = StateLayout.from_specs(specifications)
        return cls(layout.regions, layout.row_bytes, speculative_tokens, ring)
