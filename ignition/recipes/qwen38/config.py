# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen preparation capacities validated without loading execution kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.capacity import batch_capacities, request_capacity
from ignition.preparation.tuning_replicas import TuningReplica
from .state_layout import HybridStateLayout


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    device: int = 0
    devices: tuple[int, ...] | None = None
    # Each rank's host when ranks run on separate hosts; devices are then host-local ordinals.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    compile_decoder: bool = True
    retain_graphs: bool = False
    max_active: int = 4
    context: int = 262144
    prefill_chunk_size: int = 4096
    mixed_prefill: bool = False
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    pages: int | None = None
    kernel_cache: Path | None = None
    mtp: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    # Run shared experts, small-batch attention projections and selection, and GDN input projections
    # on an auxiliary stream; without overlap every branch runs serially on the main stream.
    stream_overlap: bool = True
    shard_hyperconnections: bool = False
    gguf_inputs: object = None

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        if self.gguf_inputs is None:
            object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        else:
            from ignition.preparation.qwen38_gguf import GgufInputs

            if not isinstance(self.gguf_inputs, GgufInputs) or self.image_concurrency not in (None, 0):
                raise ValueError('GGUF text preparation requires explicit inputs and no image slots')
            if self.shard_hyperconnections or len(self.devices or (self.device,)) != 1:
                raise ValueError('GGUF model preparation currently requires one CUDA device')
            object.__setattr__(self, 'image_concurrency', 0)
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if self.devices is not None and (
                not isinstance(self.devices, tuple) or len(self.devices) not in (1, 2, 4)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or len(set(zip(self.hosts or ("",) * len(self.devices), self.devices))) != len(self.devices)
                or any(type(device) is not int or device < 0 for device in self.devices)):
            raise ValueError("tensor parallelism requires one, two or four distinct CUDA devices")
        if self.hosts is not None and self.devices is None:
            raise ValueError("ranks on separate hosts require their devices")
        if self.shard_hyperconnections and len(self.devices or (self.device,)) < 2:
            raise ValueError("--shard-hyperconnections requires tensor parallelism")
        request_capacity(self.max_active)
        for name in ("device", "context", "prefill_chunk_size", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.device < 0 or not 64 <= self.context <= 262144 or self.context % 64:
            raise ValueError("device must be nonnegative and context must be 64-aligned in [64,262144]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        if self.pages is not None and (type(self.pages) is not int or self.pages <= 0):
            raise ValueError("explicit page capacity must be a positive integer")
        if type(self.mtp) is not bool or type(self.stream_overlap) is not bool:
            raise TypeError("MTP and stream overlap switches must be booleans")
        if type(self.mixed_prefill) is not bool:
            raise TypeError("mixed_prefill must be a boolean")
        if self.actlize is not None and self.mixed_shapes:
            raise ValueError('actlizeLA GDN does not implement mixed prefill/decode partitions')
        minimum_snapshot = HybridStateLayout.build(speculative_tokens=7 if self.mtp else 0,
            world_size=1 if self.devices is None else len(self.devices), kv_dtype=self.kv_dtype).row_bytes
        if self.snapshot_budget_bytes < minimum_snapshot or self.headroom_bytes < 0:
            raise ValueError("snapshot budget requires at least one state row; headroom must be nonnegative")
        widths = self.speculative_widths
        if (not isinstance(widths, tuple) or not widths or widths[0] != 1
                or any(type(width) is not int or not 1 <= width <= 8 for width in widths)
                or tuple(sorted(set(widths))) != widths):
            raise ValueError("speculative widths must be an increasing subset of 1 through 8 including one")
        if self.speculative_fixed_width is not None and (
                type(self.speculative_fixed_width) is not int or self.speculative_fixed_width not in widths):
            raise ValueError("fixed verification width must belong to the prepared catalog")

    @property
    def actlize(self):
        return None if self.gguf_inputs is None else self.gguf_inputs.actlize

    @property
    def kv_dtype(self):
        return torch.bfloat16 if self.actlize is not None else torch.float8_e4m3fn

    @property
    def preparation_pages(self):
        # One prefill chunk plus partial and verification-boundary pages for each row.
        return self.pages if self.pages is not None else self.prefill_chunk_size // 64 + 2 * self.request_capacity

    @property
    def preparation_state_rows(self):
        # Null row, live request rows, and disjoint snapshot destinations for tuning and capture.
        return 1 + 2 * self.request_capacity

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def batch_capacities(self):
        return batch_capacities(self.max_active)

    @property
    def token_capacity(self):
        return max(self.prefill_chunk_size,
                   max((b * w for b, w in self.speculative_shapes), default=0) if self.mtp else self.request_capacity,
                   max((p + (b - 1) * w for _, b, w, p in self.mixed_shapes), default=0))

    @property
    def mixed_shapes(self):
        from ignition.components.capacity import mixed_shapes

        if not self.mixed_prefill:
            return ()
        modes = (True, False) if self.mtp else (False,)
        return tuple((enabled, b, w, p) for enabled in modes
            for b, w, p in mixed_shapes(self.max_active, self.prefill_chunk_size,
                                        self.speculative_widths if enabled else (1,),
                                        self.speculative_token_capacity if enabled else None))

    @property
    def speculative_shapes(self) -> tuple[tuple[int, int], ...]:
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)

    @property
    def prefill_capacities(self) -> tuple[int, ...]:
        from ignition.components.capacity import prefill_capacities

        return prefill_capacities(self.prefill_chunk_size)

    @property
    def completion_token_capacity(self):
        return max((width for _, width in self.speculative_shapes), default=1) if self.mtp else 1
