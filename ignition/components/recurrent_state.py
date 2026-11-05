# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kernel-independent recurrent state, packed metadata and prefill storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaState:
    """Chronological BF16 convolution history and FP32 value-by-key recurrence."""

    convolution: torch.Tensor
    recurrent: torch.Tensor

    @classmethod
    def allocate(cls, slots, device, *, heads, channels, speculative_tokens=0):
        if type(slots) is not int or slots < 2:
            raise ValueError("delta state requires a null slot and at least one live state slot")
        if any(type(value) is not int or value <= 0 for value in (heads, channels)):
            raise ValueError("delta state requires positive head and convolution channel counts")
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("delta state supports zero through seven speculative tokens")
        return cls(
            torch.zeros((slots, channels, 3 + speculative_tokens), dtype=torch.bfloat16, device=device),
            torch.zeros((slots, heads, 128, 128), dtype=torch.float32, device=device),
        )



@dataclass(frozen=True)
class DeltaMetadata:
    """Fixed-address device metadata updated by the native scheduler.

    Decode consumes ``state_indices[:, accepted_tokens - 1]`` and writes each
    query token's checkpoint to its corresponding column. Prefill uses explicit
    initial/final indices and can export a checkpoint at a multiple of 16 tokens.
    Live nonnegative slots must be unique and in range. Counts and packed query
    boundaries must fit their buffer capacities. These are scheduler invariants.
    """

    query_start_loc: torch.Tensor
    convolution_slots: torch.Tensor
    fresh: torch.Tensor
    accepted_tokens: torch.Tensor
    state_indices: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor
    checkpoint_state_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls, sequences: int, device: torch.device, *, verification_columns: int = 1
    ) -> DeltaMetadata:
        if type(sequences) is not int or sequences <= 0:
            raise ValueError("delta sequence capacity must be positive")
        if type(verification_columns) is not int or not 1 <= verification_columns < 1 << 31:
            raise ValueError("delta verification columns must be positive Int32 token lengths")

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((sequences + 1,)),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            zeros((sequences,), torch.bool),
            torch.ones(sequences, dtype=torch.int32, device=device),
            zeros((sequences, verification_columns), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,)),
            zeros((1,)),
            zeros((1,)),
        )



@dataclass(frozen=True)
class PrefillBuffers:
    """Contiguous query, key and value heads and FP32 decay and update gates for one prefill chunk."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    decay: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, key_heads: int, value_heads: int, buffers=None) -> PrefillBuffers:
        def zeros(shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        return cls(zeros((tokens, key_heads, 128)), zeros((tokens, key_heads, 128)),
                   zeros((tokens, value_heads, 128)), zeros((tokens, value_heads), torch.float32).fill_(1),
                   zeros((tokens, value_heads), torch.float32))

    def rows(self, tokens: int) -> PrefillBuffers:
        return PrefillBuffers(*(getattr(self, name)[:tokens] for name in ("query", "key", "value", "decay", "beta")))

    def fits(self, tokens: int, *, key_heads: int, value_heads: int) -> bool:
        return (self.query.shape == (tokens, key_heads, 128) and self.key.shape == self.query.shape
                and self.value.shape == (tokens, value_heads, 128)
                and self.decay.shape == self.beta.shape == (tokens, value_heads))
