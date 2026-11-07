# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GDN verification storage shared by the prepared recurrence implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Verified tokens per round: the eight checkpoint columns of verification metadata.
WINDOW = 8


@dataclass(frozen=True)
class GDNRecords:
    """Per-request correction, normalized key and decay records of one GDN layer's verified tokens."""

    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True)
class GDNLayerRecords:
    """Records of every GDN layer in stacked tensors, with one view per layer for verification."""

    layers: tuple[int, ...]
    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor

    @classmethod
    def allocate(cls, layers, rows: int, device, *, key_heads: int, value_heads: int, width: int = WINDOW) -> GDNLayerRecords:
        if type(width) is not int or not 1 <= width <= 16:
            raise ValueError("GDN record width must be in [1,16]")
        layers = tuple(layers)
        count = len(layers)
        return cls(layers,
                   torch.zeros((count, rows, value_heads, width, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, key_heads, width, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, value_heads, width), dtype=torch.float32, device=device))

    def __getitem__(self, layer: int) -> GDNRecords:
        index = self.layers.index(layer)
        return GDNRecords(self.delta[index], self.keys[index], self.decay[index])
