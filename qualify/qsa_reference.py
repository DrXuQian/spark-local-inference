# SPDX-License-Identifier: Apache-2.0

"""CPU QSA oracles for qualification without a b12x/cuTile installation.

These follow the original Flare QSA reference: BF16 pooling and zero-centered
normalization, FP32 rotary/index scores, stable ties, and FP32 attention.
Nothing in the prepared model or exported runtime dispatches here.
"""

import math

import torch


def encode_index(value, weight, positions, rope, eps=1e-6):
    value = value.float()
    normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
                  * (1. + weight.float())).bfloat16()
    return rope(normalized, positions)


def compress_history(raw, weight, positions, rope):
    groups = len(raw) // 4
    pooled = raw[:groups * 4].float().reshape(groups, 4, 128).mean(1).bfloat16()
    return encode_index(pooled, weight, positions[:groups * 4:4], rope).bfloat16()


def select_positions(query, compressed, positions, length):
    selected = torch.full((len(query), 2051), -1, dtype=torch.int32)
    for row, position in enumerate(positions.tolist()):
        count = min((position + 1) // 4, length // 4, len(compressed))
        scores = torch.einsum('hd,gd->hg', query[row].float(), compressed[:count].float())
        scores = scores.relu().sum(0) / math.sqrt(query.shape[-1])
        groups = torch.argsort(scores, descending=True, stable=True)[:512]
        expanded = (groups[:, None] * 4 + torch.arange(4)[None]).flatten()
        tail = torch.arange((position + 1) // 4 * 4, position + 1)
        joined = torch.cat((expanded, tail)).int()
        selected[row, :len(joined)] = joined
    return selected


def paged_attention(query, keys, values, block_table, requests, selected, positions):
    output = torch.zeros_like(query)
    page_size, kv_heads = keys.shape[1:3]
    heads = torch.arange(query.shape[1]) // (query.shape[1] // kv_heads)
    for row in range(len(query)):
        logical = selected[row].long()
        logical = logical[(logical >= 0) & (logical <= positions[row])]
        if not len(logical):
            continue
        physical = block_table[requests[row], logical // page_size].long()
        if bool(((physical < 0) | (physical >= len(keys))).any()):
            raise ValueError('selected QSA page is outside the physical cache')
        k, v = (cache[physical, logical % page_size][:, heads].float() for cache in (keys, values))
        scores = torch.einsum('hd,khd->hk', query[row].float(), k) / math.sqrt(query.shape[-1])
        output[row] = torch.einsum('hk,khd->hd', scores.softmax(-1), v).to(query.dtype)
    return output
