# SPDX-License-Identifier: Apache-2.0

"""Bit classification must preserve the original sampling rejection rules."""

import pytest
import torch

from ignition.components.probability_checks import finite_fp32, finite_nonnegative_fp32


def bit_patterns():
    # Cover both signs, every exponent and the mantissa boundaries separating
    # zero, subnormal, normal, infinity, signaling NaN and quiet NaN.
    mantissas = (0, 1, 0x3fffff, 0x400000, 0x7fffff)
    patterns = torch.tensor([sign | (exponent << 23) | mantissa
        for sign in (0, 0x80000000) for exponent in range(256) for mantissa in mantissas], dtype=torch.int64)
    random = torch.randint(0, 1 << 32, (65536,), generator=torch.Generator().manual_seed(31), dtype=torch.int64)
    return torch.cat((patterns, random)).to(torch.int32)


@pytest.mark.parametrize('strided', [False, True])
def test_all_fp32_classes_match_pytorch_without_mutating_input(strided):
    bits = bit_patterns()
    if strided:
        backing = torch.empty(bits.numel() * 2, dtype=torch.int32)
        backing[::2].copy_(bits)
        bits = backing[::2]
    before = bits.clone()
    values = bits.view(torch.float32)
    torch.testing.assert_close(finite_fp32(values), torch.isfinite(values), rtol=0, atol=0)
    torch.testing.assert_close(finite_nonnegative_fp32(values),
                               torch.isfinite(values) & (values >= 0), rtol=0, atol=0)
    assert torch.equal(bits, before)


@pytest.mark.parametrize('shape', [(1, 0, 128, 1940), (1, 1, 128, 1940), (4, 7, 2, 17)])
def test_empty_drafts_and_sliced_probability_validity_are_unchanged(shape):
    values = torch.zeros(shape)
    if values.numel():
        rows = values.view(-1, shape[-2] * shape[-1])
        rows[:, 0] = 1
        probes = (float('nan'), float('inf'), -float('inf'), -1., -0., 0., 1e-45)
        for index in range(min(len(rows), len(probes))):
            rows[index, -1] = probes[index]
    mass = values.sum(-1).sum(-1)
    expected = ((torch.isfinite(values) & (values >= 0)).all(-1).all(-1)
                & torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
    actual = (finite_nonnegative_fp32(values).all(-1).all(-1)
              & finite_fp32(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64, torch.int32])
def test_classification_rejects_non_fp32_bit_layouts(dtype):
    for classify in (finite_fp32, finite_nonnegative_fp32):
        with pytest.raises(ValueError, match='requires FP32'):
            classify(torch.zeros(4, dtype=dtype))


def test_compiled_classification_preserves_special_values():
    # Exercise the actual fullgraph compiler on the host; PPU catalog replay is
    # checked by test_external_sampling_cuda on the deployment device.
    compiled = torch.compile(finite_nonnegative_fp32, fullgraph=True, dynamic=False)
    values = bit_patterns().view(torch.float32)
    torch.testing.assert_close(compiled(values), torch.isfinite(values) & (values >= 0), rtol=0, atol=0)
