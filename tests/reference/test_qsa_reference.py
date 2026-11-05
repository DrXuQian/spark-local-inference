# SPDX-License-Identifier: Apache-2.0

import torch

from qualify.qsa_reference import compress_history, paged_attention, select_positions


def test_completed_groups_pool_before_norm_and_keep_bf16_boundaries():
    # Two complete constant groups normalize to one; the incomplete suffix
    # cannot contribute to either group. Zero-centered weight -1 removes it.
    raw = torch.cat((torch.full((4, 128), 2.), torch.full((4, 128), 4.),
                     torch.full((3, 128), 99.))).bfloat16()
    positions = torch.arange(11)[:, None].expand(-1, 3)
    rotation_positions = []
    def identity(value, positions):
        rotation_positions.append(positions)
        return value.float()
    compressed = compress_history(raw, torch.zeros(128), positions, identity)
    torch.testing.assert_close(compressed, torch.ones(2, 128, dtype=torch.bfloat16), rtol=0, atol=0)
    assert rotation_positions[0][:, 0].tolist() == [0, 4]
    assert compress_history(raw, torch.full((128,), -1.), positions, identity).count_nonzero() == 0
    assert compress_history(raw[:3], torch.zeros(128), positions[:3], identity).shape == (0, 128)


def test_selection_ties_prefer_low_groups_and_append_only_incomplete_tail():
    positions = torch.tensor([2, 3, 2052])
    selected = select_positions(torch.zeros(3, 4, 128), torch.zeros(514, 128), positions, 2053)
    assert selected[0, :3].tolist() == [0, 1, 2] and (selected[0, 3:] == -1).all()
    assert selected[1, :4].tolist() == [0, 1, 2, 3] and (selected[1, 4:] == -1).all()
    assert selected[2, :2048].tolist() == list(range(2048))
    assert selected[2, 2048:].tolist() == [2052, -1, -1]


def test_paged_gqa_obeys_physical_mapping_head_groups_and_causal_tail():
    keys, values = torch.zeros(2, 64, 2, 4), torch.empty(2, 64, 2, 4)
    values[0, :, 0].fill_(1.); values[0, :, 1].fill_(3.)
    values[1, :, 0].fill_(5.); values[1, :, 1].fill_(7.)
    selected = torch.tensor([[0, 64, 65, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    actual = paged_attention(torch.zeros(2, 4, 4), keys, values, torch.tensor([[1, 0]]),
                            torch.zeros(2, dtype=torch.int64), selected, torch.tensor([64, 0]))
    expected = torch.zeros(2, 4, 4)
    expected[0, :2].fill_(3.); expected[0, 2:].fill_(5.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
