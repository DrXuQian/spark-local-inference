# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured continuation transfers between runtime-selected state rows and capacity bounds."""

import os
from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.sampling import AdmissionBuffers
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.attention import QSAState
from ignition.recipes.qwen38.decoder_state import DecoderInputs
from ignition.recipes.qwen38.maintenance import PreparedMaintenance
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.recipes.qwen38.state import HybridStatePool, StateTransferMetadata


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("capacity", [2, 32])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@torch.no_grad()
def test_noncontiguous_bindings_snapshot_normalization_and_runtime_bounds(capacity, kv_dtype):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    pool = HybridStatePool.allocate(max(7, capacity + 2), device, speculative_tokens=7, kv_dtype=kv_dtype)
    requests = DeviceRequestState.allocate(device, capacity=capacity, vocab_size=128)
    inputs = DecoderInputs.allocate(device, context=256, tokens=128, capacity=capacity)
    commands = StateTransferMetadata.allocate(capacity, device)
    admission = AdmissionBuffers.allocate(device, vocab_size=128, capacity=capacity)
    cache = {layer: QSAState.allocate(pages=8, slots=capacity + 1, device=device, draft_tokens=7, kv_dtype=kv_dtype)
             for layer in (*range(3, 48, 4), 48)}
    speculation = SimpleNamespace(cache=cache, draft_tables=torch.full_like(inputs.page_tables, -1),
        previous_accepted=torch.ones(capacity, dtype=torch.int32, device=device))
    records = torch.zeros((2, capacity, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    endpoint = 66 if kv_dtype == torch.bfloat16 else 65
    key_value = 11.0625 if kv_dtype == torch.bfloat16 else 11
    value_value = 12.0625 if kv_dtype == torch.bfloat16 else 12
    draft_values = (21.125, 22.125) if kv_dtype == torch.bfloat16 else (20, 22)
    operations, graphs = [], {}
    for mode in ("init", "snapshot", "restore"):
        operation = PreparedMaintenance(mode, requests, pool, commands, completion_outputs(records), frame, sequence,
            admission_buffers=admission, forward_inputs=inputs, capacity=capacity,
            qsa_states=tuple(cache[layer] for layer in range(3, 48, 4)), speculation=speculation)
        operation.prepare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        operations.append(operation)
        graphs[mode] = graph
    worker = PreparedGraphBundle(device=0, graphs=graphs, buffers={"records": records},
        resources=(pool, requests, inputs, commands, admission, cache, speculation, operations)).start_worker()

    def replay(index, *, valid=True):
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = worker.submit(index)
        completion = worker.poll(30000)
        assert completion.ticket == ticket and completion.error is None
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert (commands.errors[0].item() == 0) == valid
        assert commands.errors[1].item() == 0

    def command(slot, source, destination, *, reset=False, snapshot=False, consumed=0):
        commands.request_slots[0] = slot
        commands.source_slots[0] = source
        commands.destination_slots[0] = destination
        commands.operation[0] = int(reset)
        commands.qsa_slots[0] = -1 if snapshot else slot + 1
        commands.accepted_tokens[0] = 0 if snapshot else 1
        commands.expected_consumed[0] = consumed
        commands.num_ops.fill_(1)

    try:
        # Request slot one owns scattered rows, while row one is a snapshot destination.
        inputs.request_state_rows[1, :3] = torch.tensor([4, 2, 5], device=device)
        inputs.page_tables[1] = torch.tensor([0, 1, 2, 3], device=device)
        speculation.draft_tables[1] = torch.tensor([4, 5, 6, 7], device=device)
        admission.stage(0, 1, RequestAdmission(1, 42, eos_token_ids=()), list(range(66)))
        admission.admission_count.fill_(1)
        command(1, 4, 4, reset=True)
        replay(0)
        assert not requests.consumed_tokens.count_nonzero()
        assert requests.ple_history[1].tolist() == [248044, 248044]

        requests.consumed_tokens[1] = endpoint
        requests.ple_history[1] = torch.tensor([63, 64], device=device)
        pool.last_hidden[4].fill_(2)
        pool.rope_delta[4].fill_(7)
        for value in pool.gdn.values():
            value.recurrent[4].fill_(3)
            value.recurrent[5].fill_(-9)
            value.convolution[4].copy_(torch.arange(10, device=device).expand_as(value.convolution[4]))
        pool.ple_convolution[4].copy_(torch.arange(16, device=device).expand_as(pool.ple_convolution[4]))
        speculation.previous_accepted[1] = 3
        for value in cache.values():
            value.main_k_cache[1].fill_(key_value)
            value.main_v_cache[1].fill_(value_value)
            value.compressed_k_cache[1].fill_(13)
            value.raw_k_ring[2].fill_(14)
            value.raw_logical_positions[2].fill_(63)
            value.raw_rope_positions[2].fill_(63)
        for page, value in zip((4, 5), draft_values, strict=True):
            cache[48].main_k_cache[page].fill_(value)
            cache[48].main_v_cache[page].fill_(-value)
        inputs.row_slot_ids[0] = 1
        inputs.row_generations[0] = 1
        inputs.live_rows.fill_(1)
        command(1, 4, 1, snapshot=True, consumed=endpoint)
        replay(1)
        assert pool.consumed_tokens[1].item() == endpoint
        assert pool.ple_history[1].tolist() == [63, 64]
        assert pool.gdn[0].recurrent[1].eq(3).all()
        assert pool.gdn[0].convolution[1, 0, :3].tolist() == [2, 3, 4]
        assert pool.ple_convolution[1, 0, :9].tolist() == list(range(2, 11))
        saved = pool.backing[1].clone()

        pool.backing[2].fill_(0xA5)
        pool.backing[4].fill_(0xA5)
        pool.backing[5].fill_(0xA5)
        inputs.request_state_rows[1].zero_()
        inputs.request_state_rows[0, :3] = torch.tensor([6, 4, 2], device=device)
        inputs.page_tables[0] = torch.tensor([0, 2, 1, 3], device=device)
        speculation.draft_tables[0] = torch.tensor([4, 6, 5, 7], device=device)
        for value in cache.values():
            value.main_k_cache[2].fill_(-3)
            value.main_v_cache[2].fill_(-3)
        for page in (4, 6):
            cache[48].main_k_cache[page].fill_(-3)
            cache[48].main_v_cache[page].fill_(-3)
        admission.stage(0, 0, RequestAdmission(1, 43, eos_token_ids=()), list(range(66)))
        command(0, 1, 6, consumed=endpoint)
        before = pool.backing.clone()
        for bounds in ((6, 8), (7, 2)):
            pool.memory_capacity.copy_(torch.tensor(bounds, device=device))
            replay(2, valid=False)
            assert torch.equal(pool.backing, before)
        pool.memory_capacity.copy_(torch.tensor([7, 8], device=device))
        replay(2)
        assert requests.consumed_tokens[0].item() == endpoint
        assert requests.ple_history[0].tolist() == [63, 64]
        assert torch.equal(pool.backing[6], saved)
        assert torch.equal(pool.backing[1], saved)
        assert speculation.previous_accepted[0].item() == 1
        for layer in range(3, 48, 4):
            assert cache[layer].main_k_cache[2].eq(key_value).all()
            assert cache[layer].main_v_cache[2].eq(value_value).all()
        assert cache[48].main_k_cache[4].eq(draft_values[0]).all()
        assert cache[48].main_v_cache[4].eq(-draft_values[0]).all()
        if endpoint % 64 >= 2:
            assert cache[48].main_k_cache[6].eq(draft_values[1]).all()
            assert cache[48].main_v_cache[6].eq(-draft_values[1]).all()
        else:
            assert cache[48].main_k_cache[6].eq(-3).all()
            assert cache[48].main_v_cache[6].eq(-3).all()
        assert cache[3].raw_k_ring[1].eq(14).all()
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
