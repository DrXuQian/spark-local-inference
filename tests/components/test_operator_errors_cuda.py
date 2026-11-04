# SPDX-License-Identifier: Apache-2.0

"""Actual external lookup faults must invalidate even post-sampling completions."""

import os
import struct
import time

import gguf
import numpy as np
import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.gguf_embedding import PreparedGgufEmbedding
from ignition.components.gguf_weights import GgufWeight
from ignition.components.operator_errors import GuardedForward, OPERATOR_FAILURE, accumulate_errors
from ignition.components.sampling import DeviceRequestState, PreparedSampling, RequestAdmission
from ignition.preparation.aot import ForwardCache
from ignition.preparation.execution import GraphSignature, PreparedExecution


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='explicit CUDA assignment required')


class LateFailure:
    def __init__(self, sampler, lookup, ids):
        self.sampler, self.lookup, self.ids = sampler, lookup, ids
        self.sticky = torch.zeros(1, dtype=torch.int32, device=ids.device)
        self.device_errors = (self.sticky,)

    def __call__(self):
        # The decoder normally consumes the pending token before sampling.
        state = self.sampler.state
        state.pending_token.copy_(torch.where(state.status == 1, -1, state.pending_token))
        self.sampler()
        self.lookup()
        accumulate_errors(self.lookup.device_errors, self.sticky)
        # A later valid call clears the native per-call flag. The first failure
        # must survive, just as it must across repeated learned draft calls.
        self.ids.zero_()
        self.lookup()
        accumulate_errors(self.lookup.device_errors, self.sticky)


def test_late_lookup_failure_survives_repeated_calls_and_slot_reuse(tmp_path):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    state = DeviceRequestState.allocate(device, capacity=4, vocab_size=8)
    completion = CompletionOutputs.allocate((2, 4, 24), device=device)
    slots = torch.tensor([0, 2], dtype=torch.int32, device=device)
    epochs = torch.ones(2, dtype=torch.int64, device=device)
    live = torch.zeros(1, dtype=torch.int32, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    logits = torch.zeros(2, 8, dtype=torch.float32, device=device)
    sampler = PreparedSampling(logits, state, slots, epochs, live, completion, frame, sequence)
    sampler.prepare()
    raw = gguf.quantize(np.arange(8 * 32, dtype=np.float32).reshape(8, 32) * .001,
                        gguf.GGMLQuantizationType.Q8_0)
    weight = GgufWeight(torch.from_numpy(raw).to(device), 'Q8_0', 8, (8, 32))
    ids = torch.zeros(1, dtype=torch.int64, device=device)
    lookup = PreparedGgufEmbedding(weight, ids, torch.zeros(1, 32, dtype=torch.bfloat16, device=device),
        session=None, name='late-foreign-op')
    body = LateFailure(sampler, lookup, ids)
    guarded = GuardedForward(body, state, completion, frame, sequence)
    cache = ForwardCache(namespace={'model': 'late-external-failure'}, device=device, root=tmp_path,
                         compile_workers=1)
    compiled = cache.prepare_many({'decode': guarded}, {'decode': GraphSignature('decode', 2, 1)})['decode']
    compiled(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    executor = None
    try:
        scenarios = [(8, 2, False), (0, 2, False), (-2, 1, False),
                     (2**63 - 1, 2, True), (8, 0, False), (1, 2, False)]
        for index, (token, count, stale) in enumerate(scenarios):
            generation = 10 + index
            for slot in range(4):
                state.admit(slot, RequestAdmission(generation, 40 + slot, max_new_tokens=1, top_k=8), ())
            epochs.fill_(generation)
            if stale:
                epochs[1] -= 1
            ids.fill_(token); live.fill_(count); frame.fill_(index % 2); sequence.fill_(101 + index)
            completion.records[1 - index % 2].fill_(901 + index)
            completion.speculative_survival.fill_(.5)
            body.sticky.fill_(73)  # Must be cleared by this graph's own boundary.
            untouched = completion.records[1 - index % 2].clone()
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay(); torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            assert lookup.device_error.item() == 0, 'the last external call is valid'
            assert torch.equal(completion.records[1 - index % 2], untouched)
            bad = token < 0 or token >= 8
            affected = {0, 2} if count == 2 and not stale else {0} if count else set()
            for slot in range(4):
                if slot not in affected:
                    assert state.status[slot].item() == 1
                    assert state.error_code[slot].item() == 0
                    continue
                record = completion.records[index % 2, slot]
                assert record[0].item() == generation and record[1].item() == 101 + index
                if bad:
                    assert record[5:7].tolist() == [3, OPERATOR_FAILURE]
                    assert state.status[slot].item() == 3 and state.error_code[slot].item() == OPERATOR_FAILURE
                    assert record[4].item() == -1 and record[8:12].eq(0).all().item()
                    assert record[16:].eq(-1).all().item()
                    assert record[2:4].eq(0).all().item()
                    assert state.consumed_tokens[slot].item() == state.generated_tokens[slot].item() == 0
                    assert completion.speculative_survival[index % 2, slot].eq(-1).all().item()
                else:
                    assert record[5:7].tolist() == [2, 0]
                    assert record[8].item() == 1 and 0 <= record[4].item() < 8

        for terminal in (2, 3):
            for slot in range(4):
                state.admit(slot, RequestAdmission(30, 40 + slot, max_new_tokens=8, top_k=8), ())
            state.status[2] = terminal
            state.error_code[2] = 5 if terminal == 3 else 0
            state.consumed_tokens[2], state.generated_tokens[2], state.pending_token[2] = 6, 5, 3
            saved = {name: value[2].clone() for name, value in vars(state).items()}
            epochs.fill_(30); live.fill_(2); ids.fill_(8); frame.fill_(0); sequence.fill_(150)
            graph.replay(); torch.cuda.synchronize()
            for name, value in saved.items():
                assert torch.equal(getattr(state, name)[2], value), name
            assert completion.records[0, 0, 5:7].tolist() == [3, OPERATOR_FAILURE]
            assert completion.records[0, 2, 2:4].tolist() == [6, 5]
            assert completion.records[0, 2, 5:7].tolist() == [terminal, 5 if terminal == 3 else 0]
            assert completion.records[0, 2, 8].item() == 0

        # Use the actual native two-frame transport and completion validator.
        # A second queued graph must preserve a terminal error, then the same
        # slots must work again after a new generation has been admitted.
        bundle = PreparedExecution(device=0, max_active=4, graphs={'decode': graph},
            signatures={'decode': GraphSignature('decode', 2, 1, ('lookup_ids',))},
            buffers={**completion.buffers(), 'frame_index': frame,
                     'submission_sequence': sequence, 'lookup_ids': ids},
            resources=(guarded, compiled, state, lookup, body, epochs, live, slots))
        executor = bundle.start_executor()
        try:
            for generation, first_token in ((50, 8), (51, 0), (52, -2), (53, 1)):
                for slot in range(4):
                    state.admit(slot, RequestAdmission(generation, 60 + slot, max_new_tokens=8, top_k=8),
                                [1] * 8, consumed_tokens=7)
                state.generated_tokens.fill_(3)
                epochs.fill_(generation); live.fill_(2)
                completion.speculative_survival.fill_(-1.)
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                fences = [executor.submit(0, [('lookup_ids', 0, struct.pack('q', token)),
                          ('logprob_requested', 0, struct.pack('4i', 1, 0, 1, 0)),
                          ('logprob_token_id_counts', 0, bytes(16))])
                          for token in (first_token, 0)]
                assert all(fence is not None for fence in fences)
                assert executor.outstanding == 2
                bad = not 0 <= first_token < 8
                for round_index, fence in enumerate(fences):
                    deadline, result = time.monotonic() + 30, None
                    while result is None:
                        assert time.monotonic() < deadline, 'native completion timed out'
                        result = executor.poll()
                        if result is None:
                            time.sleep(.001)
                    assert result.fence == fence
                    for slot in (0, 2):
                        record = result.records[slot]
                        assert record[0] == generation and record[2] == 7
                        if bad:
                            assert record[3:7] == [3, -1, 3, OPERATOR_FAILURE]
                            assert record[8:12] == [0] * 4 and record[16:] == [-1] * 8
                            assert all(entry is None for entry in result.logprobs[slot])
                        else:
                            assert record[3] == 4 + round_index
                            assert record[5:7] == [1, 0] and record[8] == 1
                    for slot in (1, 3):
                        assert result.records[slot][5] == 0
                assert executor.outstanding == 0
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
        finally:
            executor.close()
            assert executor.drained
    finally:
        if executor is None or executor.drained:
            torch.cuda.synchronize(); graph.reset()
