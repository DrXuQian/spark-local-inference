# SPDX-License-Identifier: Apache-2.0

"""Compare the partial GGUF service with native fixed-sequence replay.

Capture against an otherwise idle service with five drafts and adaptive
verification disabled. Stop it before running the native test tool. The corpus
contains CPU-prepared inputs and service observations, never executable code.
"""

import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

from qualify.qwen38_gguf_http import metrics, request


MODEL = 'qwen38-gguf-partial-qualification'
ROUND_COUNTER = 'flare_speculative_rounds_total{batch="1",width="6"}'


def serialize(values):
    return [dict(buffer=name, offset=0,
        data=base64.b64encode(value.contiguous().numpy().tobytes()).decode())
        for name, value in values.items()]


def inputs(prompt, seed, budget):
    # Python only prepares host bytes; native replay loads the exported package.
    import torch
    from ignition.components.sampling import AdmissionBuffers, RequestAdmission
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.state import StateTransferMetadata

    decoder = DecoderInputs.allocate('cpu', context=1024, tokens=64, capacity=1)
    decoder.request_state_rows[0, 0] = 1
    decoder.row_slot_ids[0] = 0
    decoder.row_generations.fill_(1)
    decoder.live_rows.fill_(1)
    decoder.query_lengths.fill_(len(prompt))
    decoder.token_ids[:len(prompt)] = torch.tensor(prompt)
    decoder.prefill_positions[:len(prompt)] = torch.arange(len(prompt))[:, None]
    decoder.page_tables[0] = torch.arange(16, dtype=torch.int32)
    decoder.sample_enabled.fill_(1)
    transfer = StateTransferMetadata.allocate(1, 'cpu')
    transfer.request_slots.fill_(0)
    transfer.source_slots.fill_(1)
    transfer.destination_slots.fill_(1)
    transfer.operation.fill_(1)
    transfer.qsa_slots.fill_(1)
    transfer.accepted_tokens.fill_(1)
    transfer.expected_consumed.fill_(0)
    transfer.num_ops.fill_(1)
    admission = AdmissionBuffers.allocate('cpu', vocab_size=248320, capacity=1)
    admission.stage(0, 0, RequestAdmission(generation=1, seed=seed,
        max_new_tokens=budget, temperature=.7, top_p=.8, top_k=20), prompt)
    admission.admission_count.fill_(1)
    init = dict(graph='init1', uploads=serialize(dict(
        request_state_rows=decoder.request_state_rows,
        **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
        **{'state_' + name: getattr(transfer, name)
           for name in transfer.__dataclass_fields__ if name != 'errors'})))
    gates = dict(logprob_requested=torch.ones(1, dtype=torch.int32),
        logprob_token_id_counts=torch.zeros(1, dtype=torch.int32))
    values = dict(decoder.buffers(), sampling_grammar_mask=admission.admission_grammar_mask,
        copy_token_ids=torch.full((1, 7), -1, dtype=torch.int64),
        copy_token_count=torch.zeros(1, dtype=torch.int32),
        copy_expected_consumed=torch.full((1,), -1, dtype=torch.int64),
        copy_generation=torch.zeros(1, dtype=torch.int64),
        draft_page_tables=torch.arange(32, 48, dtype=torch.int32).reshape(1, 16), **gates)
    prefill = dict(graph='prefill64', uploads=serialize(values))
    decoder.query_lengths.fill_(6)
    # Stable row/page inputs retain their prefill values. The native executor
    # still owns frame sequences, PLE delivery and completion downloads.
    round_step = dict(graph='speculative1x6', uploads=serialize(dict(
        query_lengths=decoder.query_lengths, **gates)))
    return init, prefill, round_step


def token_observations(events):
    result = []
    for event in events:
        for choice in event['value'].get('choices', []):
            for token in (choice.get('logprobs') or {}).get('tokens', []):
                result.append((int(token.removeprefix('token_id:')), event['at_ns']))
    return result


def capture(args):
    card = request(args.url, '/v1/models')['data'][0]
    if card['id'] != MODEL:
        raise ValueError('this corpus requires the partial GGUF fixture')
    corpus = dict(model_id=MODEL, draft_tokens=5, cases=[])
    for index in range(4):
        prompt, seed, budget = [177, 83, 809 + index, 18, 4, 294, 82], 430 + index, 128
        init, prefill, round_step = inputs(prompt, seed, budget)
        body = dict(model=MODEL, prompt=prompt, max_tokens=budget, seed=seed,
            temperature=.7, top_p=.8, top_k=20, ignore_eos=True, stream=True,
            logprobs=0, return_tokens_as_token_ids=True, stream_options=dict(include_usage=True))
        before, events = metrics(args.url), []
        with urllib.request.urlopen(urllib.request.Request(args.url + '/v1/completions',
                json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                value = json.loads(line[6:])
                assert 'error' not in value, value
                events.append(dict(at_ns=time.monotonic_ns(), value=value))
        after = metrics(args.url)
        ids = [token for token, _ in token_observations(events)]
        usage = next(event['value']['usage'] for event in reversed(events) if event['value'].get('usage'))
        assert len(ids) == usage['completion_tokens'] == budget
        assert usage['prompt_tokens'] == len(prompt)
        assert usage['prompt_tokens_details']['cached_tokens'] == 0, 'restart service for a cold corpus'
        assert after['flare_requests_failed_total'] == before['flare_requests_failed_total']
        rounds = after[ROUND_COUNTER] - before[ROUND_COUNTER]
        assert rounds > 1
        case = dict(name=f'case-{index}', init=init, prefill=prefill, round=round_step,
            width=6, budget=budget, expected_tokens=ids, request=body,
            service_events=events, service_rounds=int(rounds), metrics_before=before, metrics_after=after)
        corpus['cases'].append(case)
        args.output.write_text(json.dumps(corpus) + '\n')
        print(case['name'], 'tokens', len(ids), 'rounds', int(rounds), flush=True)


def compare(args):
    corpus = json.loads(args.corpus.read_text())
    replay = json.loads(args.replay.read_text())
    assert corpus['model_id'] == MODEL and corpus['draft_tokens'] == 5
    assert len(corpus['cases']) == len(replay) == 4
    rows = []
    for case, result in zip(corpus['cases'], replay, strict=True):
        assert case['name'] == result['case']
        assert result['state_equal'] and result['logprobs_equal']
        assert result['state_bytes_compared'] > 0
        observations = token_observations(case['service_events'])
        assert [token for token, _ in observations] == result['tokens'] == case['expected_tokens']
        samples = result['samples']
        assert case['service_rounds'] == result['rounds'] == len(samples)
        accepted = sum(sample['record'][10] for sample in samples)
        accepted_key = 'flare_speculative_accepted_drafts_total{batch="1",width="6"}'
        assert accepted == case['metrics_after'][accepted_key] - case['metrics_before'][accepted_key]
        # Compare the same interior round sequence, excluding admission, prefill
        # and the first decode round. SSE includes serialization/loopback delivery.
        first_token = samples[0]['record'][3] - 1
        service_ns = observations[-1][1] - observations[first_token][1]
        floor_ns = samples[-1]['at_ns'] - samples[0]['at_ns']
        assert service_ns > 0 and floor_ns > 0
        rows.append(dict(case=case['name'], interior_rounds=len(samples) - 1,
            service_ns=service_ns, replay_ns=floor_ns, ratio=service_ns / floor_ns,
            max_replay_readiness_ns=max(sample['readiness_ns'] for sample in samples),
            passed=service_ns <= 1.05 * floor_ns))
    report = dict(scope='batch-one partial GGUF fixture, fixed width six, RAM PLE; not full-model qualification',
        timing='first completed decode round through last; service includes SSE/loopback delivery', cases=rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(row['passed'] for row in rows):
        raise SystemExit('native service exceeded the 5% replay-overhead limit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('capture')
    record.add_argument('--url', default='http://127.0.0.1:18081')
    record.add_argument('--output', required=True, type=Path)
    record.set_defaults(run=capture)
    analysis = commands.add_parser('compare')
    analysis.add_argument('--corpus', required=True, type=Path)
    analysis.add_argument('--replay', required=True, type=Path)
    analysis.add_argument('--output', required=True, type=Path)
    analysis.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)
