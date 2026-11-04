# SPDX-License-Identifier: Apache-2.0

"""Exercise a native partial-GGUF package and keep raw HTTP observations.

Run against an otherwise idle `flare run` service. Ordinary and MTP sampling
need not emit identical token sequences; conditional scores use identical
teacher-forced prefixes. This fixture does not assess language quality.
"""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(url, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + route, data,
                {'Content-Type': 'application/json'}), timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'{route}: HTTP {error.code}: {error.read().decode()}') from error


def metrics(url):
    with urllib.request.urlopen(url + '/metrics', timeout=30) as response:
        lines = response.read().decode().splitlines()
    return {key: float(value) for line in lines if line and not line.startswith('#')
        for key, value in [line.rsplit(' ', 1)]}


def cancel_and_drain(url, model):
    before = metrics(url)
    invalid = dict(model=model, prompt=[[1], [4294967295]], max_tokens=8)
    try:
        urllib.request.urlopen(urllib.request.Request(url + '/v1/completions',
            json.dumps(invalid).encode(), {'Content-Type': 'application/json'}), timeout=30).close()
    except urllib.error.HTTPError as error:
        assert error.code == 400, error.read().decode()
        error.close()
    else:
        raise AssertionError('invalid token ID was admitted')
    assert metrics(url)['flare_requests_admitted_total'] == before['flare_requests_admitted_total']
    body = dict(model=model, prompt=[147, 83, 773, 18, 4, 294, 82], max_tokens=768,
        temperature=.7, top_p=.8, seed=731, ignore_eos=True, stream=True,
        logprobs=0, return_tokens_as_token_ids=True)
    first = None
    with urllib.request.urlopen(urllib.request.Request(url + '/v1/completions',
            json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=30) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            assert 'error' not in event, event
            if any((choice.get('logprobs') or {}).get('tokens') for choice in event.get('choices', [])):
                first = event
                break
    assert first is not None, 'stream did not return generated tokens before cancellation'
    deadline = time.monotonic() + 15
    while True:
        after = metrics(url)
        if after['flare_active_requests'] == after['flare_http_inflight_requests'] == 0:
            break
        assert time.monotonic() < deadline, 'cancelled request did not drain'
        time.sleep(.02)
    assert after['flare_requests_cancelled_total'] > before['flare_requests_cancelled_total']
    return dict(before=before, after=after, request=body, first_event=first)


def run(args):
    card = request(args.url, '/v1/models')['data'][0]
    assert card['id'] == 'qwen38-gguf-partial-qualification'
    assert card['flare']['input_modalities'] == ['text']
    before = request(args.url, '/metrics?format=json')
    observations = []

    def completion(name, prompt, budget, *, seed=17):
        body = dict(model=card['id'], prompt=prompt, max_tokens=budget, seed=seed,
            temperature=.7, top_p=.8, ignore_eos=True, logprobs=0,
            return_tokens_as_token_ids=True, logprob_token_ids=list(range(32)))
        start = time.perf_counter()
        result = request(args.url, '/v1/completions', body)
        choice = result['choices'][0]
        ids = [int(value.removeprefix('token_id:')) for value in choice['logprobs']['tokens']]
        assert choice['finish_reason'] == 'length', result
        assert len(ids) == result['usage']['completion_tokens'] == budget, result
        assert result['usage']['prompt_tokens'] == len(prompt), result
        observations.append(dict(name=name, request=body, response=result,
            seconds=time.perf_counter() - start))
        args.output.write_text(json.dumps(dict(card=card, before=before,
            observations=observations), indent=2) + '\n')
        print(name, dict(tokens=len(ids), cached=result['usage']['prompt_tokens_details']['cached_tokens'],
            seconds=observations[-1]['seconds']), flush=True)
        return ids, choice['logprobs'], result['usage']['prompt_tokens_details']['cached_tokens']

    for length in (7, 63, 64, 65, 127, 128, 129):
        prompt = [100 + (index * 37 + length) % 4096 for index in range(length)]
        completion(f'prefill-{length}', prompt, 8)
    prompt = [42, 101, 17, 356, 19, 8, 14]
    for budget in (1, 2, 3, 6, 7, 33, 2):
        completion(f'budget-{budget}', prompt, budget)
    prompt = [600 + (index * 13) % 4096 for index in range(129)]
    ids, scores, _ = completion('repeat-cold', prompt, 32, seed=941)
    repeated, restored_scores, cached = completion('repeat-cached', prompt, 32, seed=941)
    assert cached > 0, 'prefix restore was not exercised; provide sufficient KV backing'
    assert ids == repeated, 'cached continuation changed the seeded token sequence'
    assert scores == restored_scores, 'cached continuation changed target scores'
    lifecycle = cancel_and_drain(args.url, card['id'])
    repeated, restored_scores, cached = completion('repeat-after-cancel', prompt, 32, seed=941)
    assert cached > 0 and ids == repeated and scores == restored_scores
    completion('new-after-cancel', [751, 9, 402, 33, 29, 205, 37], 33, seed=533)
    if args.teacher:
        teacher = json.loads(args.teacher.read_text())
        selected = next(value for value in teacher['observations'] if value['name'] == 'repeat-cold')
        prefix = selected['request']['prompt']
        tokens = [int(value.removeprefix('token_id:'))
            for value in selected['response']['choices'][0]['logprobs']['tokens']]
        for index in range(len(tokens)):
            completion(f'teacher-{index}', prefix + tokens[:index], 1, seed=941)
    after = request(args.url, '/metrics?format=json')
    assert after['failed'] == before['failed'], (before, after)
    assert request(args.url, '/health')['status'] == 'ok'
    args.output.write_text(json.dumps(dict(card=card, before=before, after=after,
        observations=observations, lifecycle=lifecycle), indent=2) + '\n')
    print('PASS native requests', len(observations), 'output', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18081')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--teacher', type=Path)
    run(parser.parse_args())
