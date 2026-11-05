# SPDX-License-Identifier: Apache-2.0

"""Actual prepared KV copy plus FA3, including complete-forward compilation."""

import importlib
import os

import pytest
import torch

from ignition.components.flash_attn3 import (
    PreparedFlashAttention3,
    PreparedPagedKVWriter,
)
from ignition.preparation.flash_attn3 import FlashAttention3Spec
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                               reason='requires an assigned device and FLARE_RUN_CUDA_TESTS=1')


@pytest.fixture(scope='module')
def library():
    extension = importlib.import_module('flash_attn_3._C')
    return FlashAttention3Spec.read(extension.__file__).load('cuda:0')


class CopyAndAttention:
    def __init__(self, writer, attention):
        self.writer, self.attention = writer, attention

    def __call__(self):
        self.writer()
        return self.attention()


@pytest.mark.parametrize('width', [1, 7, 128])
@pytest.mark.parametrize('compiler', ['none', 'flare', 'catalog'])
@torch.no_grad()
def test_prepared_copy_and_attention_replay(library, width, compiler, tmp_path, monkeypatch):
    device, rows, capacity = torch.device('cuda:0'), 3, 3 * width
    generator = torch.Generator().manual_seed(851)
    def random(shape):
        return (torch.randn(shape, generator=generator) * .25).bfloat16()

    query_cpu = random((capacity, 24, 256))
    key_cpu, value_cpu = (random((10, 64, 2, 256)) for _ in range(2))
    fresh_key, fresh_value = (random((capacity, 2, 256)) for _ in range(2))
    pages_cpu = torch.tensor([[4, 0, 7, 2], [6, 1, 8, 3], [-1, -1, -1, -1]], dtype=torch.int32)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        query, key, value = (tensor.to(device) for tensor in (query_cpu, key_cpu, value_cpu))
        # Match K/V slices of Qwen's fused QKV projection, including row padding.
        qkv = torch.zeros(capacity, 13312, dtype=torch.bfloat16, device=device)
        new_key, new_value = qkv[:, 12288:12800].view(capacity, 2, 256), qkv[:, 12800:].view(capacity, 2, 256)
        new_key.copy_(fresh_key); new_value.copy_(fresh_value)
        pages, output = pages_cpu.to(device), torch.zeros_like(query)
        slots = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        starts = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        lengths = torch.zeros(rows, dtype=torch.int32, device=device)
        arena = ScratchArena(device, overlap=False)
        writer = PreparedPagedKVWriter(new_key, new_value, key, value, slots, workspace=arena)
        attention = PreparedFlashAttention3(library, query, key, value, pages, lengths, starts, output, width=width)
        arena.freeze(); arena.finalize()

        forward = CopyAndAttention(writer, attention)

        if compiler == 'flare':
            from ignition.preparation.compilation import compile_forward

            forward = compile_forward(forward)
        elif compiler == 'catalog':
            from ignition.preparation import aot
            from ignition.preparation.compile_pool import CompilePool
            from ignition.preparation.execution import GraphSignature

            cache = aot.ForwardCache(namespace={'qualification': 'fa3-paged'}, device=device,
                                     root=tmp_path, compile_workers=1)
            signature = {'attention': GraphSignature('prefill', rows, width)}
            compiled = cache.prepare_many({'attention': forward}, signature)['attention']
            assert not compiled.cache_hit
            assert compiled.artifact._artifacts.compiled_fn._tuning_frozen

            def no_compile(*args, **kwargs):
                raise AssertionError('the completed catalog must load without compiling')

            with monkeypatch.context() as loading:
                loading.setattr(aot, 'aot_forward', no_compile)
                loading.setattr(CompilePool, 'submit_payload', no_compile)
                loaded = cache.prepare_many({'attention': CopyAndAttention(writer, attention)}, signature)['attention']
            assert loaded.cache_hit
            loaded.prepare_capture()
            forward = loaded
        forward(); forward(); stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True) if compiler == 'catalog' else torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        try:
            for first, second, prefix in ((width, max(1, width // 2), 63), (0, width, 64),
                                          (1, 0, 65), (0, 0, 0)):
                ends, prefixes = [0, first, first + second, first + second], [prefix, prefix // 2, 0]
                full_lengths = [prefixes[r] + ends[r + 1] - ends[r] if ends[r + 1] > ends[r] else 0
                                for r in range(rows)]
                assigned = torch.full((capacity,), -1, dtype=torch.int64)
                expected_key, expected_value = key_cpu.clone(), value_cpu.clone()
                reference = torch.zeros_like(query_cpu)
                for row in range(rows):
                    begin, end = ends[row:row + 2]
                    if begin == end:
                        continue
                    for index in range(end - begin):
                        position = prefixes[row] + index
                        page = pages_cpu[row, position // 64]
                        assigned[begin + index] = page * 64 + position % 64
                        expected_key[page, position % 64] = fresh_key[begin + index]
                        expected_value[page, position % 64] = fresh_value[begin + index]
                    total = full_lengths[row]
                    selected = pages_cpu[row, :(total + 63) // 64].long()
                    k = expected_key[selected].flatten(0, 1)[:total].float().repeat_interleave(12, dim=1)
                    v = expected_value[selected].flatten(0, 1)[:total].float().repeat_interleave(12, dim=1)
                    scores = torch.einsum('qhd,khd->hqk', query_cpu[begin:end].float(), k) / 16.
                    mask = torch.arange(total)[None, :] > (prefixes[row] + torch.arange(end - begin))[:, None]
                    scores.masked_fill_(mask[None], -torch.inf)
                    reference[begin:end] = torch.einsum('hqk,khd->qhd', scores.softmax(-1), v).bfloat16()
                starts.copy_(torch.tensor(ends, dtype=torch.int32))
                lengths.copy_(torch.tensor(full_lengths, dtype=torch.int32))
                slots.copy_(assigned); key.copy_(key_cpu); value.copy_(value_cpu)
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); stream.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                actual, expected = output.cpu().float(), reference.float()
                assert (actual - expected).norm() / expected.norm().clamp_min(1e-12) < .02
                torch.testing.assert_close(actual, expected, rtol=.035, atol=float(expected.abs().max()) * .02)
                torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
                torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)
                assert not output[first + second:].any().item() and writer.error.item() == 0
        finally:
            stream.synchronize(); graph.reset()


@torch.no_grad()
def test_cache_only_copy_checks_full_int64_slots_and_keeps_repeated_call_error():
    device, tokens = torch.device('cuda:0'), 7
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        source = torch.arange(tokens * 512, dtype=torch.float32, device=device).reshape(tokens, 2, 256).bfloat16()
        key, value = (torch.zeros(2, 64, 2, 256, dtype=torch.bfloat16, device=device) for _ in range(2))
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        following = slots.clone()
        arena = ScratchArena(device, overlap=False)
        writer = PreparedPagedKVWriter(source, source, key, value, slots, workspace=arena)
        arena.freeze(); arena.finalize()

        def absorb_twice():
            writer()
            slots.copy_(following)
            writer()

        absorb_twice(); stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            absorb_twice()
        try:
            for invalid in (True, False):
                initial = [-1, -2, 128, (1 << 32) + 1, -(1 << 40), 0, 64] if invalid else [-1] * 5 + [0, 64]
                second = [-1] * 5 + [63, 65]
                slots.copy_(torch.tensor(initial)); following.copy_(torch.tensor(second))
                key.zero_(); value.zero_(); writer.error.zero_()
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); stream.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert writer.error.item() == int(invalid)
                expected = torch.zeros(128, 2, 256, dtype=torch.bfloat16)
                expected[0] = expected[63] = source[5].cpu()
                expected[64] = expected[65] = source[6].cpu()
                torch.testing.assert_close(key.cpu().flatten(0, 1), expected, rtol=0, atol=0)
                torch.testing.assert_close(value.cpu().flatten(0, 1), expected, rtol=0, atol=0)
        finally:
            stream.synchronize(); graph.reset()
