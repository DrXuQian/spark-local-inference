# SPDX-License-Identifier: Apache-2.0

"""Probe the installed PPU FA3 ABI required by ordinary MTP attention.

This is a dependency gate, not a claim that the Qwen recipe uses FA3 yet.
"""

import importlib
import os

import pytest
import torch


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                               reason='requires an actual device and explicit FLARE_RUN_CUDA_TESTS=1')


@pytest.fixture(scope='module')
def native():
    # The T-Head wheel registers this operator; an import alone does not prove
    # that its build enabled paged KV, append, variable lengths or D256.
    importlib.import_module('flash_attn_3._C')
    operation = torch.ops.flash_attn_3.fwd.default
    fields = {argument.name for argument in operation._schema.arguments}
    required = {'q', 'k', 'v', 'k_new', 'v_new', 'out', 'cu_seqlens_q', 'cu_seqlens_k_new',
                'seqused_k', 'max_seqlen_q', 'page_table', 'is_causal', 'num_splits'}
    assert required <= fields, f'installed FA3 forward ABI is missing {sorted(required - fields)}'
    return operation


@pytest.mark.parametrize('width', [1, 7, 128])
@pytest.mark.parametrize('append', [False, True])
@torch.no_grad()
def test_page64_d256_packed_suffix_and_append_replay(native, width, append):
    device, rows, page_size = torch.device('cuda:0'), 3, 64
    capacity = rows * width
    generator = torch.Generator().manual_seed(725)
    query_cpu = (torch.randn(capacity, 24, 256, generator=generator) * .25).bfloat16()
    key_cpu = (torch.randn(10, page_size, 2, 256, generator=generator) * .25).bfloat16()
    value_cpu = (torch.randn(10, page_size, 2, 256, generator=generator) * .25).bfloat16()
    fresh_key_cpu = (torch.randn(capacity, 2, 256, generator=generator) * .25).bfloat16()
    fresh_value_cpu = (torch.randn(capacity, 2, 256, generator=generator) * .25).bfloat16()
    pages_cpu = torch.tensor([[4, 0, 7, 2], [6, 1, 8, 3], [-1, -1, -1, -1]], dtype=torch.int32)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        query, key, value = (tensor.to(device) for tensor in (query_cpu, key_cpu, value_cpu))
        new_key, new_value = fresh_key_cpu.to(device), fresh_value_cpu.to(device)
        output = torch.zeros_like(query)
        pages = pages_cpu.to(device)
        starts = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        lengths = torch.zeros(rows, dtype=torch.int32, device=device)

        def forward():
            output.zero_()
            result = native(query, key, value, k_new=new_key if append else None,
                v_new=new_value if append else None, out=output, cu_seqlens_q=starts,
                cu_seqlens_k_new=starts if append else None, seqused_k=lengths,
                max_seqlen_q=width, page_table=pages, is_causal=True, num_splits=1)
            assert result[0].data_ptr() == output.data_ptr()

        forward(); stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        try:
            for first, second, prefix in ((width, max(1, width // 2), 63), (0, width, 64),
                                          (1, 0, 65), (0, 0, 0)):
                ends = [0, first, first + second, first + second]
                prefixes = [prefix, prefix // 2, 0]
                full_lengths = [prefixes[r] + ends[r + 1] - ends[r] if ends[r + 1] > ends[r] else 0
                                for r in range(rows)]
                supplied = [prefixes[r] if full_lengths[r] else 0 for r in range(rows)] if append else full_lengths
                starts.copy_(torch.tensor(ends, dtype=torch.int32))
                lengths.copy_(torch.tensor(supplied, dtype=torch.int32))
                key.copy_(key_cpu); value.copy_(value_cpu)
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); stream.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                expected_key, expected_value = key_cpu.clone(), value_cpu.clone()
                reference = torch.zeros_like(query_cpu)
                for row in range(rows):
                    begin, end = ends[row:row + 2]
                    if begin == end:
                        continue
                    if append:
                        for index in range(end - begin):
                            position = prefixes[row] + index
                            page = pages_cpu[row, position // page_size]
                            expected_key[page, position % page_size] = fresh_key_cpu[begin + index]
                            expected_value[page, position % page_size] = fresh_value_cpu[begin + index]
                    total = full_lengths[row]
                    page_count = (total + page_size - 1) // page_size
                    selected = pages_cpu[row, :page_count].long()
                    k = expected_key[selected].flatten(0, 1)[:total].float().repeat_interleave(12, dim=1)
                    v = expected_value[selected].flatten(0, 1)[:total].float().repeat_interleave(12, dim=1)
                    scores = torch.einsum('qhd,khd->hqk', query_cpu[begin:end].float(), k) / 16.
                    mask = torch.arange(total)[None, :] > (prefixes[row] + torch.arange(end - begin))[:, None]
                    scores.masked_fill_(mask[None], -torch.inf)
                    reference[begin:end] = torch.einsum('hqk,khd->qhd', scores.softmax(-1), v).bfloat16()
                # Match the existing Qwen attention numerical gate; state remains exact.
                actual, expected = output.cpu().float(), reference.float()
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                assert relative < .02
                torch.testing.assert_close(actual, expected, rtol=.035, atol=float(expected.abs().max()) * .02)
                torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
                torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)
                assert not output[first + second:].any().item()
        finally:
            stream.synchronize(); graph.reset()
