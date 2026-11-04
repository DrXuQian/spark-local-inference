# SPDX-License-Identifier: Apache-2.0

import os

import gguf
import numpy as np
import pytest
import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.preparation.gguf import inspect_gguf_checkpoint


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


class EmbeddingForward:
    def __init__(self, lookup):
        self.lookup = lookup

    def __call__(self):
        self.lookup()


@pytest.mark.parametrize('qtype,width', [(gguf.GGMLQuantizationType.Q8_0, 2560),
                                       (gguf.GGMLQuantizationType.IQ4_NL, 160)])
def test_gguf_rows_file_to_aot_graph(tmp_path, qtype, width):
    rng = np.random.default_rng(137)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, (31, width // block, size), dtype=np.uint8)
    raw[..., :2] = rng.uniform(-.01, .01, raw.shape[:-1]).astype('<f2')[..., None].view(np.uint8)
    raw = raw.reshape(31, -1)
    path = tmp_path / 'rows.gguf'
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('token_embd.weight', raw, raw_dtype=qtype)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with GgufWeightLoader(inspect_gguf_checkpoint(path), 'cuda:0', read_mode='pread') as loader:
        weight = loader.load_quantized('token_embd.weight')
        expanded = loader.load_dequantized('token_embd.weight')
    ids = torch.zeros(17, dtype=torch.int64, device='cuda')
    live = torch.zeros(1, dtype=torch.int32, device='cuda')
    output = torch.empty((17, width), dtype=torch.bfloat16, device='cuda')
    lookup = PreparedEmbedding(weight, ids, output, session=None, name='gguf-rows', num_rows=live)
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.execution import GraphSignature

    cache = ForwardCache(namespace={'model': 'gguf-embedding-test'}, device='cuda:0', root=tmp_path / 'cache')
    compiled = cache.prepare_many({'lookup': EmbeddingForward(lookup)},
                                 {'lookup': GraphSignature('prefill', 1, 17)})['lookup']
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    table = torch.from_numpy(gguf.dequantize(raw, qtype)).bfloat16().cuda()
    assert torch.equal(expanded.view(torch.int16), table.view(torch.int16))
    try:
        for count in (17, 3, 0, 11):
            ids.random_(0, 31)
            ids[0:2] = 30
            live.fill_(count)
            output.fill_(float('nan'))
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            assert torch.equal(output[:count].view(torch.int16), table[ids[:count]].view(torch.int16))
            assert torch.count_nonzero(output[count:]).item() == 0
            assert lookup.device_error.item() == 0
    finally:
        stream.synchronize()
        graph.reset()
