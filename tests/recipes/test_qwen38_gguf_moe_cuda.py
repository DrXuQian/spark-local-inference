# SPDX-License-Identifier: Apache-2.0

"""Complete Qwen shared/routed MoE with real shared-Q8 MTP checkpoint weights."""

import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession

from ignition.components.gguf_moe import GgufMoeWorkspace
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader
from ignition.recipes.qwen38.moe import PreparedQwenMoe
from tests.components.test_gguf_moe_cuda import assert_moe, reference


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit CUDA assignment required')


@pytest.fixture(scope='module')
def checkpoint():
    path = os.environ.get('FLARE_GGUF_MTP')
    if not path:
        pytest.skip('set FLARE_GGUF_MTP to the real shared-Q8 MTP file')
    manifest = inspect_gguf_checkpoint(Path(path))
    with QwenGgufWeightLoader(manifest, 'cuda:0', read_mode='pread', prefixes=('blk.48.ffn_',)) as loader:
        weights = loader.load_moe(48)
    reader = gguf.GGUFReader(path)
    return weights, {tensor.name: tensor for tensor in reader.tensors}


class QwenMoeForward:
    def __init__(self, moe, source):
        self.moe, self.source = moe, source

    def __call__(self):
        self.moe(self.source)


@pytest.mark.parametrize('tokens', [1, 7])
def test_actual_mtp_moe_graph(checkpoint, tokens, tmp_path):
    weights, tensors = checkpoint
    source = torch.randn((tokens, 2560), dtype=torch.bfloat16, device='cuda') * .2
    workspace = GgufMoeWorkspace.allocate(weights.experts, chunk_rows=4, top_k=10)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        moe = PreparedQwenMoe(source, weights, session=session, name='mtp-gguf-moe',
            workspace=workspace, projection_workspace=workspace.matmul)
        from ignition.preparation.aot import ForwardCache
        from ignition.preparation.execution import GraphSignature

        forward = QwenMoeForward(moe, source)
        cache = ForwardCache(namespace={'model': 'qwen-mtp-gguf-moe-test'}, device='cuda:0', root=tmp_path)
        compiled = cache.prepare_many({'moe': forward}, {'moe': GraphSignature('prefill', 1, tokens)})['moe']
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=stream):
            compiled()
        try:
            for _ in range(2):
                source.normal_(std=.2)
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                logits = F.linear(source.float(), weights.router).bfloat16()
                torch.testing.assert_close(moe.router_logits, logits, rtol=.01, atol=.001)
                ids = 511 - moe.router_logits.flip(-1).float().argsort(dim=-1, descending=True, stable=True)[:, :10]
                assert torch.equal(moe.router.ids, ids.int())
                probabilities = moe.router_logits.float().gather(1, ids).softmax(-1)
                torch.testing.assert_close(moe.router.weights, probabilities, rtol=2e-6, atol=1e-7)
                selected, local_ids = torch.unique(ids.cpu(), return_inverse=True)

                def expand(name, selection=None):
                    tensor = tensors[f'blk.48.{name}.weight']
                    raw = tensor.data if selection is None else np.asarray(tensor.data)[selection.numpy()]
                    return torch.from_numpy(gguf.dequantize(raw, tensor.tensor_type)).bfloat16().cuda()

                up = torch.cat((expand('ffn_gate_exps', selected), expand('ffn_up_exps', selected)), dim=1)
                down = expand('ffn_down_exps', selected)
                routed = reference(source, local_ids.to(device=source.device, dtype=torch.int32),
                                   probabilities, (up, down))
                shared_gate = (source.float() @ expand('ffn_gate_shexp').float().T).bfloat16()
                shared_up = (source.float() @ expand('ffn_up_shexp').float().T).bfloat16()
                shared = ((F.silu(shared_gate) * shared_up).float() @ expand('ffn_down_shexp').float().T).bfloat16()
                gate = (source.float() @ weights.shared.gate.float().T).bfloat16().sigmoid()
                assert_moe(moe.output, routed + shared * gate)
                assert torch.count_nonzero(moe.experts.errors).item() == 0
        finally:
            stream.synchronize()
            graph.reset()
