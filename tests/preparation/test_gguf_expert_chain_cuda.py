# SPDX-License-Identifier: Apache-2.0

"""Packed target/MTP expert chains, not a complete attention/speculation round.

Keep E512/top10 and eight-row chunking; reduce H/I to 256/128 so the two
synthetic GGUF files occupy about 81 MiB. No checkpoint download or b12x is used.
"""

from contextlib import nullcontext
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

gguf = pytest.importorskip('gguf')
native = pytest.importorskip('quactlize.compat.native')

from ignition.components.gguf_moe import GgufExperts, GgufMoeWorkspace, PreparedGgufRoutedExperts
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.external_session import ExternalPreparationSession
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.preparation.gguf_packing import GgufPacker
from ignition.preparation.quactlize import QuactlizeHelpersSpec
from ignition.preparation.scratch import ScratchArena


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an assigned PPU and installed QUactlize wheel or explicit test DSO')

EXPERTS, HIDDEN, INTERMEDIATE, TOPK, CAPACITY, CHUNK = 512, 256, 128, 10, 17, 8


def write_experts(path, gate_type, down_type):
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    for index, (part, n, k, qtype) in enumerate((('gate', INTERMEDIATE, HIDDEN, gate_type),
            ('up', INTERMEDIATE, HIDDEN, gate_type), ('down', HIDDEN, INTERMEDIATE, down_type))):
        kind = gguf.GGMLQuantizationType(qtype)
        block, size = gguf.GGML_QUANT_SIZES[kind]
        raw = np.random.default_rng(qtype * 100 + index).integers(
            0, 256, (EXPERTS, n, k // block, size), dtype=np.uint8)
        scale = {7: 2**-6, 8: 2**-7, 12: 2**-12}[qtype]
        raw[..., :2] = np.array([scale], dtype='<f2').view(np.uint8)
        if qtype in (7, 12):
            raw[..., 2:4] = np.array([-.25 if qtype == 7 else 2**-12], dtype='<f2').view(np.uint8)
        writer.add_tensor(f'blk.0.ffn_{part}_exps.weight', raw.reshape(EXPERTS, n, -1), raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()


class ExpertReference:
    """Decode only the selected experts, independently of the native producer."""

    def __init__(self, path):
        self.reader = gguf.GGUFReader(path)
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.cache = {}

    def expert(self, index):
        if index not in self.cache:
            values = []
            for part in ('gate', 'up', 'down'):
                tensor = self.tensors[f'blk.0.ffn_{part}_exps.weight']
                raw = np.array(tensor.data[index])
                values.append(torch.from_numpy(gguf.dequantize(raw, tensor.tensor_type)).bfloat16())
            self.cache[index] = torch.cat(values[:2]), values[2]
        return self.cache[index]

    def __call__(self, source, ids, probabilities):
        result = torch.zeros_like(source)
        for row in range(len(source)):
            live = ids[row] >= 0
            selected = ids[row, live].tolist()
            if not selected:
                continue
            weights = [self.expert(index) for index in selected]
            up = torch.stack([pair[0] for pair in weights]).float()
            down = torch.stack([pair[1] for pair in weights]).float()
            gate, value = (source[row].float() @ up.transpose(-1, -2)).bfloat16().chunk(2, dim=-1)
            intermediate = (F.silu(gate) * value).float()
            projected = (intermediate[:, None] @ down.transpose(-1, -2)).squeeze(1).bfloat16()
            result[row] = (projected.float() * probabilities[row, live, None]).sum(0).bfloat16()
        return result


@pytest.fixture(scope='module')
def weights(tmp_path_factory):
    path = os.environ.get('FLARE_TEST_PPU_MATMUL_LIBRARY')
    if path is None:
        from quactlize import get_library_path
        path = get_library_path()
    library = native.Library(path)
    library.packed = native.Library(path, packed_route=0)
    packer = GgufPacker(library)
    directory = tmp_path_factory.mktemp('gguf-expert-chain')
    loaded, references = [], []
    for name, gate_type, down_type in (('target', 12, 7), ('mtp', 8, 8)):
        file = directory / (name + '.gguf')
        write_experts(file, gate_type, down_type)
        with GgufWeightLoader(inspect_gguf_checkpoint(file), 'cuda:0', read_mode='pread') as loader:
            up = packer(loader.load_quantized_group(('blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight')))
            down = packer(loader.load_quantized('blk.0.ffn_down_exps.weight'))
        loaded.append(GgufExperts(up, down))
        references.append(ExpertReference(file))
    assert sum(file.stat().st_size for file in directory.glob('*.gguf')) < 90 << 20
    return path, library, tuple(loaded), tuple(references)


class ExpertChain:
    def __init__(self, rows, tensors, weights, workspace, library, session):
        self.source, self.target_output, self.output = (tensors[name][:rows] for name in ('source', 'target', 'mtp'))
        self.live = tensors['live']
        self.target_logits, self.mtp_logits = (tensors[name][:rows] for name in ('target_logits', 'mtp_logits'))
        self.target_routes, self.mtp_routes = (tensors[name].rows(rows) for name in ('target_routes', 'mtp_routes'))
        self.target_router = PreparedRouter(self.target_logits, self.target_routes, session=session,
            name='target.route', num_valid_tokens=self.live)
        self.mtp_router = PreparedRouter(self.mtp_logits, self.mtp_routes, session=session,
            name='mtp.route', num_valid_tokens=self.live)
        self.target = PreparedGgufRoutedExperts(self.source, weights[0], self.target_routes.ids,
            self.target_routes.weights, self.target_output, workspace=workspace, projection_library=library,
            session=session, name='target.experts')
        self.mtp = PreparedGgufRoutedExperts(self.target_output, weights[1], self.mtp_routes.ids,
            self.mtp_routes.weights, self.output, workspace=workspace, projection_library=library,
            session=session, name='mtp.experts')

    def __call__(self):
        self.target_router.error.zero_(); self.mtp_router.error.zero_()
        self.target.errors.zero_(); self.mtp.errors.zero_()
        self.target_router(self.target_logits)
        self.target()
        self.mtp_router(self.mtp_logits)
        self.mtp()
        return self.output


def allocate():
    tensors = {name: torch.zeros((CAPACITY, HIDDEN), dtype=torch.bfloat16, device='cuda:0')
               for name in ('source', 'target', 'mtp')}
    tensors.update({name: torch.zeros((CAPACITY, EXPERTS), device='cuda:0') for name in ('target_logits', 'mtp_logits')})
    tensors.update({name: RouterScratch.allocate(CAPACITY, 'cuda:0', experts=EXPERTS, top_k=TOPK)
                    for name in ('target_routes', 'mtp_routes')})
    tensors['live'] = torch.zeros((), dtype=torch.int32, device='cuda:0')
    return tensors


def assert_output(actual, expected):
    # Same native16 projection/SwiGLU/reduction bounds as the existing GGUF MoE
    # reference tests. Each stage is checked against its actual BF16 input.
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative.item() < .008
    torch.testing.assert_close(actual, expected, rtol=.015, atol=expected.float().abs().max().item() * .01 + 1e-6)


def assert_routes(logits, routes, live):
    expected_ids = torch.full((len(logits), TOPK), -1, dtype=torch.int32)
    expected_weights = torch.zeros((len(logits), TOPK))
    for row in range(live if 0 <= live <= len(logits) else 0):
        if not torch.isfinite(logits[row]).all():
            continue
        ids = EXPERTS - 1 - logits[row].flip(0).argsort(descending=True, stable=True)[:TOPK]
        expected_ids[row] = ids.int()
        expected_weights[row] = logits[row, ids].softmax(0)
    torch.testing.assert_close(routes.ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(routes.weights.cpu(), expected_weights, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('compilation', ['none', 'catalog'])
@torch.no_grad()
def test_packed_expert_chain_catalog_and_shared_scratch(weights, tmp_path, monkeypatch, compilation):
    from ignition.preparation import aot
    from ignition.preparation.compile_pool import CompilePool

    path, library, packed, references = weights
    torch.manual_seed(90217)
    def forbidden(*args, **kwargs):
        raise AssertionError('a cached expert chain must not compile again')
    stream = torch.cuda.Stream(device=0)
    previous_handles = None
    for generation in range(2 if compilation == 'catalog' else 1):
        graphs, compiled, forwards = {}, {}, {}
        helpers = QuactlizeHelpersSpec.read(path).load('cuda:0')
        try:
            with ExternalPreparationSession(device='cuda:0', checkpoint='synthetic-gguf-expert-chain',
                    namespace={'qualification': 'packed-expert-chain-v1'}, compile_workers=1) as session:
                session.quactlize_helpers = helpers
                tensors = allocate()
                workspace = max((GgufMoeWorkspace.allocate(experts, chunk_rows=CHUNK, top_k=TOPK,
                    projection_library=library) for experts in packed), key=lambda value: value.matmul.nbytes)
                forwards = {f'chain-{rows}': ExpertChain(rows, tensors, packed, workspace, library, session)
                            for rows in (1, 7, CAPACITY)}
                handles = (helpers.handle, *(op.handle for item in forwards.values()
                    for experts in (item.target, item.mtp) for _, _, up, down in experts.calls for op in (up, down)))
                if previous_handles is not None:
                    assert all(old != new for old, new in zip(previous_handles, handles, strict=True))
                previous_handles = handles
                signatures = {name: GraphSignature('prefill', 1, forward.source.shape[0])
                              for name, forward in forwards.items()}
                session.register_scratch(ScratchArena('cuda:0', overlap=False))
                session.tune()
                with monkeypatch.context() if generation else nullcontext() as cached:
                    if generation:
                        cached.setattr(aot, 'aot_forward', forbidden)
                        cached.setattr(CompilePool, 'submit_payload', forbidden)
                    graphs, compiled = capture_forwards(forwards, signatures, session=session,
                        device='cuda:0', stream=stream, pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path,
                        compile_decoder=compilation == 'catalog', retain_graphs=True, report_catalog=False)
                if compiled:
                    assert all(bound.cache_hit == bool(generation) for bound in compiled.values())
                session.freeze()
                with torch.cuda.stream(stream):
                    for step, (rows, live, poisoned_logits) in enumerate(((17, 17, False), (1, 1, False),
                            (7, 3, False), (17, 9, True), (1, 0, False), (17, -1, False), (7, 8, False), (7, 7, False))):
                        forward = forwards[f'chain-{rows}']
                        tensors['source'].normal_(std=.1)
                        tensors['source'][max(live, 0):].fill_(float('nan'))
                        tensors['target'].fill_(-987); tensors['mtp'].fill_(-987)
                        tensors['live'].fill_(live)
                        rng = np.random.default_rng(718 + step)
                        for name in ('target_logits', 'mtp_logits'):
                            scores = torch.from_numpy(rng.integers(-4, 5, (CAPACITY, EXPERTS)).astype(np.float32)) / 8
                            if step == 0:
                                scores.zero_()  # Tie order must select IDs 511..502.
                            else:
                                for row in range(rows):
                                    chosen = (torch.arange(TOPK) * 47 + step * 53 + row * 19) % EXPERTS
                                    scores[row, chosen] = 4
                            tensors[name].copy_(scores)
                        if poisoned_logits:
                            tensors['target_logits'][0, 0] = float('nan')
                        tensors['target_logits'][max(live, 0):].fill_(float('nan'))
                        tensors['mtp_logits'][max(live, 0):].fill_(float('nan'))
                        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                        graphs[f'chain-{rows}'].replay()
                        stream.synchronize()
                        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                        bad_count = not 0 <= live <= rows
                        assert forward.target_router.error.item() == (4 if bad_count else 2 if poisoned_logits else 0)
                        assert forward.mtp_router.error.item() == (4 if bad_count else 0)
                        assert not torch.count_nonzero(forward.target.errors).item()
                        assert not torch.count_nonzero(forward.mtp.errors).item()
                        for logits, routes in ((forward.target_logits, forward.target_routes),
                                               (forward.mtp_logits, forward.mtp_routes)):
                            assert_routes(logits.cpu(), routes, live)
                        for reference, source, routes, output in ((references[0], forward.source,
                                forward.target_routes, forward.target_output), (references[1], forward.target_output,
                                forward.mtp_routes, forward.output)):
                            expected = reference(source.cpu(), routes.ids.cpu(), routes.weights.cpu())
                            assert_output(output.cpu(), expected)
                            assert not torch.count_nonzero(output[max(live, 0):]).item()
                        assert torch.equal(tensors['target'][rows:], torch.full_like(tensors['target'][rows:], -987))
                        assert torch.equal(tensors['mtp'][rows:], torch.full_like(tensors['mtp'][rows:], -987))
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    for _ in range(20):
                        graphs['chain-7'].replay()
                    stream.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
        finally:
            stream.synchronize()
            for graph in graphs.values():
                graph.reset()
            compiled.clear(); forwards.clear()
            forward = None
            helpers.close()
