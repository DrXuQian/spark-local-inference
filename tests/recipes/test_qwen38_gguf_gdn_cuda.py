# SPDX-License-Identifier: Apache-2.0

"""Qualify target GDN layout reversal using a complete, actual GGUF shard."""

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import gguf
import numpy as np
import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.gated_delta import GDNScratch, GDNState, PreparedGDN
from ignition.components.recurrent import DeltaMetadata
from ignition.preparation.aot import ForwardCache
from ignition.preparation.execution import GraphSignature
from ignition.preparation.gguf import GgufCheckpoint, GgufShardMetadata, GgufTensorMetadata, file_identity
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader
from tests.recipes.test_qwen38_gdn import _metadata, _numerical, _prefill_checks


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='explicit CUDA assignment required')


@pytest.fixture(params=('b12x', 'actlize'))
def recurrence_library(request):
    if request.param == 'b12x':
        return None
    path = os.environ.get('FLARE_ACTLIZE_LIBRARY')
    if not path:
        pytest.skip('actual actlizeLA SM120 serving library required')
    from actlize_la.serving import NativeLibrary

    return NativeLibrary(path, target='cuda_sm120')


@pytest.fixture(scope='module')
def checkpoint():
    path, frontend = os.environ.get('FLARE_GGUF_TARGET_SHARD'), os.environ.get('FLARE_GGUF_FRONTEND')
    if not path or not frontend:
        pytest.skip('actual target shard and pinned frontend assets required')
    path = Path(path).resolve()
    # This component fixture is intentionally a shard, never full-model preflight.
    reader = gguf.GGUFReader(path)
    inventory = {}
    tensors = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith('blk.44.') or '_exps.' in tensor.name:
            continue
        shape = tuple(int(n) for n in reversed(tensor.shape))
        block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        storage = shape if block == 1 else (*shape[:-1], shape[-1] // block, size)
        inventory[tensor.name] = GgufTensorMetadata(tensor.name, path.name, tensor.data_offset,
            tensor.n_bytes, shape, tensor.tensor_type.name if block == 1 else 'U8',
            tensor.tensor_type.name, storage, 'decoder_other')
        tensors[tensor.name] = tensor
    header = hashlib.sha256(reader.data[:reader.data_offset]).hexdigest()
    manifest = GgufCheckpoint(path.parent, MappingProxyType(inventory),
        (GgufShardMetadata(path.name, path.stat().st_size, reader.data_offset, header, len(inventory), file_identity(path)),),
        MappingProxyType({'general.architecture': 'qwen4exp'}), None)
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

    config = Qwen4ExpTextConfig.from_dict(json.loads((Path(frontend) / 'config.json').read_text())['text_config'])
    with torch.device('meta'):
        module = Qwen4ExpTextGatedDeltaNet(config, layer_idx=44)
    with QwenGgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        weights = loader.load_gdn(module, 44)

    def expand(name):
        tensor = tensors['blk.44.' + name]
        return torch.from_numpy(np.array(gguf.dequantize(tensor.data, tensor.tensor_type)))

    return module.eval().requires_grad_(False), weights, expand


class Forward:
    def __init__(self, operator, source, commit=None):
        self.operator, self.source, self.commit = operator, source, commit

    def __call__(self):
        self.operator(self.source)
        if self.commit is not None:
            for operation in self.commit:
                operation()


class LayerForward:
    def __init__(self, operations, source, commits):
        self.operations, self.source, self.commits = tuple(operations), source, commits

    def __call__(self):
        for operation in self.operations:
            operation(self.source)
        for commit in self.commits:
            commit()


def test_target_gdn_inverse_head_layout(checkpoint):
    module, weights, expand = checkpoint
    heads = torch.tensor([(head % 3) * 16 + head // 3 for head in range(48)])
    channels = (heads[:, None] * 128 + torch.arange(128)).flatten()
    qkv = expand('attn_qkv.weight').bfloat16()
    expected = torch.cat((qkv[:4096], qkv[4096:][channels], expand('attn_gate.weight').bfloat16()[channels]))
    assert torch.equal(weights['in_proj_qkvz'].cpu(), expected)
    assert torch.equal(weights['in_proj_ba'].cpu(), torch.cat([expand(f'ssm_{part}.weight')[heads] for part in ('beta', 'alpha')]))
    assert torch.equal(weights['out_proj'].cpu(), expand('ssm_out.weight').bfloat16()[:, channels])
    conv = expand('ssm_conv1d.weight')
    assert torch.equal(module.conv1d.weight.cpu().squeeze(1), torch.cat((conv[:4096], conv[4096:][channels])))
    assert torch.equal(-module.A_log.cpu().exp(), expand('ssm_a')[heads])
    assert torch.equal(module.dt_bias.cpu(), expand('ssm_dt.bias')[heads])


@pytest.mark.parametrize('tokens', [1, 17, 64])
def test_target_gdn_prefill_graph(checkpoint, tokens, tmp_path, recurrence_library):
    module, weights, _ = checkpoint
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device='cuda')
    scratch = GDNScratch.allocate(tokens, torch.device('cuda:0'))
    state = GDNState.allocate(3, torch.device('cuda:0'))
    metadata = DeltaMetadata.allocate(1, torch.device('cuda:0'))
    arena = ScratchArena('cuda:0')
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=0) as session:
        op = PreparedGDN(module, weights, source, scratch, state, metadata, mode='prefill',
            session=session, name='target.gguf.gdn', workspace=arena, projection_workspace=arena,
            actlize_library=recurrence_library)
        arena.finalize(); arena.freeze()
        cache = ForwardCache(namespace={'model': 'target-gguf-gdn'}, device='cuda:0', root=tmp_path)
        forward = cache.prepare_many({'gdn': Forward(op, source)}, {'gdn': GraphSignature('prefill', 1, tokens)})['gdn']
        forward(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph): forward()
        try:
            import copy
            from types import SimpleNamespace

            convolution_weight = module.conv1d.weight.cpu()
            assert torch.equal(convolution_weight.bfloat16().float(), convolution_weight)
            reference = SimpleNamespace(A_log=module.A_log.cpu(), dt_bias=module.dt_bias.cpu(),
                conv1d=SimpleNamespace(weight=convolution_weight.bfloat16()), norm=copy.deepcopy(module.norm).cpu())
            for live in (tokens, max(1, tokens // 2), 0):
                source.normal_(std=.2)
                initial = state.recurrent.cpu().clone()
                convolution = state.convolution.cpu().clone()
                _metadata(metadata, [live], [1], fresh=False, initial=[1])
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert all(not error.any().item() for error in op.device_errors)
                if live:
                    _prefill_checks(op, reference, initial, convolution)
                    _numerical(scratch.output[:live], (scratch.gated[:live].float() @ weights['out_proj'].float().T).bfloat16(), limit=.004)
                else:
                    assert torch.equal(state.recurrent.cpu(), initial)
                    assert torch.equal(state.convolution.cpu(), convolution)
        finally:
            torch.cuda.synchronize(); graph.reset()


@pytest.mark.parametrize('recurrence_library', ['actlize'], indirect=True)
def test_actlize_retains_each_layers_records_through_shared_scratch(checkpoint, recurrence_library, tmp_path):
    from ignition.components.actlize_gdn import prepare_layer_commits
    from ignition.components.gdn_recovery import GDNLayerRecords
    from tests.recipes.test_qwen38_gdn_recovery_cuda import _reference

    module, weights, _ = checkpoint
    device = torch.device('cuda:0')
    source = torch.zeros((6, 2560), dtype=torch.bfloat16, device=device)
    scratch = GDNScratch.allocate(6, device)
    states = {layer: GDNState.allocate(9, device, speculative_tokens=7) for layer in (0, 1)}
    metadata = DeltaMetadata.allocate(2, device, verification_columns=8)
    arena = ScratchArena(device)
    records = GDNLayerRecords.allocate(states, 2, device, key_heads=16, value_heads=48)
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operations = {layer: PreparedGDN(module, weights, source, scratch, state, metadata,
            mode='decode', session=session, name=f'gguf.layers.{layer}', workspace=arena,
            projection_workspace=arena, records=records[layer], actlize_library=recurrence_library)
            for layer, state in states.items()}
        commits = prepare_layer_commits(states[0].recurrent.view(torch.uint8).reshape(9, -1),
            {layer: state.recurrent for layer, state in states.items()}, records, metadata, counts, operations)
        arena.finalize(); arena.freeze()
        forward = ForwardCache(namespace={'model': 'gguf-actlize-shared-records'}, device=device,
            root=tmp_path).prepare_many({'verify': LayerForward(operations.values(), source, commits)},
            {'verify': GraphSignature('speculative', 2, 6)})['verify']
        forward(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            source.normal_(std=.2)
            metadata.query_start_loc.copy_(torch.tensor([0, 6, 6], dtype=torch.int32, device=device))
            metadata.state_indices[:, 0].copy_(torch.tensor([1, 2], device=device))
            metadata.convolution_slots.copy_(torch.tensor([1, 2], device=device))
            metadata.accepted_tokens.fill_(1)
            metadata.num_seqs.fill_(2); metadata.num_tokens.fill_(6)
            initial, convolution, expected = {}, {}, {}
            for layer, operation in operations.items():
                state = states[layer]
                initial[layer] = torch.randn_like(state.recurrent) * .01
                initial[layer][0].zero_()
                convolution[layer] = torch.randn_like(state.convolution) * .1
                state.recurrent.copy_(initial[layer]); state.convolution.copy_(convolution[layer])
                operation(source)
                reference_state = initial[layer].clone()
                _reference(dict(mixed_qkv=scratch.convolved_qkv, a=scratch.a, b=scratch.b,
                    A_log=module.A_log, dt_bias=module.dt_bias, recurrent_state=reference_state,
                    z=scratch.z.view(6, 48, 128), norm_weight=module.norm.weight,
                    query_start_loc=metadata.query_start_loc, num_accepted_tokens=metadata.accepted_tokens,
                    state_indices=torch.arange(1, 9, device=device, dtype=torch.int64).repeat(2, 1),
                    num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
                    output=torch.empty((6, 48, 128), dtype=torch.bfloat16, device=device)), 8)
                expected[layer] = reference_state
            for consumed in range(7):
                for layer, state in states.items():
                    state.recurrent.copy_(initial[layer]); state.convolution.copy_(convolution[layer])
                counts.copy_(torch.tensor([consumed, 0], dtype=torch.int32, device=device))
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                for layer, state in states.items():
                    assert all(not error.any().item() for error in operations[layer].device_errors)
                    wanted = expected[layer][consumed] if consumed else initial[layer][1]
                    torch.testing.assert_close(state.recurrent[1], wanted, rtol=1e-5, atol=1e-6)
                    assert torch.equal(state.recurrent[0], initial[layer][0])
                    assert torch.equal(state.recurrent[2:], initial[layer][2:])
                    assert torch.equal(state.convolution[0], convolution[layer][0])
                    assert torch.equal(state.convolution[2:], convolution[layer][2:])
                    if not consumed:
                        assert torch.equal(state.convolution, convolution[layer])
        finally:
            torch.cuda.synchronize(); graph.reset()


@pytest.mark.parametrize('width', [1, 6])
def test_actual_gguf_decode_commit_matches_flare_checkpoints(checkpoint, width, tmp_path, recurrence_library):
    from ignition.components.gdn_recovery import GDNLayerRecords
    from ignition.components.actlize_gdn import prepare_layer_commits
    from tests.recipes.test_qwen38_gdn_recovery_cuda import _reference

    module, weights, _ = checkpoint
    device = torch.device('cuda:0')
    source = torch.zeros((width, 2560), dtype=torch.bfloat16, device=device)
    scratch = GDNScratch.allocate(width, device)
    state = GDNState.allocate(9, device, speculative_tokens=7)
    metadata = DeltaMetadata.allocate(1, device, verification_columns=8)
    arena = ScratchArena(device)
    records = GDNLayerRecords.allocate((44,), 1, device, key_heads=16, value_heads=48)
    counts = torch.zeros(1, dtype=torch.int32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        op = PreparedGDN(module, weights, source, scratch, state, metadata, mode='decode',
            session=session, name='target.gguf.verify', workspace=arena, projection_workspace=arena,
            records=records[44] if width > 1 else None, in_place=width == 1,
            actlize_library=recurrence_library)
        commit = None if width == 1 else prepare_layer_commits(
            state.recurrent.view(torch.uint8).reshape(9, -1), {44: state.recurrent}, records,
            metadata, counts, {44: op})
        arena.finalize(); arena.freeze()
        cache = ForwardCache(namespace={'model': 'gguf-commit-checkpoints', 'width': width},
            device=device, root=tmp_path)
        forward = cache.prepare_many({'decode': Forward(op, source, commit)},
            {'decode': GraphSignature('speculative', 1, width)})['decode']
        forward(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            source.normal_(std=.2)
            first_source = source.clone()
            initial = torch.randn_like(state.recurrent) * .01
            initial[0].zero_()
            convolution = torch.randn_like(state.convolution) * .1
            state.recurrent.copy_(initial); state.convolution.copy_(convolution)
            metadata.query_start_loc[1].fill_(width)
            metadata.state_indices[0, 0].fill_(1)
            metadata.convolution_slots.fill_(1)
            metadata.accepted_tokens.fill_(1)
            metadata.num_seqs.fill_(1); metadata.num_tokens.fill_(width)
            graph.replay(); torch.cuda.synchronize()
            if width > 1:
                assert torch.equal(state.recurrent, initial), 'verification changed committed state'
            reference_state = initial.clone()
            expected = torch.empty((width, 48, 128), dtype=torch.bfloat16, device=device)
            reference_indices = torch.arange(1, 9, device=device, dtype=torch.int32).reshape(1, 8)
            _reference(dict(mixed_qkv=scratch.convolved_qkv, a=scratch.a, b=scratch.b,
                A_log=module.A_log, dt_bias=module.dt_bias, recurrent_state=reference_state,
                z=scratch.z.view(width, 48, 128), norm_weight=module.norm.weight,
                query_start_loc=metadata.query_start_loc, num_accepted_tokens=metadata.accepted_tokens,
                state_indices=reference_indices, num_seqs=metadata.num_seqs,
                num_tokens=metadata.num_tokens, output=expected), 8)
            for consumed in ((1,) if width == 1 else range(width + 1)):
                state.recurrent.copy_(initial); state.convolution.copy_(convolution)
                source.copy_(first_source); metadata.accepted_tokens.fill_(1)
                counts.fill_(consumed); torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                assert all(not error.any().item() for error in op.device_errors)
                if op.actlize is not None and consumed == 0:
                    assert torch.equal(state.convolution, convolution)
                assert torch.equal(state.recurrent[0], initial[0])
                assert torch.equal(state.recurrent[2:], initial[2:])
                wanted = reference_state[consumed] if consumed else initial[1]
                torch.testing.assert_close(state.recurrent[1], wanted, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(scratch.gated.view_as(expected), expected, rtol=2**-6, atol=2e-2)
                assert (scratch.gated.view_as(expected) == expected).float().mean() > .99
                if op.actlize is not None and width > 1:
                    import torch.nn.functional as F

                    history = convolution[1, :, :3].cpu()
                    if consumed:
                        history = torch.cat((history, scratch.projected_qkv.cpu().T), -1)[:, consumed:consumed + 3]
                    # Retry from the committed prefix with new tokens. ShortConv
                    # must select the positive prefix once, or the old history
                    # after zero consumption; the physical tail is not shifted twice.
                    source.normal_(std=.2); counts.fill_(width)
                    metadata.accepted_tokens.fill_(max(1, consumed))
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                    assert all(not error.any().item() for error in op.device_errors)
                    projected = torch.cat((history, scratch.projected_qkv.cpu().T), -1)
                    convolved = F.silu(F.conv1d(projected[None], module.conv1d.weight.cpu().bfloat16(),
                                               groups=projected.shape[0]))[0].T
                    _numerical(scratch.convolved_qkv, convolved, limit=.004)
                    assert torch.equal(state.convolution[0], convolution[0])
                    assert torch.equal(state.convolution[2:], convolution[2:])
        finally:
            torch.cuda.synchronize(); graph.reset()
