# SPDX-License-Identifier: Apache-2.0

"""Real GGUF files through the native loader and external QUactlize kernels."""

import os
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')
pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='set FLARE_RUN_CUDA_TESTS=1 on a CUDA host')


class ProjectionPair:
    def __init__(self, source, first, second):
        self.source, self.first, self.second = source, first, second

    def __call__(self):
        self.first(self.source)
        self.second(self.source)


def assert_projection(output, source, weight):
    # FP64 oracle over the checkpoint-expanded, rounded BF16 operands. The
    # bound includes output rounding and FP32 accumulation, including cancellation.
    source, weight = source.cpu().double(), weight.cpu().double()
    expected = source @ weight.T
    magnitude = source.abs() @ weight.abs().T
    bound = expected.abs() * torch.finfo(output.dtype).eps + magnitude * (2e-6 + source.shape[1] * 2**-24)
    assert bool(((output.cpu().double() - expected).abs() <= bound + 1e-7).all())


def blocks(qtype, shape, seed):
    block, width = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, (*shape[:-1], shape[-1] // block, width), dtype=np.uint8)
    offsets = (0, 2) if qtype == gguf.GGMLQuantizationType.Q5_1 else (0,)
    for offset in offsets:
        scales = rng.uniform(-.01, .01, values.shape[:-1]).astype('<f2')
        values[..., offset:offset + 2] = scales[..., None].view(np.uint8)
    return values


@pytest.fixture
def checkpoint(tmp_path):
    from ignition.preparation.gguf import inspect_gguf_checkpoint

    gate = blocks(gguf.GGMLQuantizationType.Q5_1, (3, 4, 640), 17)
    up = blocks(gguf.GGMLQuantizationType.Q5_1, (3, 4, 640), 29)
    dense = blocks(gguf.GGMLQuantizationType.Q8_0, (8, 640), 31)
    bf16 = np.array([0x3f80, 0xc000, 0x3e80, 0x8000], dtype=np.uint16)
    tensors = [
        ('blk.0.ffn_gate_exps.weight', gate.reshape(3, 4, -1), gguf.GGMLQuantizationType.Q5_1),
        ('blk.0.ffn_up_exps.weight', up.reshape(3, 4, -1), gguf.GGMLQuantizationType.Q5_1),
        ('blk.0.hc_attn_up.weight', dense.reshape(8, -1), gguf.GGMLQuantizationType.Q8_0),
        ('norm.weight', np.arange(8, dtype=np.float32), None),
        ('bf16.weight', bf16.view(np.uint8), gguf.GGMLQuantizationType.BF16),
        ('per_layer_token_embd.weight', np.zeros((16384, 90), dtype=np.uint8), gguf.GGMLQuantizationType.IQ4_NL),
    ]
    path = tmp_path / 'model.gguf'
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    for name, data, kind in tensors:
        writer.add_tensor(name, data, raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return inspect_gguf_checkpoint(path), gate, up, dense, bf16


def test_native_reads_fuse_within_each_expert_and_preserve_plain_bits(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader

    manifest, gate, up, _, bf16 = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        assert 'per_layer_token_embd.weight' not in loader._sources
        weight = loader.load_quantized_group(('blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight'))
        assert weight.shape == (3, 8, 640) and weight.qtype == 7
        assert np.array_equal(weight.raw.cpu().numpy(), np.concatenate((gate, up), axis=1))
        plain = loader.load_tensor('norm.weight')
        bits = loader.load_tensor('bf16.weight')
        torch.testing.assert_close(plain.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
        assert np.array_equal(bits.cpu().view(torch.uint16).numpy(), bf16)
        assert loader._session.payload_bytes == gate.nbytes + up.nbytes + 32 + bf16.nbytes
        assert loader._session.bf16_expansion_bytes == 0
        assert loader._session.torch_copy_bytes == 0


def test_selected_blocks_and_experts_are_exact(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader

    manifest, gate, _, dense, _ = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        chosen = loader.load_quantized('blk.0.ffn_gate_exps.weight', experts=(1, 3), rows=(1, 4), columns=(32, 608))
        assert chosen.shape == (2, 3, 576)
        assert np.array_equal(chosen.raw.cpu().numpy(), gate[1:3, 1:4, 1:19])
        projection = loader.load_linear('blk.0.hc_attn_up', rows=(2, 5), columns=(64, 320))
        assert projection.shape == (3, 256)
        assert np.array_equal(projection.raw.cpu().numpy(), dense[2:5, 2:10])
        with pytest.raises(ValueError):
            loader.load_quantized('blk.0.hc_attn_up.weight', columns=(1, 320))
        with pytest.raises(ValueError, match='equal encoding'):
            loader.load_quantized_group(('blk.0.ffn_gate_exps.weight', 'blk.0.hc_attn_up.weight'))


def test_changed_file_is_rejected_before_queued_transfer(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader

    manifest, *_ = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        with manifest.shard_paths[0].open('r+b') as file:
            file.seek(manifest.tensors['blk.0.hc_attn_up.weight'].offset)
            file.write(b'\0\0')
        stat = manifest.shard_paths[0].stat()
        os.utime(manifest.shard_paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with pytest.raises(RuntimeError, match='source changed'):
            loader.load_quantized('blk.0.hc_attn_up.weight')


def test_pread_strided_bf16_expansion_and_staging_lifetime(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.preparation.gguf_pread import ALIGNMENT, BLOCK_BYTES

    manifest, _, _, _, bf16 = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        session = loader._session
        destination = torch.full((4, 3), -19., device='cuda', dtype=torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            session(destination[:, 1], loader._sources['bf16.weight'])
            session.flush()
        expected = torch.full((4, 3), -19., dtype=torch.float32)
        expected[:, 1] = torch.from_numpy(bf16).view(torch.bfloat16).float()
        torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
        stats = session.stats()
        assert stats['bf16_expansion_bytes'] == bf16.nbytes
        assert stats['pinned_staging_bytes'] <= 3 * BLOCK_BYTES + 2 * ALIGNMENT
        assert stats['torch_copy_bytes'] == 0
        assert not session.pending
    assert session.staging is None and session.expanded is None and not session.files


def test_pread_overlapping_destinations_rejected_before_write(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader

    manifest, *_ = checkpoint
    destination = torch.full((8,), -19., device='cuda', dtype=torch.float32)
    with pytest.raises(ValueError, match='overlapping checkpoint destinations'):
        with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
            loader._session(destination, loader._sources['norm.weight'])
            loader._session(destination[1:5], loader._sources['norm.weight'][:4])
            loader._session.flush()
    assert torch.all(destination == -19.).item()


def test_pread_multiple_blocks_keep_file_and_destination_tails(tmp_path):
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.preparation.gguf import inspect_gguf_checkpoint
    from ignition.preparation.gguf_pread import BLOCK_BYTES

    values = (np.arange(2 * (BLOCK_BYTES + 13), dtype=np.int32) % 127).astype(np.int8).reshape(2, -1)
    path = tmp_path / 'wide.gguf'
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('wide.weight', values)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with GgufWeightLoader(inspect_gguf_checkpoint(path), 'cuda:0', read_mode='pread') as loader:
        actual = loader.load_tensor('wide.weight', extent=(slice(None), slice(1, None)))
        assert np.array_equal(actual.cpu().numpy(), values[:, 1:])
        assert loader._session.payload_bytes == values[:, 1:].nbytes


def test_file_to_quantized_projection_graph(checkpoint):
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.components.gguf_projection import projection_workspace_bytes
    from ignition.components.linear import prepare_projection

    manifest, _, _, raw, _ = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        weight = loader.load_linear('blk.0.hc_attn_up')
    source = torch.randn((4, 640), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((4, 8), dtype=source.dtype, device=source.device)
    workspace = torch.empty(projection_workspace_bytes(source, weight), dtype=torch.uint8, device=source.device)
    prepared = prepare_projection(source, weight, output, session=None, name='gguf-test', workspace=workspace)
    with pytest.raises(ValueError, match='input differs'):
        prepared(source.float())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared(source)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prepared(source)
    dequant = gguf.dequantize(raw.reshape(8, -1), gguf.GGMLQuantizationType.Q8_0)
    w = torch.from_numpy(dequant).bfloat16().cuda()
    for _ in range(3):
        source.normal_()
        graph.replay()
        assert_projection(output, source, w)
        assert prepared.device_error.item() == 0
    torch.cuda.synchronize()
    del graph


@pytest.mark.parametrize('compilation', ['torch', 'flare'])
def test_compiled_projections_share_workspace_and_rebind_handles(checkpoint, compilation, tmp_path):
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.components.gguf_projection import PreparedGgufProjection, projection_workspace_bytes

    if compilation == 'flare' and importlib.util.find_spec('torch._dynamo.aot_compile_types') is None:
        pytest.skip('Flare AOT requires the locked preparation environment')
    manifest, _, _, raw, _ = checkpoint
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        weights = (loader.load_quantized('blk.0.hc_attn_up.weight'),
                   loader.load_quantized('blk.0.hc_attn_up.weight', rows=(4, 8)))

    def prepare():
        source = torch.randn((7, 640), dtype=torch.bfloat16, device='cuda')
        workspace = torch.empty(max(projection_workspace_bytes(source, w) for w in weights),
                                dtype=torch.uint8, device=source.device)
        calls = tuple(PreparedGgufProjection(source, w,
            torch.empty((7, w.out_features), dtype=source.dtype, device=source.device),
            workspace=workspace) for w in weights)
        return ProjectionPair(source, *calls)

    forward = prepare()
    if compilation == 'torch':
        compiled = torch.compile(forward, fullgraph=True)
    else:
        from ignition.preparation.aot import ForwardCache
        from ignition.preparation.execution import GraphSignature

        cache = ForwardCache(namespace={'model': 'gguf-projection-test'}, device='cuda:0', root=tmp_path)
        signature = {'pair': GraphSignature('prefill', 1, 7)}
        original = cache.prepare_many({'pair': forward}, signature)['pair']
        assert not original.cache_hit
        fresh = prepare()
        assert fresh.first.handle != forward.first.handle and fresh.second.handle != forward.second.handle
        compiled = cache.prepare_many({'pair': fresh}, signature)['pair']
        assert compiled.cache_hit
        forward = fresh
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    w = torch.from_numpy(gguf.dequantize(raw.reshape(8, -1), gguf.GGMLQuantizationType.Q8_0)).bfloat16().cuda()
    try:
        for _ in range(3):
            forward.source.normal_()
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            assert_projection(forward.first.output, forward.source, w)
            assert_projection(forward.second.output, forward.source, w[4:])
            assert forward.first.workspace.data_ptr() == forward.second.workspace.data_ptr()
            assert forward.first.device_error.item() == forward.second.device_error.item() == 0
    finally:
        graph.reset()


def test_real_shared_mtp_selected_ranges():
    from ignition.preparation.gguf import inspect_gguf_checkpoint
    from ignition.components.gguf_weights import GgufWeightLoader

    path = os.environ.get('FLARE_GGUF_MTP')
    if not path:
        pytest.skip('set FLARE_GGUF_MTP to the shared-Q8 GGUF file')
    manifest = inspect_gguf_checkpoint(Path(path))
    dense_name = 'blk.48.hc_attn_up.weight'
    expert_name = 'blk.48.ffn_down_exps.weight'
    with GgufWeightLoader(manifest, 'cuda:0', prefixes=(dense_name, expert_name), read_mode='pread') as loader:
        dense = loader.load_quantized(dense_name)
        expert = loader.load_quantized(expert_name, experts=(511, 512), rows=(0, 32))
        assert dense.shape == (10240, 320) and expert.shape == (1, 32, 640)
        assert loader._session.payload_bytes == dense.raw.nbytes + expert.raw.nbytes
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert np.array_equal(dense.raw.cpu().numpy().reshape(10240, -1), tensors[dense_name].data)
    expected = tensors[expert_name].data[511:512, :32]
    assert np.array_equal(expert.raw.cpu().numpy().reshape(expected.shape), expected)
    from ignition.components.gguf_projection import PreparedGgufProjection, projection_workspace_bytes

    source = torch.randn((5, 320), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((5, 10240), dtype=source.dtype, device=source.device)
    workspace = torch.empty(projection_workspace_bytes(source, dense), dtype=torch.uint8, device=source.device)
    projection = PreparedGgufProjection(source, dense, output, workspace=workspace)
    projection(source)
    reference_weight = torch.from_numpy(gguf.dequantize(tensors[dense_name].data,
        gguf.GGMLQuantizationType.Q8_0)).bfloat16().cuda()
    assert_projection(output, source, reference_weight)
