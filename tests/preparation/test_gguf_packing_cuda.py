# SPDX-License-Identifier: Apache-2.0

"""Tiny GGUF -> native GPU shuffle -> prepared packed projection and graph replay."""

import gc
import os
import weakref

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')
native = pytest.importorskip('quactlize.compat.native')

from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.ppu_gguf_projection import PreparedPpuGgufProjection, projection_sizes
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.preparation.gguf_packing import GgufPacker
from ignition.preparation import provenance
from tests.components.gguf_reference import dequantize


pytestmark = pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires a real PPU and installed QUactlize PPU wheel or explicit test DSO')


class PackedForward:
    def __init__(self, library, weight, rows):
        k, n = weight.in_features, weight.out_features
        shape = (rows, 2, k) if k == 640 else (rows, k)
        self.source = torch.randn(shape, dtype=torch.bfloat16, device=weight.device) * .2
        self.ids = torch.zeros((rows, 2), dtype=torch.int32, device=weight.device)
        self.output = torch.empty((rows, 2, n), dtype=torch.bfloat16, device=weight.device)
        _, sizes = projection_sizes(library, self.source, weight, self.output, ids=self.ids)
        workspace = torch.empty(sizes.workspace_bytes, dtype=torch.uint8, device=weight.device)
        self.projection = PreparedPpuGgufProjection(library, self.source, weight, self.output,
                                                  ids=self.ids, workspace=workspace)

    def __call__(self):
        self.projection.device_error.zero_()
        self.projection(self.source)
        return self.output


CASES = [(q, m, 'none', *((2560, 640) if q in (7, 8) else (1280, 2560)))
         for q in (7, 8, 12, 13) for m in (1, 9)]
CASES += [(7, m, 'catalog', 2560, 640) for m in (1, 9)]
CASES += [(42, m, mode, n, k) for m in (1, 9) for mode in ('none', 'catalog')
          for n, k in ((1280, 2560), (2560, 640))]


@pytest.mark.parametrize('qtype,rows,compilation,n,k', CASES)
@torch.no_grad()
def test_file_pack_and_prepared_projection_replay(tmp_path, monkeypatch, qtype, rows, compilation, n, k):
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.execution import GraphSignature

    path = os.environ.get('FLARE_TEST_PPU_MATMUL_LIBRARY')
    library = native.Library(path)
    library.packed = native.Library(path, packed_route=0)
    packer = GgufPacker(library)
    kind = gguf.GGMLQuantizationType(qtype)
    block, size = gguf.GGML_QUANT_SIZES[kind]
    rng = np.random.default_rng(645 + qtype)
    raw = rng.integers(0, 256, (3, n, k // block, size), dtype=np.uint8)
    for offset in ((0,) if qtype in (8, 42) else (0, 2)):
        values = rng.uniform(-.003, .003, raw.shape[:-1]).astype('<f2')
        raw[..., offset:offset + 2] = values[..., None].view(np.uint8)
    reference = torch.from_numpy(dequantize(raw.reshape(3 * n, -1), kind)).reshape(3, n, k).bfloat16().double()
    file = tmp_path / 'experts.gguf'
    writer = gguf.GGUFWriter(file, 'qwen4exp')
    writer.add_tensor('experts.weight', raw.reshape(3, n, -1), raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        with GgufWeightLoader(inspect_gguf_checkpoint(file), 'cuda:0', read_mode='pread') as loader:
            original = loader.load_quantized('experts.weight')
            raw_owner = weakref.ref(original.raw)
            weight = packer(original)
            del original
        gc.collect()
        assert raw_owner() is None and weight.raw is None and packer.pending is None
        planes = tuple(value for value in (weight.packed.low, weight.packed.high, weight.packed.units) if value is not None)
        assert sum(value.nbytes for value in planes) == raw.nbytes
        for plane in planes:
            metadata = provenance.lookup(plane)
            assert metadata['packing'] == 'gguf_canonical_kpack'
            assert metadata['packing_parameters']['arrangement'] == weight.packed.arrangement

        def forbidden(*args, **kwargs):
            raise AssertionError('packed forwards must not repack or recompile after preparation')
        monkeypatch.setattr(packer, 'repack', forbidden)
        forward = PackedForward(library, weight, rows)
        execute = forward
        if compilation == 'catalog':
            from ignition.preparation import aot
            from ignition.preparation.compile_pool import CompilePool

            cache = ForwardCache(namespace={'qualification': 'gguf-packed', 'qtype': qtype}, device='cuda:0',
                                 root=tmp_path / 'cache', compile_workers=1)
            signatures = {'projection': GraphSignature('prefill', rows, 1)}
            first = cache.prepare_many({'projection': forward}, signatures)['projection']
            assert not first.cache_hit
            fresh = PackedForward(library, weight, rows)
            assert fresh.projection.handle != forward.projection.handle
            with monkeypatch.context() as cached:
                cached.setattr(aot, 'aot_forward', forbidden)
                cached.setattr(CompilePool, 'submit_payload', forbidden)
                execute = cache.prepare_many({'projection': fresh}, signatures)['projection']
            assert execute.cache_hit
            forward = fresh
        execute(); stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            execute()
        try:
            for route in ([2, 2], [-1, -1], [3, 1], [-2, 0], [0, 1]):
                forward.ids.copy_(torch.tensor([route] * rows, dtype=torch.int32, device='cuda'))
                forward.source.normal_()
                if route == [-1, -1]:
                    forward.source.fill_(float('nan'))
                graph.replay(); stream.synchronize()
                assert forward.projection.device_error.item() == int(any(expert < -1 or expert >= 3 for expert in route))
                source = forward.source.cpu().double()
                for slot, expert in enumerate(route):
                    actual = forward.output[:, slot].cpu().double()
                    if not 0 <= expert < 3:
                        assert torch.count_nonzero(actual) == 0
                        continue
                    x = source[:, slot] if source.ndim == 3 else source
                    expected, magnitude = x @ reference[expert].T, x.abs() @ reference[expert].abs().T
                    bound = expected.abs() * torch.finfo(torch.bfloat16).eps + magnitude * (2e-6 + k * 2**-24)
                    assert bool(((actual - expected).abs() <= bound + 1e-7).all())
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            for _ in range(20):
                graph.replay()
            stream.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
        finally:
            stream.synchronize()
            graph.reset()
