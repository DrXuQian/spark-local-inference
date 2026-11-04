# SPDX-License-Identifier: Apache-2.0

"""Opt-in actual PPU prepare/capture/replay; never substitutes NVIDIA matmul."""

import os

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')
native = pytest.importorskip('quactlize.compat.native')

from ignition.components.gguf_weights import GgufWeight
from ignition.components.ppu_gguf_projection import PreparedPpuGgufProjection, projection_sizes


pytestmark = pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not os.environ.get('FLARE_TEST_PPU_MATMUL_LIBRARY'),
    reason='assign a real PPU and its FLARE_TEST_PPU_MATMUL_LIBRARY explicitly')


@pytest.mark.parametrize('qtype', [7, 8])
@pytest.mark.parametrize('m', [1, 9])
def test_native_prepare_and_replay_with_changing_routes_and_addresses(qtype, m):
    library = native.Library(os.environ['FLARE_TEST_PPU_MATMUL_LIBRARY'])
    kind = gguf.GGMLQuantizationType(qtype)
    block, size = gguf.GGML_QUANT_SIZES[kind]
    n = k = 640
    rng = np.random.default_rng(651)
    raw = rng.integers(0, 256, (3, n, k // block, size), dtype=np.uint8)
    for offset in ((0, 2) if qtype == 7 else (0,)):
        scales = rng.uniform(-.005, .005, raw.shape[:-1]).astype('<f2')
        raw[..., offset:offset + 2] = scales[..., None].view(np.uint8)
    reference = torch.from_numpy(gguf.dequantize(raw.reshape(3 * n, -1), kind)).view(3, n, k).bfloat16().double()
    weight = GgufWeight(torch.from_numpy(raw).cuda(), kind.name, qtype, (3, n, k))
    source = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * .2
    ids = torch.zeros((m, 2), dtype=torch.int32, device='cuda')
    output = torch.empty((m, 2, n), dtype=source.dtype, device=source.device)
    _, sizes = projection_sizes(library, source, weight, output, ids=ids)
    workspace = torch.empty(sizes.workspace_bytes, dtype=torch.uint8, device=source.device)
    prepared = PreparedPpuGgufProjection(library, source, weight, output, ids=ids, workspace=workspace)
    # Capture different addresses from native prepare, while keeping the exact
    # prepared geometry. The C ABI must bind these addresses, not stale ones.
    source2, ids2, output2 = source.clone(), ids.clone(), torch.empty_like(output)
    error2 = torch.zeros_like(prepared.device_error)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        error2.zero_()
        prepared(source2, ids=ids2, output=output2, device_error=error2)
    for route in ([2, 2], [-1, 0], [3, 1], [0, 1]):
        ids2.copy_(torch.tensor([route] * m, dtype=torch.int32, device=source.device))
        graph.replay()
        torch.cuda.synchronize()
        assert error2.item() == int(3 in route)
        x = source2.cpu().double()
        for slot, expert in enumerate(route):
            actual = output2[:, slot].cpu().double()
            if expert < 0 or expert >= 3:
                assert torch.count_nonzero(actual) == 0
                continue
            expected = x @ reference[expert].T
            magnitude = x.abs() @ reference[expert].abs().T
            bound = expected.abs() * torch.finfo(source.dtype).eps + magnitude * (2e-6 + k * 2**-24)
            assert bool(((actual - expected).abs() <= bound + 1e-7).all())
    # Graph, owners and every captured tensor remain live until the drain above.
