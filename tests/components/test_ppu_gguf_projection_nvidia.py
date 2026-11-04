# SPDX-License-Identifier: Apache-2.0

"""Exercise the PPU record bridge using QUactlize's explicit NVIDIA implementation."""

import os

import pytest

pytest.importorskip('quactlize.compat.native')
cuda_compat = pytest.importorskip('quactlize.cuda_compat')
if not hasattr(cuda_compat, 'CudaPpuMatmulLibrary'):
    pytest.skip('explicit NVIDIA implementation of PPU matmul records required', allow_module_level=True)
from tests.components.test_ppu_gguf_projection_cuda import check_captured_projection

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='explicit NVIDIA CUDA assignment required')


@pytest.mark.parametrize('qtype', [7, 8])
@pytest.mark.parametrize('m', [1, 9])
def test_nvidia_implementation_of_ppu_matmul_records(qtype, m):
    check_captured_projection(cuda_compat.CudaPpuMatmulLibrary(), qtype, m)
