# SPDX-License-Identifier: Apache-2.0

"""Exercise the PPU record bridge using QUactlize's explicit NVIDIA implementation."""

import os
import json

import pytest

pytest.importorskip('quactlize.compat.native')
from tests.components.test_ppu_gguf_projection_cuda import check_captured_projection

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1'
                              or not os.environ.get('FLARE_TEST_CUDA_MATMUL_LIBRARY'),
                                reason='explicit NVIDIA CUDA assignment required')


@pytest.mark.parametrize('qtype', [7, 8])
@pytest.mark.parametrize('m', [1, 9])
def test_nvidia_implementation_of_ppu_matmul_records(tmp_path, qtype, m):
    from ignition.preparation.qwen38_gguf import GgufInputs

    # The checkpoint placeholders are only input paths; actual weights below
    # come from the independent numerical fixture, not a fake checkpoint load.
    for name in ('main.gguf', 'mtp.gguf'):
        (tmp_path / name).touch()
    (tmp_path / 'frontend').mkdir()
    library_path = os.environ['FLARE_TEST_CUDA_MATMUL_LIBRARY']
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(schema='qwen38-gguf/1',
        main='main.gguf', mtp='mtp.gguf', frontend='frontend', read_mode='pread',
        quactlize_abi_library=library_path)))
    inputs = GgufInputs.read(tmp_path)
    libraries = inputs.load_operators('cuda:0')
    assert libraries['actlize_library'] is libraries['actlize_qsa_library'] is None
    check_captured_projection(libraries['gguf_projection_library'], qtype, m)
