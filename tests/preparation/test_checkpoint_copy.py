# SPDX-License-Identifier: Apache-2.0

"""Driver copy records against SDK headers and host bytes, not device execution."""

import ctypes as C
import os
from pathlib import Path
import subprocess

import pytest

from ignition.preparation.checkpoint_copy import CheckpointCopy, _Memcpy2D
from ignition.preparation import runtime_package


@pytest.fixture(scope='module', params=['cuda', 'ppu'])
def native(request, tmp_path_factory):
    if request.param == 'ppu':
        sdk = Path(os.environ.get('FLARE_TEST_PPU_SDK', '/usr/local/PPU_SDK'))
        include = sdk / 'CUDA_SDK/targets/x86_64-linux/include'
    else:
        include = Path('/usr/local/cuda/include')
    if not (include / 'cuda.h').exists():
        pytest.skip(f'{request.param} SDK headers unavailable: {include}')
    path = tmp_path_factory.mktemp('checkpoint-copy-' + request.param) / 'driver.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', '-Wall', '-Werror', '-I', str(include),
        str(Path(__file__).parent / 'fixtures/checkpoint_copy.c'), '-o', str(path)], check=True, capture_output=True)
    library = C.CDLL(str(path))
    library.set_context.argtypes = [C.c_void_p]
    library.set_copy_error.argtypes = [C.c_int]
    library.copied_stream.restype = C.c_uint64
    library.descriptor_size.restype = library.descriptor_offset.restype = C.c_size_t
    library.descriptor_offset.argtypes = [C.c_uint]
    return library


@pytest.fixture
def driver(native, monkeypatch):
    native.reset_copy()
    def open_driver(name):
        assert name == 'libcuda.so.1'
        return native
    monkeypatch.setattr(runtime_package.C, 'CDLL', open_driver)
    return native


def test_descriptor_matches_every_platform_header_field(native):
    assert C.sizeof(_Memcpy2D) == native.descriptor_size()
    for index, (name, _) in enumerate(_Memcpy2D._fields_):
        assert getattr(_Memcpy2D, name).offset == native.descriptor_offset(index), name


@pytest.mark.parametrize('stream', [0, 0x123456789abc])
@pytest.mark.parametrize('rows,width,source_pitch,destination_pitch', [(1, 41, 41, 41), (7, 13, 17, 31)])
def test_driver_copies_exact_rows_and_retains_strided_holes(driver, stream, rows, width, source_pitch, destination_pitch):
    source = (C.c_ubyte * (rows * source_pitch))(*range(rows * source_pitch))
    destination = (C.c_ubyte * (rows * destination_pitch))(*([231] * (rows * destination_pitch)))
    copy = CheckpointCopy()
    copy(C.addressof(destination), destination_pitch, C.addressof(source), source_pitch, width, rows, stream)
    expected = bytearray([231] * len(destination))
    for row in range(rows):
        expected[row * destination_pitch:row * destination_pitch + width] = bytes(source)[row * source_pitch:row * source_pitch + width]
    assert bytes(destination) == expected
    assert driver.copied_stream() == stream and driver.copy_calls() == 1


def test_absent_torch_context_fails_without_initializing_another(driver):
    driver.set_context(None)
    with pytest.raises(RuntimeError, match='Torch.*current CUDA context'):
        CheckpointCopy()
    assert driver.copy_calls() == 0


def test_driver_error_retains_native_code_and_does_not_retry(driver):
    driver.set_copy_error(201)
    source, destination = (C.c_ubyte * 8)(*range(8)), (C.c_ubyte * 8)(*([231] * 8))
    with pytest.raises(RuntimeError, match='checkpoint cuMemcpy2DAsync_v2: CUDA 201:.*fixture copy failure'):
        CheckpointCopy()(C.addressof(destination), 8, C.addressof(source), 8, 8, 1, 0)
    assert driver.copy_calls() == 1 and bytes(destination) == bytes([231] * 8)
