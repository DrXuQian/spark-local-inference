# SPDX-License-Identifier: Apache-2.0

"""Host C-ABI checks for the standalone PPU export fixture; no device execution."""

import ctypes as C
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ignition.preparation.runtime_package import _Cuda
from qualify.ppu_export import launch, parser, write_global


@pytest.fixture
def driver(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('host C compiler required for Driver API argument checks')
    source = tmp_path / 'driver.c'
    source.write_text('''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static uint64_t storage[2];
static uint64_t observed[13];
int cuModuleGetGlobal_v2(uint64_t *address, size_t *size, void *module, const char *name) {
    if ((uintptr_t)module != 123) return 1;
    *address = (uintptr_t)storage;
    *size = strcmp(name, "scalar") == 0 ? 8 : 16;
    return 0;
}
int cuMemcpyHtoD_v2(uint64_t address, const void *source, size_t size) {
    if (address != (uintptr_t)storage || size > sizeof(storage)) return 1;
    memcpy(storage, source, size);
    return 0;
}
int cuLaunchKernel(void *function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void *stream,
    void **parameters, void **extra) {
    observed[0] = (uintptr_t)function;
    observed[1] = gx; observed[2] = gy; observed[3] = gz;
    observed[4] = bx; observed[5] = by; observed[6] = bz;
    observed[7] = shared; observed[8] = (uintptr_t)stream;
    observed[9] = *(uint64_t *)parameters[0];
    observed[10] = *(uint64_t *)parameters[1];
    observed[11] = *(uint32_t *)parameters[2];
    observed[12] = (uintptr_t)extra;
    return 0;
}
uint64_t get_storage(unsigned index) { return storage[index]; }
uint64_t get_observed(unsigned index) { return observed[index]; }
''')
    library = tmp_path / 'driver.so'
    subprocess.run([compiler, '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                    str(source), '-o', str(library)], check=True, capture_output=True)
    cuda = _Cuda.__new__(_Cuda)
    cuda.lib = C.CDLL(str(library))
    for name in ('get_storage', 'get_observed'):
        function = getattr(cuda.lib, name)
        function.argtypes, function.restype = [C.c_uint], C.c_uint64
    return cuda


def test_global_writes_preserve_scalar_pointer_and_constant_array_bytes(driver):
    module = C.c_void_p(123)
    write_global(driver, module, 'scalar', C.c_uint64(0x123456789000))
    assert driver.lib.get_storage(0) == 0x123456789000
    write_global(driver, module, 'array', (C.c_uint64 * 2)(31, 47))
    assert [driver.lib.get_storage(i) for i in range(2)] == [31, 47]
    with pytest.raises(ValueError, match='global extent 16 != 8'):
        write_global(driver, module, 'array', C.c_uint64(99))
    assert [driver.lib.get_storage(i) for i in range(2)] == [31, 47]


def test_probe_launch_passes_pointer_arguments_and_a_32_bit_scalar(driver):
    launch(driver, C.c_void_p(456), SimpleNamespace(data_ptr=lambda: 0x12345678000),
           SimpleNamespace(data_ptr=lambda: 0xabcdef01000), 0xfedcba98,
           SimpleNamespace(cuda_stream=789))
    assert [driver.lib.get_observed(i) for i in range(13)] == [
        456, 1, 1, 1, 32, 1, 1, 0, 789, 0x12345678000, 0xabcdef01000, 0xfedcba98, 0]


def test_offline_module_check_requires_explicit_ppu_target(tmp_path):
    args = parser().parse_args(['--target', 'ppu10', '--output', str(tmp_path), '--compile-only'])
    assert args.compile_only and args.target == 'ppu10'
    with pytest.raises(SystemExit):
        parser().parse_args(['--target', 'cuda_sm120', '--output', str(tmp_path)])
