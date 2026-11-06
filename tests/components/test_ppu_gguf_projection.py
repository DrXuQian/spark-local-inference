# SPDX-License-Identifier: Apache-2.0

"""Check the PPU bridge against QUactlize's actual host contract, without a GPU.

Set FLARE_TEST_QUACTLIZE_PPU_SOURCE to the feat/ppu-flash-next checkout and
FLARE_TEST_PPU_SDK to its SDK headers, and install that checkout's Python
declarations. These checks do not execute or emulate a PPU kernel.
"""

import ctypes as C
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

native = pytest.importorskip('quactlize.compat.native')

from ignition.components.gguf_weights import GgufWeight
from ignition.components.ppu_gguf_projection import _buffers, _disjoint, _extent, matmul, projection_sizes


@pytest.fixture(scope='module')
def contract(tmp_path_factory):
    configured = os.environ.get('FLARE_TEST_QUACTLIZE_PPU_SOURCE')
    if not configured:
        pytest.skip('set FLARE_TEST_QUACTLIZE_PPU_SOURCE for the real host query/validation contract')
    source = Path(configured).resolve(strict=True)
    sdk = Path(os.environ.get('FLARE_TEST_PPU_SDK', '/usr/local/PPU_SDK'))
    if not (sdk / 'include/hggc_runtime.h').is_file():
        pytest.skip('host contract compilation requires the real PPU SDK headers')
    directory = tmp_path_factory.mktemp('ppu-matmul-contract')
    cpp, library = directory / 'contract.cpp', directory / 'contract.so'
    cpp.write_text('''
#include "quactlize/compat/contract.hpp"
#include <string>
static thread_local std::string error;
extern "C" int query(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a, qpm_sizes_v1 *s) {
    try { *s = quactlize::compat::layout(*p, a).sizes; return 0; }
    catch (std::exception const &e) { error = e.what(); return 20; }
}
extern "C" int query_packed(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a, qpm_sizes_v1 *s) {
    try { *s = quactlize::compat::layout(*p, a, true, QPM_PACKED_AUTO).sizes; return 0; }
    catch (std::exception const &e) { error = e.what(); return 20; }
}
// A host query fixture must never execute the device producer.
extern "C" int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
    uint8_t const *, uint8_t *, uint8_t *, uint8_t *, int, int, int, int,
    quactlize_ppu_placed_arrangement_v2 const *, void *) { return QPM_RUNTIME; }
extern "C" int validate(qpm_problem_v1 const *p, qpm_buffers_v1 const *b) {
    try { auto l = quactlize::compat::layout(*p, nullptr);
          quactlize::compat::validate(*p, l, *b); return 0; }
    catch (std::exception const &e) { error = e.what(); return 20; }
}
extern "C" char const *last_error() { return error.c_str(); }
''')
    result = subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared',
        '-I' + str(source), '-I' + str(source / 'quactlize/include'),
        '-I' + str(source / 'third_party/actlize/include'), '-I' + str(sdk / 'include'), str(cpp),
        str(source / 'quactlize/packing/sizes.cpp'), '-o', str(library)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lib = C.CDLL(str(library))
    lib.query.argtypes = [C.POINTER(native.Problem), C.c_void_p, C.POINTER(native.Sizes)]
    lib.query.restype = C.c_int
    lib.query_packed.argtypes, lib.query_packed.restype = lib.query.argtypes, C.c_int
    lib.validate.argtypes = [C.POINTER(native.Problem), C.POINTER(native.Buffers)]
    lib.validate.restype = C.c_int
    lib.last_error.restype = C.c_char_p

    class HostContract:
        # Only the real header's query and validation exist. No fake prepare/run.
        query, validate = lib.query, lib.validate

        @staticmethod
        def check(rc):
            if rc:
                raise ValueError(lib.last_error().decode())

    from types import SimpleNamespace

    result = HostContract()
    result.lib = lib
    result.packed = SimpleNamespace(query=lib.query_packed, check=result.check)
    return result


def operands(m, *, n=640, k=640, qtype=7, indexed=True, channels=1, padding=0):
    e, topk = (512, 10) if indexed else (1, 1)
    source_shape = (m, k) if channels == 1 else (m, topk, k)
    source_strides = (k + padding, 1) if channels == 1 else (topk * (k + padding), k + padding, 1)
    source = torch.empty_strided(source_shape, source_strides, dtype=torch.bfloat16, device='cuda:0')
    ids = torch.empty_strided((m, topk), (topk + padding, 1), dtype=torch.int32, device='cuda:0') if indexed else None
    output_shape = (m, topk, n) if indexed else (m, n)
    output_strides = (topk * (n + padding), n + padding, 1) if indexed else (n + padding, 1)
    output = torch.empty_strided(output_shape, output_strides, dtype=source.dtype, device='cuda:0')
    block, size = {7: (32, 24), 8: (32, 34), 12: (256, 144), 42: (64, 18)}[qtype]
    raw = torch.empty(e * n * k // block * size, dtype=torch.uint8, device='cuda:0')
    weight = GgufWeight(raw, {7: 'Q5_1', 8: 'Q8_0', 12: 'Q4_K', 42: 'Q2_0'}[qtype], qtype,
                        (e, n, k) if indexed else (n, k))
    return source, weight, output, ids


@pytest.mark.parametrize('m', [1, 8, 9, 64])
@pytest.mark.parametrize('channels,padding', [(1, 0), (1, 7), (10, 7)])
@pytest.mark.parametrize('qtype,n,k', [(7, 2560, 640), (12, 1280, 2560),
                                    (42, 1280, 2560), (42, 2560, 640)])
def test_actual_expert_geometry_routes_and_strides(contract, m, channels, padding, qtype, n, k):
    with FakeTensorMode():
        source, weight, output, ids = operands(m, n=n, k=k, qtype=qtype, channels=channels, padding=padding)
        problem, sizes = projection_sizes(contract, source, weight, output, ids=ids)
        assert problem.input_token_stride == source.stride(0)
        assert problem.input_row_stride == (source.stride(1) if channels > 1 else k)
        assert problem.ids_stride == ids.stride(0) and problem.output_stride == output.stride(1)
        assert sizes.input_bytes == _extent(source)
        assert sizes.output_bytes == _extent(output) and sizes.ids_bytes == _extent(ids)
        assert sizes.route == int(m > 8)
        assert sizes.raw_bytes == weight.raw.nbytes
        assert sizes.workspace_bytes == 0 if m <= 8 else sizes.workspace_bytes > 0
        assert sizes.expanded_weight_bytes <= 2 * 4 * 512 * k


@pytest.mark.parametrize('n,k', [(320, 10240), (10240, 320), (4, 10240), (2560, 640)])
def test_raw_dense_special_shapes_do_not_request_canonical_layout(contract, n, k):
    with FakeTensorMode():
        source, weight, output, ids = operands(9, n=n, k=k, qtype=8, indexed=False, padding=3)
        problem, sizes = projection_sizes(contract, source, weight, output)
        assert problem.layout == 0 and problem.mode == 0 and problem.ids_stride == 0
        assert sizes.low_bytes == sizes.high_bytes == sizes.units_bytes == sizes.ids_bytes == 0


def test_fieldwise_buffers_use_spans_and_native_alias_checks(contract):
    # Host memory is used only by the host contract's address/extent validator.
    # There is no CPU matmul implementation or substitution here.
    with FakeTensorMode():
        source, weight, output, ids = operands(9, n=37, indexed=False, padding=7)
        problem, sizes = projection_sizes(contract, source, weight, output)
        source_geometry, output_geometry = (source.shape, source.stride()), (output.shape, output.stride())
    source = torch.empty_strided(*source_geometry, dtype=torch.bfloat16)
    output = torch.empty_strided(*output_geometry, dtype=torch.bfloat16)
    raw = torch.empty(sizes.raw_bytes, dtype=torch.uint8)
    storage = torch.empty(sizes.workspace_bytes + 255, dtype=torch.uint8)
    start = (-storage.data_ptr()) % 256
    workspace = storage[start:start + sizes.workspace_bytes]
    error = torch.zeros((), dtype=torch.int32)
    b = _buffers(source, raw, None, output, workspace, error, 0x12345678)
    assert b.input == source.data_ptr() and b.input_bytes > source.nbytes
    assert b.raw_bytes == raw.nbytes and b.output_bytes > output.nbytes
    assert b.ids is None and b.ids_bytes == 0 and b.stream == 0x12345678
    assert contract.validate(C.byref(problem), C.byref(b)) == 0
    b.input_bytes = source.nbytes
    assert contract.validate(C.byref(problem), C.byref(b)) != 0
    b.input_bytes = _extent(source)
    b.output = b.input
    assert contract.validate(C.byref(problem), C.byref(b)) != 0
    _disjoint((source, raw, None, output, workspace, error))
    with pytest.raises(ValueError, match='disjoint'):
        _disjoint((source, raw, None, source, workspace, error))
    # An unused zero-byte scratch view may share a storage address.
    _disjoint((source, raw, None, output, raw[:0], error))


def test_invalid_stride_and_weight_extent_are_rejected(contract):
    with FakeTensorMode():
        source, weight, output, ids = operands(9, n=37, indexed=False)
        with pytest.raises(ValueError, match='extent'):
            short = torch.empty(weight.raw.numel() - 1, dtype=weight.raw.dtype, device=weight.raw.device)
            projection_sizes(contract, source, GgufWeight(short, weight.encoding, weight.qtype, weight.shape), output)
        overlapping = torch.empty_strided(source.shape, (1, 1), device=source.device, dtype=source.dtype)
        with pytest.raises(ValueError, match='overlapping input tokens'):
            projection_sizes(contract, overlapping, weight, output)


def test_fake_export_preserves_all_three_mutations_and_readonly_inputs():
    class Forward(torch.nn.Module):
        def forward(self, source, raw, ids, output, workspace, error):
            matmul(source, raw, ids, output, workspace, error, 987654321)
            return output

    args = (torch.empty(2, 640, dtype=torch.bfloat16), torch.empty(24, dtype=torch.uint8),
            torch.empty(2, 10, dtype=torch.int32), torch.empty(2, 10, 37, dtype=torch.bfloat16),
            torch.empty(256, dtype=torch.uint8), torch.empty((), dtype=torch.int32))
    exported = torch.export.export(Forward(), args).run_decompositions()
    mutated = set(exported.graph_signature.user_inputs_to_mutate.values())
    assert mutated == {'output', 'workspace', 'error'}
    # Running this metadata-only graph must fail, never choose a host/NVIDIA path.
    with pytest.raises(RuntimeError, match='retain the prepared PPU projection'):
        matmul(*args, 987654321)
