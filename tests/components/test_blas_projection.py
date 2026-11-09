# SPDX-License-Identifier: Apache-2.0

"""CPU ABI/layout and export-metadata checks; GPU behavior has a separate test."""

import ctypes as C
from pathlib import Path
import subprocess

import pytest
import torch

from ignition.components.blas_projection import BlasLibrary, ProjectionGeometry, projection


@pytest.fixture(scope='module', params=['cublas', 'acblas'])
def library(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp('blas-abi')
    path = directory / 'librecording.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', f'-DAPI={request.param}',
                    str(Path(__file__).parent / 'fixtures/blas_recording.c'), '-o', str(path)], check=True)
    result = BlasLibrary(path, api=request.param)
    result.library.inspect.argtypes = [C.c_int]
    result.library.inspect.restype = C.c_uint64
    result.library.fail_with.argtypes = [C.c_int]
    return result


@pytest.mark.parametrize('input_dtype,output_dtype', [
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float32)])
def test_c_abi_projects_padded_rows_without_transpose_or_weight_conversion(library, input_dtype, output_dtype):
    x = torch.zeros(3, 14, dtype=input_dtype)[:, 2:9]
    x.copy_(torch.arange(21).view(3, 7).remainder(9).sub(4).float() / 8)
    w = torch.zeros(5, 11, dtype=input_dtype)[:, 1:8]
    w.copy_(torch.arange(35).view(5, 7).remainder(7).sub(3).float() / 8)
    backing = torch.full((3, 12), -37, dtype=output_dtype)
    out = backing[:, 3:8]
    workspace = torch.empty(256, dtype=torch.uint8)
    geometry = ProjectionGeometry.from_tensors(x, w, out)
    handle = C.c_void_p()
    library.check(library.create(C.byref(handle)), 'Create')
    library.check(library.pointer_mode(handle, 0), 'SetPointerMode')
    library.check(library.math_mode(handle, 16), 'SetMathMode')
    try:
        for stream in (17, 23):
            workspace = torch.empty_like(workspace)
            library.enqueue(handle, geometry, x.data_ptr(), w.data_ptr(), out.data_ptr(),
                            workspace.data_ptr(), workspace.nbytes, stream)
            expected = (x.double() @ w.double().T).to(output_dtype)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            assert torch.all(backing[:, :3] == -37) and torch.all(backing[:, 8:] == -37)
            assert [library.library.inspect(i) for i in range(8)] == [
                stream, workspace.data_ptr(), 256, 68, 16,
                14 if input_dtype == torch.bfloat16 else 0,
                14 if input_dtype == torch.bfloat16 else 0,
                14 if output_dtype == torch.bfloat16 else 0]
            x.neg_()
    finally:
        library.check(library.destroy(handle), 'Destroy')


def test_library_failure_is_not_silently_replaced(library):
    x, w, out = torch.zeros(2, 3), torch.zeros(4, 3), torch.empty(2, 4)
    g = ProjectionGeometry.from_tensors(x, w, out)
    library.library.fail_with(15)
    try:
        with pytest.raises(RuntimeError, match='SetStream failed with status 15'):
            library.enqueue(C.c_void_p(1234), g, x.data_ptr(), w.data_ptr(), out.data_ptr(), 256, 256, 0)
    finally:
        library.library.fail_with(0)


def test_missing_or_wrong_library_api_is_rejected(library):
    wrong = 'acblas' if library.api == 'cublas' else 'cublas'
    with pytest.raises(ValueError, match='missing'):
        BlasLibrary(library.path, api=wrong)


def test_geometry_preserves_flattened_hc_rows_and_rejects_copying_layouts():
    x = torch.empty(2, 4, 11)[:, :, :7]
    out = torch.empty(2, 4, 13)[:, :, :5]
    g = ProjectionGeometry.from_tensors(x, torch.empty(5, 7), out)
    assert (g.rows, g.columns, g.inner, g.input_stride, g.output_stride) == (8, 5, 7, 11, 13)
    with pytest.raises(ValueError, match='flatten without a copy'):
        ProjectionGeometry.from_tensors(x.transpose(0, 1), torch.empty(5, 7), out.transpose(0, 1))
    with pytest.raises(ValueError, match='contiguous columns'):
        ProjectionGeometry.from_tensors(torch.empty(3, 14)[:, ::2], torch.empty(5, 7), torch.empty(3, 5))
    with pytest.raises(TypeError, match='BF16 x BF16'):
        ProjectionGeometry.from_tensors(torch.empty(3, 7), torch.empty(5, 7, dtype=torch.bfloat16), torch.empty(3, 5))
    with pytest.raises(ValueError, match='dimensions must be positive'):
        ProjectionGeometry.from_tensors(torch.empty(0, 7), torch.empty(5, 7), torch.empty(0, 5))
    with pytest.raises(ValueError, match='inference tensors'):
        ProjectionGeometry.from_tensors(torch.empty(3, 7, requires_grad=True), torch.empty(5, 7), torch.empty(3, 5))


def test_single_row_geometry_matches_traced_fake_views():
    from torch._subclasses import FakeTensorMode

    # Eager view() recomputes a size-one row stride; fake tensor views keep the
    # padded one. Compiled forwards compare their traced geometry with this.
    source, weight, output = torch.empty(1, 24), torch.empty(5, 24), torch.empty(1, 21)[:, :5]
    prepared = ProjectionGeometry.from_tensors(source, weight, output)
    with FakeTensorMode() as mode:
        traced = ProjectionGeometry.from_tensors(*(mode.from_tensor(t) for t in (source, weight, output)))
    assert traced == prepared
    assert (prepared.rows, prepared.input_stride, prepared.output_stride) == (1, 24, 5)
    padded = torch.empty(2, 21)[:, :5]
    assert ProjectionGeometry.from_tensors(torch.empty(2, 24), weight, padded).output_stride == 21
    with pytest.raises(ValueError, match='contiguous columns'):
        ProjectionGeometry.from_tensors(torch.empty(1, 48)[:, ::2], weight, output)


def test_fake_export_retains_output_and_workspace_mutations_without_native_library():
    class Forward(torch.nn.Module):
        def forward(self, x, w, out, workspace):
            projection(x, w, out, workspace, 999999)
            return out

    arguments = (torch.empty(3, 7), torch.empty(5, 7), torch.empty(3, 5), torch.empty(256, dtype=torch.uint8))
    exported = torch.export.export(Forward(), arguments)
    # No owner for 999999 exists: tracing must use the fake implementation.
    assert any(node.target == torch.ops.flare_blas.projection.default for node in exported.graph.nodes)
    schema = torch.ops.flare_blas.projection.default._schema
    writes = {arg.name for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write}
    assert writes == {'output', 'workspace'}


def test_flare_storage_lowering_keeps_shared_workspace_and_rebindable_handles():
    from ignition.preparation.compilation import ForwardTracer, _lower_storage

    source = torch.empty(3, 7)
    first = torch.empty(5, 7)
    second = torch.empty(2, 5)
    intermediate = torch.empty(3, 9)[:, 2:7]
    output = torch.empty(3, 4)[:, 1:3]
    workspace = torch.empty(256, dtype=torch.uint8)

    def forward():
        projection(source, first, intermediate, workspace, 99100)
        projection(intermediate, second, output, workspace, 99101)
        return output

    module = ForwardTracer()(forward)
    invocations = _lower_storage(module)
    assert [call.target for call in invocations] == [torch.ops.flare_blas.projection.default] * 2
    assert [call.leaves[-1] for call in invocations] == [99100, 99101]
    nodes = [node for node in module.graph.nodes if node.target == torch.ops.flare.storage_call.default]
    assert len(nodes) == 2 and [node.args[2] for node in nodes] == [0, 1]
    # The compiled artifact uses call ordinals. Owners retain process-local BLAS
    # bindings separately; both calls mutate the same explicit scratch storage.
    assert len(nodes[0].args[1]) == len(nodes[1].args[1]) == 2
    assert nodes[0].args[1][1] is nodes[1].args[1][1]
    first_output = next(view for position, mutable, index, view in invocations[0].operands if position == 2)
    second_output = next(view for position, mutable, index, view in invocations[1].operands if position == 2)
    assert first_output[1:3] == ((3, 5), (9, 1))
    assert second_output[1:3] == ((3, 2), (4, 1))
