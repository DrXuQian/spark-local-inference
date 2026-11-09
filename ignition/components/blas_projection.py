# SPDX-License-Identifier: Apache-2.0

"""Dense projections through the explicitly selected cuBLAS-compatible library.

Preparation owns the handle and scratch. The opaque operation records library
kernels into the complete forward graph; native replay never calls Python.
"""

import ctypes as C
from dataclasses import dataclass
import itertools
from math import prod
import weakref

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor


_owners = weakref.WeakValueDictionary()
_handles = itertools.count(1)
_DTYPES = {torch.float32: 0, torch.bfloat16: 14}


@dataclass(frozen=True)
class ProjectionGeometry:
    rows: int
    columns: int
    inner: int
    input_stride: int
    weight_stride: int
    output_stride: int
    input_type: int
    output_type: int

    @classmethod
    def from_tensors(cls, source, weight, output):
        if (source.ndim < 2 or weight.ndim != 2
                or source.shape[-1] != weight.shape[1]
                or tuple(output.shape) != (*source.shape[:-1], weight.shape[0])):
            raise ValueError('BLAS projection requires [...,K] @ [N,K].T -> [...,N]')
        if (source.dtype not in _DTYPES or weight.dtype != source.dtype
                or output.dtype not in _DTYPES
                or source.dtype == torch.float32 and output.dtype != torch.float32):
            raise TypeError('BLAS projection supports BF16 x BF16 -> BF16/FP32 and FP32 x FP32 -> FP32')
        if any(t.device != source.device or t.layout != torch.strided for t in (source, weight, output)):
            raise ValueError('BLAS projection operands must be strided tensors on the same device')
        if any(t.requires_grad for t in (source, weight, output)):
            raise ValueError('prepared BLAS projections require inference tensors')
        rows, inner, columns = prod(source.shape[:-1]), source.shape[-1], weight.shape[0]
        if min(rows, inner, columns) <= 0:
            raise ValueError('BLAS projection dimensions must be positive')
        try:
            matrices = source.view(rows, inner), weight, output.view(rows, columns)
        except RuntimeError as exc:
            raise ValueError('BLAS projection batch dimensions must flatten without a copy') from exc
        strides = []
        for tensor in matrices:
            # The stride of a size-one dimension addresses no element. Eager views
            # and traced fake views may report different values for it, so a
            # single row uses its dense leading dimension.
            stride = tensor.stride(0) if tensor.shape[0] > 1 else tensor.shape[1]
            if tensor.shape[1] > 1 and tensor.stride(1) != 1 or stride < tensor.shape[1]:
                raise ValueError('BLAS projection requires contiguous columns and nonoverlapping rows')
            strides.append(stride)
        if any(n >= 1 << 31 for n in (rows, columns, inner, *strides)):
            raise ValueError('BLAS projection dimensions and leading dimensions must fit int32')
        return cls(rows, columns, inner, *strides, _DTYPES[source.dtype], _DTYPES[output.dtype])


class BlasLibrary:
    """Bind the matching NVIDIA cuBLAS or PPU acBLAS C ABI, without launching."""

    def __init__(self, path, *, api):
        if api not in ('cublas', 'acblas'):
            raise ValueError('BLAS API must be cublas or acblas')
        self.path, self.api = str(path), api
        self.library = C.CDLL(self.path)
        signatures = {
            'create': ('Create_v2', [C.POINTER(C.c_void_p)]),
            'destroy': ('Destroy_v2', [C.c_void_p]),
            'stream': ('SetStream_v2', [C.c_void_p, C.c_void_p]),
            'workspace': ('SetWorkspace_v2', [C.c_void_p, C.c_void_p, C.c_size_t]),
            'pointer_mode': ('SetPointerMode_v2', [C.c_void_p, C.c_int]),
            'math_mode': ('SetMathMode', [C.c_void_p, C.c_int]),
            'gemm': ('GemmEx', [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                              C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                              C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                              C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]),
        }
        for name, (suffix, arguments) in signatures.items():
            symbol = api + suffix
            try:
                function = getattr(self.library, symbol)
            except AttributeError as exc:
                raise ValueError(f'BLAS library is missing {symbol}') from exc
            function.argtypes, function.restype = arguments, C.c_int
            setattr(self, name, function)

    def check(self, status, operation):
        if status:
            raise RuntimeError(f'{self.api} {operation} failed with status {status}')

    def enqueue(self, handle, geometry, source, weight, output, workspace, workspace_bytes, stream):
        # SetStream resets the library's workspace binding: always restore the
        # caller's scratch afterwards, including when warming a new capture stream.
        self.check(self.stream(handle, stream), 'SetStream')
        self.check(self.workspace(handle, workspace, workspace_bytes), 'SetWorkspace')
        alpha, beta = C.c_float(1), C.c_float(0)
        g = geometry
        # C_row = X_row W_row^T => C_col = W_col^T X_col. No transpose copies.
        self.check(self.gemm(handle, 1, 0, g.columns, g.rows, g.inner,
            C.byref(alpha), weight, g.input_type, g.weight_stride,
            source, g.input_type, g.input_stride, C.byref(beta),
            output, g.output_type, g.output_stride,
            68, -1), 'GemmEx')  # COMPUTE_32F; library-selected default algorithm.


def _span(tensor):
    size = (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True)))
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


def _disjoint(source, weight, output, workspace):
    inputs = [_span(source), _span(weight)]
    for tensor in (output, workspace):
        begin, end = _span(tensor)
        if any(begin < stop and start < end for start, stop in inputs):
            raise ValueError('BLAS output and workspace byte spans must be disjoint from live operands')
        inputs.append((begin, end))


def _destroy(library, handle, device):
    with torch.cuda.device(device):
        library.destroy(handle)


class PreparedBlas:
    """One handle and caller-owned scratch for serialized dense projections.

    Each concurrent branch needs its own instance and disjoint workspace. The
    shared workspace is an explicit custom-op mutation, including during whole
    forward compilation. Keep the owner alive until captured work drains.
    """

    def __init__(self, library, workspace):
        if (workspace.device.type != 'cuda' or workspace.dtype != torch.uint8
                or workspace.ndim != 1 or not workspace.is_contiguous()
                or workspace.numel() <= 0 or workspace.data_ptr() % 256):
            raise ValueError('BLAS requires a nonempty, 256-byte-aligned device byte workspace')
        self.library, self.workspace, self.device = library, workspace, workspace.device
        self.native = C.c_void_p()
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('prepare BLAS resources before capture')
            library.check(library.create(C.byref(self.native)), 'Create')
            if not self.native.value:
                raise RuntimeError('BLAS Create succeeded without returning a handle')
            try:
                library.check(library.pointer_mode(self.native, 0), 'SetPointerMode')
                # Do not permit BF16 reduction truncation or TF32 downconversion.
                library.check(library.math_mode(self.native, 16), 'SetMathMode')
            except BaseException:
                library.destroy(self.native)
                raise
        self.finalizer = weakref.finalize(self, _destroy, library, self.native, self.device)
        self.finalizer.atexit = False
        self.handle = next(_handles)
        _owners[self.handle] = self
        self.narrowed = {}

    def operand(self, source, weight):
        """Weight operand for `source`, with one shared BF16 copy per exact FP32 weight.

        GGUF stores small BF16 checkpoint matrices, such as GDN B/A, as exact FP32
        values; GemmEx multiplies operands of one type. Other weights are unchanged.
        """
        if source.dtype != torch.bfloat16 or weight.dtype != torch.float32:
            return weight
        retained = self.narrowed.get(id(weight))
        if retained is None:
            narrowed = weight.bfloat16()
            if not torch.equal(narrowed.float(), weight):
                raise ValueError('FP32 BLAS weight with BF16 input is not exactly representable in BF16')
            # Retaining the source weight keeps its identity from naming another tensor.
            retained = self.narrowed[id(weight)] = (weight, narrowed)
        return retained[1]

    def launch(self, source, weight, output, workspace):
        # Functionalization/rebinding may supply a new storage address. The
        # explicit operand, never a hidden captured pointer, owns this launch.
        if (source.device != self.device or workspace.device != self.device
                or workspace.dtype != torch.uint8 or workspace.shape != self.workspace.shape
                or workspace.stride() != self.workspace.stride() or workspace.data_ptr() % 256):
            raise ValueError('BLAS resource device or workspace geometry changed')
        geometry = ProjectionGeometry.from_tensors(source, weight, output)
        _disjoint(source, weight, output, workspace)
        with torch.cuda.device(self.device):
            self.library.enqueue(self.native, geometry, source.data_ptr(), weight.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), workspace.nbytes, torch.cuda.current_stream(self.device).cuda_stream)


@torch.library.custom_op('flare_blas::projection', mutates_args=('output', 'workspace'))
def projection(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
               workspace: torch.Tensor, handle: int) -> None:
    try:
        owner = _owners[handle]
    except KeyError as exc:
        raise RuntimeError('retain prepared BLAS resources until graph work drains') from exc
    owner.launch(source, weight, output, workspace)


@projection.register_fake
def _fake_projection(source, weight, output, workspace, handle):
    ProjectionGeometry.from_tensors(source, weight, output)


class PreparedBlasProjection:
    """BF16 weights stay BF16 even when the caller requests FP32 logits."""

    def __init__(self, source, weight, output, *, owner, bias=None):
        if bias is not None:
            raise ValueError('the prepared BLAS projection requires unbiased weights')
        self.geometry = ProjectionGeometry.from_tensors(source, weight, output)
        if source.device != owner.device:
            raise ValueError('BLAS projection operands must share their resource device')
        _disjoint(source, weight, output, owner.workspace)
        self.owner, self.weight, self.output = owner, weight, output
        self.workspace, self.handle = owner.workspace, owner.handle
        self.plan = None
        record_numeric_tensor(weight)

    def __getstate__(self):
        return {name: value for name, value in vars(self).items() if name != 'owner'}

    def __call__(self, source):
        geometry = ProjectionGeometry.from_tensors(source, self.weight, self.output)
        if geometry != self.geometry:
            raise ValueError(f'BLAS projection shape, precision or strides changed after preparation: '
                             f'prepared {self.geometry}, called with {geometry}')
        projection(source, self.weight, self.output, self.workspace, self.handle)
        return self.output
