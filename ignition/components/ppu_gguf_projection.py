# SPDX-License-Identifier: Apache-2.0

"""Preparation bridge to QUactlize's distinct PPU raw-GGUF matmul ABI.

The caller supplies quactlize.compat.native.Library or the explicitly selected
NVIDIA implementation of those same records. Each library supplies its own
workspace query. Physical PPU recipe/export qualification remains required.
"""

import ctypes as C
import itertools
import weakref

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor
from .gguf_weights import GgufWeight


_owners = weakref.WeakValueDictionary()
_handles = itertools.count(1)


def _problem(source, weight, output, ids):
    from quactlize.compat.native import Problem

    indexed = ids is not None
    if (len(weight.shape) != (3 if indexed else 2)
            or source.ndim not in ((2, 3) if indexed else (2,))
            or source.shape[-1] != weight.in_features or source.shape[0] < 1
            or source.dtype not in (torch.float16, torch.bfloat16)
            or source.device.type != 'cuda' or source.stride(-1) != 1):
        raise ValueError('PPU GGUF projection requires device 16-bit input with contiguous columns')
    m, n, k = source.shape[0], weight.out_features, weight.in_features
    if indexed and (ids.ndim != 2 or ids.shape[0] != m or ids.stride(-1) != 1
                    or ids.dtype != torch.int32 or ids.device != source.device):
        raise ValueError('PPU indexed projection requires int32 IDs [M,topk]')
    topk = ids.shape[1] if indexed else 1
    channels = source.shape[1] if source.ndim == 3 else 1
    expected = (m, topk, n) if indexed else (m, n)
    if (channels not in (1, topk) or tuple(output.shape) != expected
            or output.dtype != source.dtype or output.device != source.device
            or output.stride(-1) != 1
            or (indexed and output.stride(0) != topk * output.stride(1))):
        raise ValueError('PPU projection output requires uniformly strided token/slot rows')
    if (weight.raw.dtype != torch.uint8 or weight.raw.device != source.device
            or not weight.raw.is_contiguous()):
        raise ValueError('PPU projection weights require contiguous original GGUF bytes')
    values = dict(qtype=weight.qtype, m=m, n=n, k=k, experts=weight.experts,
                  topk=topk, channels=channels, device=source.device.index)
    if any(type(v) is not int or not 0 <= v < 1 << 31 for v in values.values()):
        raise ValueError('PPU projection geometry exceeds its int32 ABI')
    return Problem(version=1, size=C.sizeof(Problem), layout=0, mode=int(indexed),
        dtype=int(source.dtype == torch.bfloat16), **values,
        input_row_stride=source.stride(1) if source.ndim == 3 else k,
        input_token_stride=source.stride(0), ids_stride=ids.stride(0) if indexed else 0,
        output_stride=output.stride(-2))


def _extent(tensor):
    # numel() undercounts row-padded views; the ABI addresses the full byte span.
    if not tensor.numel():
        return 0
    if any(s < 0 for s in tensor.stride()):
        raise ValueError('PPU projection requires nonnegative element strides')
    return (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))) * tensor.element_size()


def _buffers(source, raw, ids, output, workspace, error, stream):
    from quactlize.compat.native import Buffers

    buffers = Buffers(version=1, size=C.sizeof(Buffers))
    for name, value in (('input', source), ('raw', raw), ('ids', ids),
                        ('output', output), ('workspace', workspace)):
        if value is not None:
            setattr(buffers, name, value.data_ptr())
            setattr(buffers, name + '_bytes', _extent(value))
    buffers.device_error, buffers.stream = error.data_ptr(), stream
    return buffers


def projection_sizes(library, source, weight: GgufWeight, output, *, ids=None):
    """Query the selected implementation's capacity, route and scratch sizes."""
    from quactlize.compat.native import Sizes

    problem = _problem(source, weight, output, ids)
    sizes = Sizes()
    library.check(library.query(C.byref(problem), None, C.byref(sizes)))
    if sizes.version != 1 or sizes.size != C.sizeof(Sizes):
        raise ValueError('PPU matmul query returned an incompatible sizes record')
    if weight.raw.nbytes != sizes.raw_bytes:
        raise ValueError('PPU raw weight extent differs from its GGUF geometry')
    return problem, sizes


@torch.library.custom_op('flare_ppu_gguf::matmul', mutates_args=('output', 'workspace', 'error'))
def matmul(source: torch.Tensor, raw: torch.Tensor, ids: torch.Tensor | None,
           output: torch.Tensor, workspace: torch.Tensor, error: torch.Tensor, handle: int) -> None:
    try:
        owner = _owners[handle]
    except KeyError as exc:
        raise RuntimeError('retain the prepared PPU projection until graph work drains') from exc
    owner.launch(source, raw, ids, output, workspace, error)


@matmul.register_fake
def _fake_matmul(source, raw, ids, output, workspace, error, handle):
    pass


def _signature(value):
    return None if value is None else (tuple(value.shape), value.stride(), value.dtype, value.device)


def _disjoint(operands):
    spans = [(value.data_ptr(), value.data_ptr() + _extent(value)) if value is not None else (0, 0)
             for value in operands]
    for i in range(3, len(spans)):
        begin, end = spans[i]
        if begin == end:
            continue
        if any(begin < other_end and other_begin < end
               for other_begin, other_end in spans[:i] if other_begin != other_end):
            raise ValueError('PPU projection writes must be disjoint from live operands')


class _NativeOwner:
    def __init__(self, library, problem, operands):
        _disjoint(operands)
        self.library, self.signatures = library, tuple(map(_signature, operands))
        self.device = operands[0].device
        self.native = C.c_void_p()
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('prepare PPU projections before capture')
            source, raw, ids, output, workspace, error = operands
            # The native prepare runs its first call. Shared serving IDs/input
            # are not initialized until the complete forward executes.
            warm_source = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype, device=self.device)
            warm_source.zero_()
            warm_ids = None if ids is None else torch.empty_strided(ids.shape, ids.stride(), dtype=ids.dtype, device=self.device)
            if warm_ids is not None:
                warm_ids.zero_()
            warm_error = torch.zeros_like(error)
            buffers = _buffers(warm_source, raw, warm_ids, output, workspace, warm_error,
                               torch.cuda.current_stream(self.device).cuda_stream)
            library.check(library.prepare(C.byref(problem), None, C.byref(buffers), C.byref(self.native)))
            if not self.native.value:
                raise RuntimeError('PPU prepare succeeded without returning a native handle')
            self.finalizer = weakref.finalize(self, library.destroy, self.native)
            if warm_error.item():
                raise RuntimeError('PPU projection warmup reported a device error')
        self.handle = next(_handles)
        _owners[self.handle] = self

    def launch(self, *operands):
        if tuple(map(_signature, operands)) != self.signatures:
            raise ValueError('PPU projection operands changed shape, dtype, device or element strides')
        with torch.cuda.device(self.device):
            buffers = _buffers(*operands, torch.cuda.current_stream(self.device).cuda_stream)
            self.library.check(self.library.run(self.native, C.byref(buffers)))


class PreparedPpuGgufProjection:
    """Retain raw weights, queried scratch and the independently owned PPU handle.

    Construct all capacities before capture, with disjoint output/status and
    scratch for overlapping work. Serial operations may share scratch. Error
    flags are sticky and must join the existing complete-round error guard.
    The native prepare warms and synchronizes before returning; subsequent
    calls only enqueue the external kernels. Python is absent from replay.
    """

    def __init__(self, library, source, weight: GgufWeight, output, *, workspace, ids=None):
        problem, sizes = projection_sizes(library, source, weight, output, ids=ids)
        with torch.cuda.device(source.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('prepare PPU projections before capture')
        if (workspace.device != source.device or workspace.dtype != torch.uint8
                or not workspace.is_contiguous() or workspace.nbytes < sizes.workspace_bytes
                or (sizes.workspace_bytes and workspace.data_ptr() % 256)):
            raise ValueError(f'PPU projection requires {sizes.workspace_bytes} bytes of aligned caller-owned scratch')
        self.raw, self.ids, self.output, self.workspace = weight.raw, ids, output, workspace
        self.device_error = torch.zeros((), dtype=torch.int32, device=source.device)
        self.device_errors = (self.device_error,)
        self.owner = _NativeOwner(library, problem,
            (source, self.raw, ids, output, workspace, self.device_error))
        self.handle = self.owner.handle
        record_numeric_tensor(self.raw)

    def __getstate__(self):
        # Metadata-only compilation workers never load or pickle a native library.
        return {key: value for key, value in vars(self).items() if key != 'owner'}

    def __call__(self, source, *, ids=None, output=None, device_error=None):
        matmul(source, self.raw, self.ids if ids is None else ids,
               self.output if output is None else output, self.workspace,
               self.device_error if device_error is None else device_error, self.handle)
        return self.output if output is None else output
