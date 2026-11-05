# SPDX-License-Identifier: Apache-2.0

"""One-time GPU packing through the selected QUactlize library's producer ABI."""

import ctypes as C

from ignition.components.gguf_weights import GgufPlanes, GgufWeight
from . import provenance
from .tensor_metadata import record_numeric_tensor


_undrained = []


def arrangement_descriptor(weight):
    if weight.packed is None:
        return None
    from quactlize.packing.native import Arrangement

    values = weight.packed.arrangement
    if len(values) != len(Arrangement._fields_) or any(type(value) is not int for value in values):
        raise ValueError('GGUF packed weight requires its complete producer arrangement')
    descriptor = Arrangement(*values)
    if tuple(getattr(descriptor, name) for name, _ in Arrangement._fields_) != values:
        raise ValueError('GGUF producer arrangement exceeds its native field widths')
    return descriptor


class GgufPacker:
    """Prepare exact compressed planes; retain raw bytes only until the pack drains."""

    def __init__(self, library):
        from quactlize.packing.native import Arrangement, Sizes

        self.library, self.stream, self.pending = library.lib, None, None
        self.canonical = library.lib.quactlize_ppu_kpack_canonical_arrangement_v1
        self.canonical.argtypes = [C.c_int, C.POINTER(Arrangement)]
        self.canonical.restype = C.c_int
        self.sizes = library.lib.quactlize_ppu_kpack_sizes_for_arrangement_v1
        self.sizes.argtypes = [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)]
        self.sizes.restype = C.c_int
        self.repack = library.lib.quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2
        self.repack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(Arrangement), C.c_void_p]
        self.repack.restype = C.c_int

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f'QUactlize GGUF {operation} failed: {status}')

    def describe(self, weight):
        from quactlize.packing.native import Arrangement, Sizes

        if weight.packed is not None or len(weight.shape) not in (2, 3):
            raise ValueError('GGUF packing requires original dense/expert blocks')
        geometry = (weight.out_features, weight.in_features, weight.experts, weight.qtype)
        if any(type(value) is not int or not 0 < value < 1 << 31 for value in geometry):
            raise ValueError('GGUF packing geometry exceeds the producer int32 ABI')
        descriptor, sizes = Arrangement(), Sizes()
        self._check(self.canonical(weight.qtype, C.byref(descriptor)), 'canonical descriptor query')
        self._check(self.sizes(*geometry, C.byref(descriptor), C.byref(sizes)), 'plane size query')
        if (sizes.raw_bytes != weight.raw.nbytes or not sizes.low_bytes or not sizes.units_bytes
                or sizes.low_bytes + sizes.high_bytes + sizes.units_bytes != sizes.raw_bytes):
            raise ValueError('GGUF packing must preserve the exact compressed byte budget')
        return descriptor, sizes

    def __call__(self, weight):
        import torch

        if self.pending is not None:
            raise RuntimeError('previous GGUF packing work did not drain')
        descriptor, sizes = self.describe(weight)
        raw = weight.raw
        if raw.device.type != 'cuda' or raw.dtype != torch.uint8 or not raw.is_contiguous():
            raise ValueError('GGUF packing requires contiguous device bytes')
        with torch.cuda.device(raw.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GGUF weights must be packed before graph capture')
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=raw.device)
            elif self.stream.device != raw.device:
                raise ValueError('GGUF packer belongs to one device')
            planes = tuple(torch.empty(size, dtype=torch.uint8, device=raw.device) if size else None
                           for size in (sizes.low_bytes, sizes.high_bytes, sizes.units_bytes))
            self.pending = (raw, planes, descriptor)
            try:
                self.stream.wait_stream(torch.cuda.current_stream(raw.device))
                self._check(self.repack(raw.data_ptr(), *(None if plane is None else plane.data_ptr() for plane in planes),
                    weight.out_features, weight.in_features, weight.experts, weight.qtype,
                    C.byref(descriptor), self.stream.cuda_stream), 'GPU repack')
            finally:
                try:
                    self.stream.synchronize()
                except BaseException:
                    _undrained.append(self)
                    raise
                else:
                    self.pending = None
        identity = tuple(getattr(descriptor, name) for name, _ in descriptor._fields_)
        origin = provenance.lookup(raw)
        for role, plane in zip(('low', 'high', 'units'), planes, strict=True):
            if plane is None:
                continue
            record_numeric_tensor(plane)
            if origin is not None:
                provenance.record_sources(plane, origin['sources'], packing='gguf_canonical_kpack',
                    packing_parameters=dict(encoding=weight.encoding, shape=weight.shape,
                                            arrangement=identity, plane=role))
        return GgufWeight(None, weight.encoding, weight.qtype, weight.shape, GgufPlanes(*planes, identity))
