# SPDX-License-Identifier: Apache-2.0

"""Copy pinned checkpoint rows through the driver owning Torch's CUDA context."""

import ctypes as C

from .runtime_package import _Cuda


class _Memcpy2D(C.Structure):
    # CUDA_MEMCPY2D_v2 from cuda.h; device addresses remain 64-bit integers.
    _fields_ = [
        ('srcXInBytes', C.c_size_t), ('srcY', C.c_size_t), ('srcMemoryType', C.c_int),
        ('srcHost', C.c_void_p), ('srcDevice', C.c_uint64), ('srcArray', C.c_void_p), ('srcPitch', C.c_size_t),
        ('dstXInBytes', C.c_size_t), ('dstY', C.c_size_t), ('dstMemoryType', C.c_int),
        ('dstHost', C.c_void_p), ('dstDevice', C.c_uint64), ('dstArray', C.c_void_p), ('dstPitch', C.c_size_t),
        ('WidthInBytes', C.c_size_t), ('Height', C.c_size_t),
    ]


class CheckpointCopy:
    """Bind once after Torch initializes the selected device; callers own/drain storage.

    No CUDA Runtime is initialized here. A separate cuda-python Runtime can
    reject the PPU driver even when the platform Torch already owns a working
    context. Use the platform driver directly, as native graph export does.
    """

    def __init__(self):
        self.driver = _Cuda()
        context = C.c_void_p()
        self.driver.call('cuCtxGetCurrent', C.byref(context))
        if not context.value:
            raise RuntimeError('checkpoint copy requires the platform driver to see Torch\'s current CUDA context')
        self.copy = self.driver.lib.cuMemcpy2DAsync_v2
        self.copy.argtypes, self.copy.restype = [C.POINTER(_Memcpy2D), C.c_void_p], C.c_int

    def __call__(self, destination, destination_pitch, source, source_pitch, width, rows, stream):
        descriptor = _Memcpy2D(srcMemoryType=1, srcHost=source, srcPitch=source_pitch,
                              dstMemoryType=2, dstDevice=destination, dstPitch=destination_pitch,
                              WidthInBytes=width, Height=rows)
        self.driver.check('checkpoint cuMemcpy2DAsync_v2', self.copy(C.byref(descriptor), C.c_void_p(stream)))
