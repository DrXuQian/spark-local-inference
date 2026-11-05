// SPDX-License-Identifier: Apache-2.0
// Host-only ABI fixture compiled against the actual platform cuda.h.
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uintptr_t current_context = 0x123456789abc;
static uintptr_t last_stream;
static int calls;
static CUresult copy_error;

void reset_copy(void) { current_context = 0x123456789abc; last_stream = 0; calls = 0; copy_error = CUDA_SUCCESS; }
void set_context(uintptr_t context) { current_context = context; }
void set_copy_error(int error) { copy_error = (CUresult)error; }
uintptr_t copied_stream(void) { return last_stream; }
int copy_calls(void) { return calls; }
size_t descriptor_size(void) { return sizeof(CUDA_MEMCPY2D); }
size_t descriptor_offset(unsigned index) {
    const size_t offsets[] = {
        offsetof(CUDA_MEMCPY2D, srcXInBytes), offsetof(CUDA_MEMCPY2D, srcY),
        offsetof(CUDA_MEMCPY2D, srcMemoryType), offsetof(CUDA_MEMCPY2D, srcHost),
        offsetof(CUDA_MEMCPY2D, srcDevice), offsetof(CUDA_MEMCPY2D, srcArray), offsetof(CUDA_MEMCPY2D, srcPitch),
        offsetof(CUDA_MEMCPY2D, dstXInBytes), offsetof(CUDA_MEMCPY2D, dstY),
        offsetof(CUDA_MEMCPY2D, dstMemoryType), offsetof(CUDA_MEMCPY2D, dstHost),
        offsetof(CUDA_MEMCPY2D, dstDevice), offsetof(CUDA_MEMCPY2D, dstArray), offsetof(CUDA_MEMCPY2D, dstPitch),
        offsetof(CUDA_MEMCPY2D, WidthInBytes), offsetof(CUDA_MEMCPY2D, Height)
    };
    return offsets[index];
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context) {
    *context = (CUcontext)current_context;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **description) {
    (void)error;
    *description = "fixture copy failure";
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *copy, CUstream stream) {
    ++calls;
    last_stream = (uintptr_t)stream;
    if (copy_error) return copy_error;
    if (copy->srcMemoryType != CU_MEMORYTYPE_HOST || copy->dstMemoryType != CU_MEMORYTYPE_DEVICE ||
        !copy->srcHost || !copy->dstDevice || copy->srcDevice || copy->dstHost ||
        copy->srcArray || copy->dstArray || copy->srcXInBytes || copy->dstXInBytes || copy->srcY || copy->dstY ||
        copy->srcPitch < copy->WidthInBytes || copy->dstPitch < copy->WidthInBytes)
        return CUDA_ERROR_INVALID_VALUE;
    for (size_t row = 0; row < copy->Height; ++row)
        memcpy((void *)(uintptr_t)(copy->dstDevice + row * copy->dstPitch),
               (const char *)copy->srcHost + row * copy->srcPitch, copy->WidthInBytes);
    return CUDA_SUCCESS;
}
