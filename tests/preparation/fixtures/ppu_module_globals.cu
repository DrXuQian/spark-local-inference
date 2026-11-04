// SPDX-License-Identifier: Apache-2.0
// Offline export qualification; this kernel is not a serving operator.
#include <hggc_runtime.h>

extern "C" {
__device__ unsigned long long qwen_export_counter = 17;
__device__ unsigned long long qwen_export_zero;
__constant__ unsigned long long qwen_export_constants[2] = {23, 29};
__device__ unsigned long long *qwen_export_pointer = &qwen_export_counter;

__global__ void qwen_export_probe(unsigned long long *output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = *qwen_export_pointer + qwen_export_constants[threadIdx.y & 1];
        qwen_export_counter += output[0];
        qwen_export_zero = output[0];
    }
}
}
