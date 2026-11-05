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

__global__ void qwen_export_state_probe(unsigned long long *output,
                                        const unsigned long long *control,
                                        unsigned int tag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long indirect = *qwen_export_pointer;
        output[0] = indirect;
        output[1] = qwen_export_counter;
        output[2] = qwen_export_zero;
        output[3] = qwen_export_constants[0];
        output[4] = qwen_export_constants[1];
        output[5] = tag;
        if (control[0]) {
            qwen_export_counter = control[1];
            qwen_export_zero = control[2];
        } else {
            qwen_export_counter += indirect + qwen_export_constants[0];
            qwen_export_zero += 1;
        }
    }
}
}
