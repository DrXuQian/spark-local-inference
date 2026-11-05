/* Host-only foreign-function fixture. This is not a GPU BLAS implementation. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifndef API
#define API cublas
#endif
#define JOIN_(a,b) a##b
#define JOIN(a,b) JOIN_(a,b)
#define FN(name) JOIN(API,name)

static int sequence, stream_step, workspace_step, failure;
static uintptr_t last_stream, last_workspace;
static size_t last_bytes;
static int last_compute, last_math, last_types[3];

int FN(Create_v2)(void **handle) { *handle = (void *)(uintptr_t)1234; return 0; }
int FN(Destroy_v2)(void *handle) { return handle != (void *)(uintptr_t)1234; }
int FN(SetStream_v2)(void *handle, void *stream) {
    stream_step = ++sequence; last_stream = (uintptr_t)stream;
    last_workspace = 0; return failure;
}
int FN(SetWorkspace_v2)(void *handle, void *workspace, size_t bytes) {
    workspace_step = ++sequence; last_workspace = (uintptr_t)workspace;
    last_bytes = bytes; return failure;
}
int FN(SetPointerMode_v2)(void *handle, int mode) { return mode == 0 ? 0 : 7; }
int FN(SetMathMode)(void *handle, int mode) { last_math = mode; return 0; }

static float read_value(void const *base, int type, size_t index) {
    if (type == 0) return ((float const *)base)[index];
    uint32_t bits = (uint32_t)((uint16_t const *)base)[index] << 16;
    float result; memcpy(&result, &bits, 4); return result;
}
static void write_value(void *base, int type, size_t index, float value) {
    if (type == 0) { ((float *)base)[index] = value; return; }
    uint32_t bits; memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    ((uint16_t *)base)[index] = (uint16_t)(bits >> 16);
}

int FN(GemmEx)(void *handle, int transa, int transb, int m, int n, int k,
        float const *alpha, void const *a, int atype, int lda,
        void const *b, int btype, int ldb, float const *beta,
        void *c, int ctype, int ldc, int compute, int algorithm) {
    if (failure) return failure;
    if (transa != 1 || transb != 0 || lda < k || ldb < k || ldc < m
            || *alpha != 1 || *beta != 0 || algorithm != -1
            || stream_step + 1 != workspace_step || !last_workspace) return 7;
    last_compute = compute; last_types[0] = atype; last_types[1] = btype; last_types[2] = ctype;
    for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) {
        float result = 0;
        for (int p=0; p<k; ++p)
            result += read_value(a,atype,p+(size_t)i*lda)*read_value(b,btype,p+(size_t)j*ldb);
        write_value(c,ctype,i+(size_t)j*ldc,result);
    }
    return 0;
}
uint64_t inspect(int field) {
    switch(field) {
        case 0: return last_stream;
        case 1: return last_workspace;
        case 2: return last_bytes;
        case 3: return last_compute;
        case 4: return last_math;
        case 5: return last_types[0];
        case 6: return last_types[1];
        case 7: return last_types[2];
    }
    return 0;
}
void fail_with(int status) { failure=status; }
