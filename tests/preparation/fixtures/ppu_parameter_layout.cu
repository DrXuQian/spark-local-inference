// SPDX-License-Identifier: Apache-2.0
// Offline parameter ABI qualification, never used for serving.
#include <hggc_runtime.h>

using U64 = unsigned long long;
struct Three { char bytes[3]; };
struct Odd { char first; long long middle; char last; };
struct alignas(16) Aligned { long long data[2]; };
static_assert(sizeof(Three) == 3 && sizeof(Odd) == 24 && sizeof(Aligned) == 16);

extern "C" {
__device__ volatile U64 result;
__global__ void empty() { result = 3; }
__global__ void scalars(char a, short b, int c, long long d, float e,
                        double f, bool g, unsigned char h, unsigned short i) {
    result = a + b + c + d + e + f + g + h + i;
}
__global__ void structs(Three a, Odd b, Aligned c, int d) {
    result = a.bytes[2] + b.last + c.data[1] + d;
}
__global__ void unused(U64 *a, int b, Three c, double d) { result = 9; }
__global__ void words(unsigned int a, float b, double c, long long d) {
    result = a + b + c + d;
}

__global__ void spilled(
    U64 p0, U64 p1, U64 p2, U64 p3,
    U64 p4, U64 p5, U64 p6, U64 p7,
    U64 p8, U64 p9, U64 p10, U64 p11,
    U64 p12, U64 p13, U64 p14, U64 p15,
    U64 p16, U64 p17, U64 p18, U64 p19,
    U64 p20, U64 p21, U64 p22, U64 p23) {
    result = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 +
             p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15 +
             p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23;
}

__global__ void pointers(
    const unsigned char *p0, const unsigned char *p1, const unsigned char *p2, const unsigned char *p3,
    const unsigned char *p4, const unsigned char *p5, const unsigned char *p6, const unsigned char *p7,
    const unsigned char *p8, const unsigned char *p9, const unsigned char *p10, const unsigned char *p11,
    const unsigned char *p12, const unsigned char *p13, const unsigned char *p14, const unsigned char *p15,
    const unsigned char *p16, const unsigned char *p17, const unsigned char *p18, const unsigned char *p19,
    const unsigned char *p20, const unsigned char *p21, const unsigned char *p22, const unsigned char *p23) {
    result = *p0 + *p1 + *p2 + *p3 + *p4 + *p5 + *p6 + *p7 +
             *p8 + *p9 + *p10 + *p11 + *p12 + *p13 + *p14 + *p15 +
             *p16 + *p17 + *p18 + *p19 + *p20 + *p21 + *p22 + *p23;
}
}
