/* SPDX-License-Identifier: Apache-2.0 */
/* Host ABI fixture. It exercises recorder subscription, not device delivery. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned options, counts[5];
static const char *names[743] = {
    [18] = "cuModuleLoad", [19] = "cuModuleLoadData", [20] = "cuModuleLoadDataEx",
    [21] = "cuModuleLoadFatBinary", [22] = "cuModuleUnload", [23] = "cuModuleGetFunction",
    [678] = "cuLibraryLoadData", [679] = "cuLibraryLoadFromFile", [680] = "cuLibraryUnload",
    [681] = "cuLibraryGetKernel", [682] = "cuLibraryGetModule", [683] = "cuKernelGetFunction",
    [687] = "cuKernelSetAttribute", [742] = "cuFuncLoad", [85] = "cuFuncGetAttribute",
    [686] = "cuKernelGetAttribute", [374] = "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    [384] = "cuOccupancyMaxPotentialBlockSize",
    [451] = "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    [452] = "cuOccupancyMaxPotentialBlockSizeWithFlags",
    [543] = "cuOccupancyAvailableDynamicSMemPerBlock", [675] = "cuOccupancyMaxPotentialClusterSize",
    [676] = "cuOccupancyMaxActiveClusters", [307] = "cuLaunchKernel", [442] = "cuLaunchKernel_ptsz",
    [477] = "cuLaunchCooperativeKernel", [478] = "cuLaunchCooperativeKernel_ptsz",
    [481] = "cuFuncSetAttribute", [652] = "cuLaunchKernelEx", [653] = "cuLaunchKernelEx_ptsz",
    [243] = "cuMemAlloc_v2", [245] = "cuMemFree_v2", [38] = "cuMemFreeHost",
    [39] = "cuMemHostAlloc", [271] = "cuMemHostAlloc_v2", [294] = "cuMemAllocHost_v2",
    [301] = "cuMemHostRegister", [302] = "cuMemHostUnregister", [379] = "cuMemHostRegister_v2",
    [551] = "cuMemMap", [552] = "cuMemUnmap", [598] = "cuMemAllocAsync",
    [599] = "cuMemAllocAsync_ptsz", [600] = "cuMemFreeAsync", [601] = "cuMemFreeAsync_ptsz",
    [611] = "cuMemAllocFromPoolAsync", [612] = "cuMemAllocFromPoolAsync_ptsz",
};

void configure(unsigned value) { options = value; memset(counts, 0, sizeof(counts)); }
unsigned observed(unsigned index) { return counts[index]; }

int cuptiGetCallbackName(int domain, uint32_t id, const char **name) {
    static char translated[128];
    counts[0]++;
    *name = NULL;
    if (domain != 1 || id >= 743 || !names[id]) return 1;
    if (((options & 2) && id == 742) || ((options & 4) && id == 18)) return 1;
    if ((options & 16) && id == 742) return 0;
    const char *value = ((options & 8) && id == 742) ? "cuFuncIsLoaded" : names[id];
    if (options & 1) {
        snprintf(translated, sizeof(translated), "hg%s", value + 2);
        value = translated;
    }
    *name = value;
    return 0;
}

int cuptiSubscribe(void **subscriber, void (*callback)(void *, int, uint32_t, const void *), void *user) {
    if (!callback || !user) return 1;
    counts[1]++;
    *subscriber = (void *)(uintptr_t)0x1234;
    return 0;
}
int cuptiEnableCallback(uint32_t enable, void *subscriber, int domain, uint32_t id) {
    if (subscriber != (void *)(uintptr_t)0x1234 || enable != 1) return 1;
    counts[2]++;
    if ((options & 32) && domain == 1 && id == 742) return 27;
    if (domain == 1) {
        if (id >= 743 || !names[id]) return 1;
        if (id == 21) counts[4]++;
    } else if (domain != 3 || (id != 2 && id != 6 && id != 7)) return 1;
    return 0;
}
int cuptiUnsubscribe(void *subscriber) {
    if (subscriber != (void *)(uintptr_t)0x1234) return 1;
    counts[3]++;
    return 0;
}
int cuptiGetResultString(int status, const char **message) {
    *message = status == 27 ? "not supported" : "invalid parameter";
    return 0;
}
