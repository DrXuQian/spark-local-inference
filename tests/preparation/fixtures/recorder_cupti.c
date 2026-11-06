/* SPDX-License-Identifier: Apache-2.0 */
/* Host ABI fixture. It exercises recorder subscription, not device delivery. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned options, counts[5];
static void (*notify)(void *, int, uint32_t, const void *);
static void *notify_user;
static const char *names[743] = {
    [124] = "cuStreamCreate", [367] = "cuStreamCreateWithPriority",
    [127] = "cuStreamDestroy", [326] = "cuStreamDestroy_v2",
    [495] = "cuStreamBeginCapture", [496] = "cuStreamBeginCapture_ptsz",
    [539] = "cuStreamBeginCapture_v2", [540] = "cuStreamBeginCapture_v2_ptsz",
    [497] = "cuStreamEndCapture", [498] = "cuStreamEndCapture_ptsz",
    [720] = "cuStreamBeginCaptureToGraph", [721] = "cuStreamBeginCaptureToGraph_ptsz",
    [517] = "cuGraphDestroy", [571] = "cuGraphKernelNodeSetAttribute",
    [569] = "cuGraphKernelNodeCopyAttributes",
    [576] = "cuStreamSetAttribute", [577] = "cuStreamSetAttribute_ptsz",
    [572] = "cuStreamCopyAttributes", [573] = "cuStreamCopyAttributes_ptsz",
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
    if ((options & 64) && id == 497) return 1;
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
    notify = callback;
    notify_user = user;
    *subscriber = (void *)(uintptr_t)0x1234;
    return 0;
}
int cuptiEnableCallback(uint32_t enable, void *subscriber, int domain, uint32_t id) {
    if (subscriber != (void *)(uintptr_t)0x1234 || enable != 1) return 1;
    counts[2]++;
    if (((options & 32) && domain == 1 && id == 742) ||
        ((options & 128) && domain == 1 && id == 497)) return 27;
    if (domain == 1) {
        if (id >= 743 || !names[id]) return 1;
        if (id == 21) counts[4]++;
    } else if (domain != 3 || (id != 2 && id != 6 && id != 7)) return 1;
    return 0;
}
int cuptiUnsubscribe(void *subscriber) {
    if (subscriber != (void *)(uintptr_t)0x1234) return 1;
    counts[3]++;
    notify = NULL;
    notify_user = NULL;
    return 0;
}
int cuptiGetResultString(int status, const char **message) {
    *message = status == 27 ? "not supported" : "invalid parameter";
    return 0;
}

/* The SDK 2.2.1 module payload has this layout, without Resource around it. */
struct Module { uint32_t id; size_t size; const unsigned char *image; };
struct Resource { uintptr_t context, handle; const struct Module *descriptor; };
struct Call {
    int site;
    const char *name;
    const uintptr_t *params;
    const int *result;
    const char *symbol;
    uintptr_t context;
    uint32_t context_id;
    uint64_t *correlation;
    uint32_t correlation_id;
};

void emit_module(uintptr_t context, int bare) {
    uintptr_t function = 0x500;
    uintptr_t params[] = {(uintptr_t)&function, 0x100, (uintptr_t)"probe"};
    int success = 0;
    struct Call call = {.name = (options & 1) ? "hgModuleGetFunction" : "cuModuleGetFunction",
                        .params = params, .result = &success, .context = context};
    const unsigned char bytes[] = "native PPU image";
    struct Module module = {19, sizeof(bytes), bytes};
    struct Resource resource = {context, 0, &module};
    notify(notify_user, 1, 23, &call);
    notify(notify_user, 3, 6, bare ? (const void *)&module : (const void *)&resource);
    call.site = 1;
    notify(notify_user, 1, 23, &call);
    notify(notify_user, 3, 7, bare ? (const void *)&module : (const void *)&resource);
    notify(notify_user, 3, 2, &resource);
}

/* Also permits a real SDK dispatcher to deliver into this host test subscriber. */
void forward_callback(void *user, int domain, uint32_t id, const void *data) {
    (void)user;
    notify(notify_user, domain, id, data);
}
