# SPDX-License-Identifier: Apache-2.0

"""Prepared native-precision paged attention through the installed FA3 operator."""

import torch

from ignition.preparation.scratch import ScratchArena


@torch.library.custom_op('flare::flash_attn3_paged', mutates_args=('output',))
def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            pages: torch.Tensor, lengths: torch.Tensor, starts: torch.Tensor,
            output: torch.Tensor, width: int, splits: int) -> None:
    # The native library was loaded and checked before compilation. Its private
    # LSE/split buffers live in Torch's captured graph pool, not across forwards.
    torch.ops.flash_attn_3.fwd.default(query, key, value, out=output,
        cu_seqlens_q=starts, seqused_k=lengths, max_seqlen_q=width,
        page_table=pages, is_causal=True, num_splits=splits)


@_attend.register_fake
def _fake_attend(query, key, value, pages, lengths, starts, output, width, splits):
    pass


def _rows(tensor, *, dimensions, dtype, device):
    if (tensor.ndim != dimensions or tensor.dtype != dtype or tensor.device != device
            or tensor.stride(-1) != 1 or any(s <= 0 for s in tensor.shape)):
        raise ValueError('paged operands require matching dtype/device and contiguous nonempty columns')
    for axis in range(dimensions - 1):
        if tensor.stride(axis) < tensor.shape[axis + 1] * tensor.stride(axis + 1):
            raise ValueError('paged operand rows must not overlap')


def _disjoint(reads, writes):
    from torch._subclasses.fake_tensor import FakeTensor

    def span(tensor):
        size = tensor.element_size()
        extent = (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))) * size
        if isinstance(tensor, FakeTensor) or tensor.device.type == 'meta':
            return tensor.untyped_storage()._cdata, tensor.storage_offset() * size, extent
        return None, tensor.data_ptr(), extent

    previous = [span(tensor) for tensor in reads]
    for tensor in writes:
        storage, begin, extent = span(tensor)
        if any(storage == other and begin < start + length and start < begin + extent
               for other, start, length in previous):
            raise ValueError('paged attention writes must be disjoint from live operands')
        previous.append((storage, begin, extent))


class PreparedPagedKVWriter:
    """Unquantized cache layout copy with caller-visible sticky bounds status."""

    def __init__(self, key, value, key_cache, value_cache, slots, *, workspace, max_cache_pages=None):
        if not isinstance(workspace, ScratchArena) or workspace.device != key.device:
            raise ValueError('paged KV writer requires the preparation scratch/finalizer lifecycle')
        device = key.device
        if device.type != 'cuda' or key.dtype != torch.bfloat16:
            raise ValueError('external Qwen paged KV writes require device BF16 storage')
        for tensor in (key, value):
            _rows(tensor, dimensions=3, dtype=key.dtype, device=device)
        for tensor in (key_cache, value_cache):
            _rows(tensor, dimensions=4, dtype=key.dtype, device=device)
        if (key.shape != value.shape or key_cache.shape != value_cache.shape
                or key.shape[1:] != key_cache.shape[2:] or key.shape[2] != 256
                or key_cache.shape[1] != 64):
            raise ValueError('external Qwen KV writer requires page64/D256 and matching K/V geometry')
        if (slots.shape != (key.shape[0],) or slots.dtype != torch.int64 or slots.device != device
                or slots.stride(0) <= 0):
            raise ValueError('paged KV slots must be device int64 with one slot per input row')
        self.pages = key_cache.shape[0] if max_cache_pages is None else max_cache_pages
        if type(self.pages) is not int or self.pages < key_cache.shape[0]:
            raise ValueError('planned cache capacity must cover bound KV pages')
        self.arguments = {'key': key, 'value': value, 'k_cache': key_cache, 'v_cache': value_cache, 'slots': slots}
        self.error = torch.zeros((), dtype=torch.int32, device=device)
        self.device_errors = (self.error,)
        self.cache_abi = tuple((tuple(t.shape[1:]), t.stride(), t.dtype, t.device) for t in (key_cache, value_cache))
        self.rebind()
        self.compiled = None
        workspace.register_finalizer(self.prepare)

    def rebind(self):
        a = self.arguments
        caches = (a['k_cache'], a['v_cache'])
        if (tuple((tuple(t.shape[1:]), t.stride(), t.dtype, t.device) for t in caches) != self.cache_abi
                or any(not 0 < t.shape[0] <= self.pages for t in caches) or caches[0].shape != caches[1].shape):
            raise ValueError('paged KV rebinding must preserve the planned page ABI and capacity')
        _disjoint((a['key'], a['value'], a['slots']), (*caches, self.error))

    def _arguments(self):
        a = self.arguments
        key, value, kc, vc, slots = (a[name] for name in ('key', 'value', 'k_cache', 'v_cache', 'slots'))
        return (key, value, slots, kc, vc, self.error, *key.stride()[:2], *value.stride()[:2], slots.stride(0),
                *kc.stride()[:3], *vc.stride()[:3], key.shape[1], key.shape[2], kc.shape[1], self.pages * kc.shape[1])

    def prepare(self):
        import triton

        from ignition.preparation.triton import warmup_triton

        from .paged_kv_copy import write_rows

        key = self.arguments['key']
        self.grid = (key.shape[0],)
        self.block = triton.next_power_of_2(key.shape[1] * key.shape[2])
        self.compiled = warmup_triton(write_rows, (*self._arguments(), self.block), self.grid, num_warps=4)

    def __call__(self):
        from ignition.preparation.triton import launch_triton

        from .paged_kv_copy import write_rows

        if self.compiled is None:
            raise RuntimeError('finalize paged KV bindings before warmup/capture')
        launch_triton(write_rows, self.compiled, (*self._arguments(), self.block), self.grid, num_warps=4)


class PreparedFlashAttention3:
    def __init__(self, library, query, key, value, pages, lengths, starts, output, *, width, splits=0):
        device = query.device
        if device.type != 'cuda' or query.dtype != torch.bfloat16:
            raise ValueError('external Qwen attention requires BF16 device storage')
        for tensor in (query, output):
            _rows(tensor, dimensions=3, dtype=query.dtype, device=device)
        for tensor in (key, value):
            _rows(tensor, dimensions=4, dtype=query.dtype, device=device)
        if (query.shape != output.shape or key.shape != value.shape or query.shape[-1] != 256
                or key.shape[1] != 64 or key.shape[-1] != 256 or query.shape[1] % key.shape[2]):
            raise ValueError('FA3 Qwen attention requires page64/D256 with integral GQA head groups')
        _rows(pages, dimensions=2, dtype=torch.int32, device=device)
        for tensor, shape in ((lengths, (pages.shape[0],)), (starts, (pages.shape[0] + 1,))):
            if tensor.shape != shape or tensor.dtype != torch.int32 or tensor.device != device or not tensor.is_contiguous():
                raise ValueError('FA3 packed lengths/starts require contiguous device int32 rows')
        if type(width) is not int or width < 1 or type(splits) is not int or not 0 <= splits <= 256:
            raise ValueError('FA3 requires positive maximum query length and a legal split count')
        self.library, self.width, self.splits = library, width, splits
        self.arguments = {'q': query, 'k_cache': key, 'v_cache': value, 'page_table': pages,
                              'cache_seqlens': lengths, 'cu_seqlens_q': starts, 'output': output}
        self.cache_abi = tuple((tuple(t.shape[1:]), t.stride(), t.dtype, t.device) for t in (key, value))
        self.rebind()

    def __getstate__(self):
        return {key: value for key, value in vars(self).items() if key != 'library'}

    def rebind(self):
        a = self.arguments
        caches = (a['k_cache'], a['v_cache'])
        if (tuple((tuple(t.shape[1:]), t.stride(), t.dtype, t.device) for t in caches) != self.cache_abi
                or caches[0].shape != caches[1].shape or caches[0].shape[0] < 1):
            raise ValueError('FA3 rebinding must preserve the cache page ABI')
        _disjoint(tuple(tensor for name, tensor in a.items() if name != 'output'), (a['output'],))

    def __call__(self):
        a = self.arguments
        a['output'].zero_()
        _attend(a['q'], a['k_cache'], a['v_cache'], a['page_table'], a['cache_seqlens'],
                a['cu_seqlens_q'], a['output'], self.width, self.splits)
        return a['output']
