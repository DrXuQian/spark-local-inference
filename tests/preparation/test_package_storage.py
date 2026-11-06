# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Package storage keeps only requested allocator bytes and, on unified memory, library
allocations above the addresses a serving context occupies."""

import ctypes
import inspect
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.package_storage import PackageStorage, ScratchPool, pool_snapshot


@pytest.mark.parametrize('pool_id', [(0, 7), (3, 0)])
def test_pool_snapshot_uses_real_torch_wrapper_without_changing_metadata(monkeypatch, pool_id):
    segments = [dict(address=0x80000000000, total_size=512, segment_pool_id=pool_id,
                     blocks=[dict(address=0x80000000000, size=256, requested_size=193,
                                  state='active_awaiting_free')])]
    calls = []

    def native_snapshot(argument):
        calls.append(argument)
        return {'segments': segments}

    # Keep the installed Torch Python wrapper, replacing only its native device
    # boundary. Run this test with both Torch 2.9 and the newer preparation Torch.
    monkeypatch.setattr(torch._C, '_cuda_memorySnapshot', native_snapshot, raising=False)
    assert pool_snapshot(pool_id) is segments
    traces_supported = 'include_traces' in inspect.signature(torch.cuda.memory_snapshot).parameters
    assert calls == [(*pool_id, False) if traces_supported else pool_id]


def test_pool_snapshot_does_not_retry_allocator_type_errors(monkeypatch):
    calls = []

    def native_snapshot(argument):
        calls.append(argument)
        raise TypeError('allocator failure')

    monkeypatch.setattr(torch._C, '_cuda_memorySnapshot', native_snapshot, raising=False)
    with pytest.raises(TypeError, match='allocator failure'):
        pool_snapshot((0, 7))
    assert len(calls) == 1


@pytest.mark.parametrize('fail_second_unmap', [False, True])
def test_scratch_snapshot_keeps_pool_scope_and_restores_partial_unmaps(monkeypatch, fail_second_unmap):
    from ignition.preparation import package_storage

    addresses = (0x80000000000, 0x80000010000)
    pool_id = (0, 7)
    calls, restored = [], []
    traces_supported = 'include_traces' in inspect.signature(torch.cuda.memory_snapshot).parameters

    def native_snapshot(argument):
        assert argument == ((*pool_id, False) if traces_supported else pool_id)
        return {'segments': [dict(address=address, total_size=65536) for address in addresses]}

    def cuda_call(name, address, size):
        calls.append((name, address.value, size.value))
        if fail_second_unmap and address.value == addresses[1]:
            raise RuntimeError('unmap failed')

    monkeypatch.setattr(torch._C, '_cuda_memorySnapshot', native_snapshot, raising=False)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(('synchronize', device)))
    monkeypatch.setattr(package_storage, 'DeviceRegion',
                        lambda cuda, address, size, device: restored.append((address, size, device)))
    pool = ScratchPool.__new__(ScratchPool)
    pool.pool_id, pool.unbacked = pool_id, []
    pool.storage = SimpleNamespace(device=2, cuda=SimpleNamespace(call=cuda_call),
                                   mappings={a: SimpleNamespace(nbytes=65536) for a in addresses})
    if fail_second_unmap:
        with pytest.raises(RuntimeError, match='unmap failed'):
            pool.release_backing()
        assert restored == [(addresses[0], 65536, 2)] and pool.unbacked == []
    else:
        pool.release_backing()
        assert pool.unbacked == [(a, 65536) for a in addresses] and restored == []
    assert calls == [('synchronize', 2), *(('cuMemUnmap', a, 65536) for a in addresses)]


@pytest.mark.parametrize("local_ranks", [0, 4, True])
def test_local_export_rank_count_must_fit_the_global_group(local_ranks):
    with pytest.raises(ValueError, match="local preparation ranks"):
        PackageStorage(world=3, local_ranks=local_ranks)


def test_cuda_access_descriptor_array_matches_driver_abi():
    from ignition.preparation.package_storage import _Access, _Location

    access = (_Access * 2)(_Access(_Location(3, 0), 3), _Access(_Location(1, 7), 3))
    assert bytes(access) == struct.pack("<6i", 3, 0, 3, 1, 7, 3)
    assert ctypes.sizeof(_Access) == 12


def test_unused_bytes_include_live_block_padding_and_retired_pools(monkeypatch):
    base = 0x200000000000
    storage = PackageStorage.__new__(PackageStorage)
    storage.device = 0
    storage._preparation_pools = {(0, 1), (0, 2)}

    def block(offset, size, requested, state="active_allocated"):
        return dict(address=base + offset, size=size, requested_size=requested, state=state)

    def segment(device, pool, blocks):
        return dict(device=device, segment_pool_id=pool, blocks=blocks)

    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [
        segment(0, (0, 1), [block(0, 64, 32), block(64, 32, 0, "inactive"), block(96, 32, 32)]),
        segment(0, (0, 2), [block(128, 64, 24, "active_awaiting_free")]),
        segment(0, (0, 3), [block(192, 64, 0, "inactive")]),
        segment(1, (0, 1), [block(256, 64, 0, "inactive")]),
    ])
    unused = storage.unused_ranges()
    assert unused == [(base + 32, base + 64), (base + 64, base + 96), (base + 152, base + 192)]

@pytest.mark.skipif(os.name == "nt", reason="Linux package address layout")
def test_linux_device_ranges_avoid_mimalloc_arena_hints():
    # mimalloc hints its aligned arena reservations into 2-6 TiB; PyTorch allocates CPU tensors there.
    hint_start, hint_end = 2 << 40, 6 << 40
    for rank in range(16):
        start = PackageStorage.DEVICE_BASE + rank * PackageStorage.RESERVATION_BYTES
        end = start + PackageStorage.RESERVATION_BYTES
        assert end <= hint_start or start >= hint_end


@pytest.mark.skipif(os.name != "nt", reason="Windows WDDM address layout")
def test_windows_host_range_stays_below_stalling_wddm_addresses():
    # RTX Spark's driver stalls on host-address hints at 16 TiB and 32 TiB.
    assert PackageStorage.DEVICE_BASE + PackageStorage.RESERVATION_BYTES <= PackageStorage.HOST_BASE
    assert PackageStorage.HOST_BASE + PackageStorage.RESERVATION_BYTES <= 16 << 40
