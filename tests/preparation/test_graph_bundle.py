# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from ignition.preparation.graphs import PreparedGraphBundle


def bundle(**overrides):
    arguments = {"device": 0, "graphs": {"decode": object()}, "buffers": {}}
    arguments.update(overrides)
    return PreparedGraphBundle(**arguments)


def test_import_does_not_load_torch_or_native_extension():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import ignition.preparation.graphs; "
        "assert 'torch' not in sys.modules; assert 'flare_heliopause' not in sys.modules",
    ], check=True)


@pytest.mark.parametrize("arguments", [
    {"device": -1}, {"device": True}, {"device": "cuda:0"},
    {"graphs": {}}, {"graphs": {" ": object()}},
    {"buffers": {"": object()}}, {"buffers": {1: object()}},
    {"operator_target": 'unknown'}, {"operator_target": True},
])
def test_invalid_identity_rejected_before_cuda_import(arguments):
    with pytest.raises(ValueError):
        bundle(**arguments)


def test_registration_metadata_is_copied_and_read_only():
    graphs = {"decode": object(), "prefill": object()}
    buffers = {"state": object()}
    resources = [object()]
    prepared = bundle(graphs=graphs, buffers=buffers, resources=resources)
    graphs.clear()
    buffers.clear()
    resources.clear()
    assert tuple(prepared.graphs) == ("decode", "prefill")
    assert tuple(prepared.buffers) == ("state",)
    assert len(prepared.resources) == 1
    assert dict(prepared.graph_indices) == {"decode": 0, "prefill": 1}
    with pytest.raises(TypeError):
        prepared.graph_indices["decode"] = 3


@pytest.fixture
def resource_metadata(monkeypatch):
    """Exercise validation with real CPU storage and reported device metadata."""
    from contextlib import nullcontext
    import torch

    class DeviceTensor(torch.Tensor):
        ordinal = 0

        @property
        def device(self):
            return torch.device('cuda', self.ordinal)

    class Graph:
        def __init__(self, handle=11):
            self.handle = handle

        def raw_cuda_graph_exec(self):
            return self.handle

    native = SimpleNamespace(current_context=lambda: (123, 0))
    monkeypatch.setitem(sys.modules, 'flare_heliopause', native)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    tensor = torch.arange(8, dtype=torch.int32).as_subclass(DeviceTensor)
    return Graph, tensor, native


@pytest.mark.parametrize('target', ['ppu10', 'ppu15'])
def test_explicit_ppu_graph_target_keeps_context_and_storage_validation(resource_metadata, target):
    Graph, tensor, native = resource_metadata
    prepared = bundle(graphs={'decode': Graph()}, buffers={'state': tensor}, operator_target=target)
    assert prepared._validate_resources() == (123, [11], [(tensor.data_ptr(), 32)])
    assert not prepared.consumed
    native.current_context = lambda: (123, 1)
    with pytest.raises(RuntimeError, match='context belongs to device 1'):
        prepared._validate_resources()


@pytest.mark.parametrize('target', [None, 'cuda_sm120'])
def test_ppu_capability_is_not_admitted_without_the_explicit_target(resource_metadata, target):
    Graph, tensor, _ = resource_metadata
    with pytest.raises(ValueError, match='SM'):
        bundle(graphs={'decode': Graph()}, buffers={'state': tensor}, operator_target=target)._validate_resources()


@pytest.mark.parametrize('problem', ['wrong-device', 'strided', 'empty', 'zero-handle', 'duplicate-handles'])
def test_ppu_target_does_not_weaken_graph_and_buffer_checks(resource_metadata, problem):
    Graph, tensor, _ = resource_metadata
    graphs = {'decode': Graph()}
    if problem == 'wrong-device':
        tensor.ordinal = 1
        message = 'must reside on CUDA device 0'
    elif problem == 'strided':
        tensor = tensor[::2]
        message = 'must be contiguous'
    elif problem == 'empty':
        tensor = tensor[:0]
        message = 'nonempty allocated storage'
    elif problem == 'zero-handle':
        graphs['decode'].handle = 0
        message = 'no instantiated executable handle'
    else:
        graphs['prefill'] = Graph()
        message = 'unique executable handles'
    prepared = bundle(graphs=graphs, buffers={'state': tensor}, operator_target='ppu10')
    with pytest.raises(ValueError, match=message):
        prepared._validate_resources()
    assert not prepared.consumed


def test_raw_replay_requires_qualification_binding_before_cuda(monkeypatch):
    prepared = bundle()
    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace())

    def forbidden():
        pytest.fail("qualification feature rejection must precede CUDA validation")

    monkeypatch.setattr(prepared, "_validate_resources", forbidden)
    with pytest.raises(RuntimeError, match="native qualification Cargo feature"):
        prepared.start_worker()
    assert not prepared.consumed


def test_failed_registration_allows_retry_but_success_is_one_shot(monkeypatch):
    prepared = bundle()
    calls = 0
    receiver = object()

    def register(capacity):
        nonlocal calls
        calls += 1
        assert capacity == 3
        if calls == 1:
            raise RuntimeError("worker initialization failed before ready")
        return receiver

    monkeypatch.setattr(prepared, "_register", register)
    with pytest.raises(RuntimeError, match="initialization failed"):
        prepared.start_worker(queue_capacity=3)
    assert not prepared.consumed
    assert prepared.start_worker(queue_capacity=3) is receiver
    assert prepared.consumed
    with pytest.raises(RuntimeError, match="already been consumed"):
        prepared.start_worker(queue_capacity=3)
    assert calls == 2


def test_concurrent_registration_only_transfers_once(monkeypatch):
    prepared = bundle()
    entered = Event()
    release = Event()
    receiver = object()

    def register(_capacity):
        entered.set()
        assert release.wait(5)
        return receiver

    monkeypatch.setattr(prepared, "_register", register)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(prepared.start_worker)
        assert entered.wait(5)
        second = executor.submit(prepared.start_worker)
        release.set()
        assert first.result(timeout=5) is receiver
        with pytest.raises(RuntimeError, match="already been consumed"):
            second.result(timeout=5)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_never_consumes_bundle(capacity):
    prepared = bundle()
    with pytest.raises(ValueError, match="queue_capacity"):
        prepared.start_worker(queue_capacity=capacity)
    assert not prepared.consumed
