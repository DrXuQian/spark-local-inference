# SPDX-License-Identifier: Apache-2.0

"""Exercise file bytes and recipe mapping with a host copy transport, not kernels."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip('gguf')

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.recipes.qwen38.mtp_gguf_weights import HFMtpGgufWeightLoader
from tests.preparation.qwen38_mtp_gguf_fixture import inventory, write_hf_mtp


@pytest.fixture(autouse=True)
def host_only(monkeypatch):
    if not hasattr(gguf.GGMLQuantizationType, 'Q2_0'):
        pytest.skip('requires the PPU GGUF reader with Q2_0 ID42')
    def forbidden(*args, **kwargs):
        pytest.fail('host weight mapping must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def stored(tmp_path):
    values = {}
    for index, entry in enumerate(inventory()):
        if entry['encoding'] == 'F32':
            values[entry['name']] = ((np.arange(entry['shape'][0]) % 9 - 4) / 8).astype('<f4')
        elif entry['encoding'] == 'BF16':
            # Only a witness prefix is materialized; the remaining sparse file is zero.
            values[entry['name']] = np.full(32, 0x3e00 + index * 16, dtype='<u2')
    path = write_hf_mtp(tmp_path / 'mtp.gguf', values=values)
    manifest = inspect_gguf_checkpoint(path)
    reader = gguf.GGUFReader(path, mode='c')  # Private writable pages, never change the source file.
    sources = {}
    for tensor in reader.tensors:
        metadata = manifest.tensors[tensor.name]
        value = torch.from_numpy(tensor.data)
        if metadata.encoding == 'BF16':
            value = value.view(torch.bfloat16)
        sources[tensor.name] = value.reshape(metadata.storage_shape)
    return manifest, sources, reader


def loader_for(stored, rank=0, world=1):
    manifest, sources, _ = stored
    loader = HFMtpGgufWeightLoader(manifest, 'cpu', shard=TensorParallelShard(rank, world))
    # Use the real parameter/quantized-row mapping with only the transfer replaced.
    # Production DirectWeightLoader still requires an actual CUDA/PPU context.
    loader._session = SimpleNamespace(flush=lambda: None)
    loader._sources = sources
    loader._copy = lambda destination, source: destination.copy_(source)
    return loader


def module_owner():
    # Canonical MTP's 24 parameter owners, without importing Transformers/device packages.
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleList([torch.nn.Module()])
    for entry in inventory():
        name = entry['name'].removeprefix('mtp.')
        if name.startswith('layers.0.mlp.'):
            continue
        owner = module
        *parts, attribute = name.split('.')
        for part in parts:
            if not hasattr(owner, part):
                owner.add_module(part, torch.nn.Module())
            owner = getattr(owner, part)
        owner.register_parameter(attribute, torch.nn.Parameter(torch.empty(entry['shape'], device='meta')))
    return module


@pytest.mark.parametrize('rank,world', [(0, 1), (1, 2)])
def test_plain_parameters_preserve_zero_centered_norms_and_qkv_order(stored, rank, world):
    loader = loader_for(stored, rank, world)
    module = module_owner()
    attention = loader.load_module(module)
    _, sources, _ = stored
    assert module.layers[0].self_attn.compress_ratio == 0
    ranges = {'q_proj': (rank * 12288 // world, (rank + 1) * 12288 // world),
              'k_proj': (rank * 512 // world, (rank + 1) * 512 // world),
              'v_proj': (rank * 512 // world, (rank + 1) * 512 // world)}
    for name, parameter in module.named_parameters():
        expected = sources['mtp.' + name]
        part = name.split('.')[-2]
        if name.startswith('layers.0.self_attn.') and part in ranges:
            first, last = ranges[part]
            expected = expected[first:last]
        elif name == 'layers.0.self_attn.o_proj.weight':
            expected = expected[:, rank * 6144 // world:(rank + 1) * 6144 // world]
        torch.testing.assert_close(parameter.float(), expected.float(), rtol=0, atol=0)
        assert parameter.dtype == (torch.bfloat16 if name.startswith('pre_fc_norm_') else expected.dtype)
    qkv = attention['qkv_proj']
    cursor = 0
    for part in ('q', 'k', 'v'):
        weight = getattr(module.layers[0].self_attn, part + '_proj').weight
        assert weight.data_ptr() == qkv[cursor].data_ptr()
        cursor += weight.shape[0]
    assert cursor == qkv.shape[0]
    assert module.pre_fc_norm_embedding.weight[0].item() == -.5  # Neither +1 nor -1 was applied.


@pytest.mark.parametrize('rank,world', [(0, 1), (1, 2)])
def test_fused_experts_keep_compressed_bytes_and_each_expert_partition(stored, rank, world):
    loader = loader_for(stored, rank, world)
    prefix = 'mtp.layers.0.mlp'
    gate = loader._sources[prefix + '.experts.gate_up_proj']
    down = loader._sources[prefix + '.experts.down_proj']
    for expert in (0, 1, 511):
        for row in (0, 319, 320, 639, 640, 959, 960, 1279):
            gate[expert, row, 0].fill_((expert + row) % 251)
        down[expert, :, :, 2].fill_(expert % 251)
    weights = loader.load_moe()
    first, last = rank * 640 // world, (rank + 1) * 640 // world
    width = last - first
    packed = weights.experts.gate_up
    assert packed.shape == (512, 2 * width, 2560) and packed.qtype == 42
    assert packed.raw.dtype == torch.uint8
    assert torch.equal(packed.raw[:, :width], gate[:, first:last])
    assert torch.equal(packed.raw[:, width:], gate[:, 640 + first:640 + last])
    assert torch.equal(weights.experts.down.raw, down[:, :, first // 64:last // 64])
    assert packed.raw.nbytes + weights.experts.down.raw.nbytes == 707788800 // world
    assert torch.equal(weights.shared.gate_up[:width], loader._sources[prefix + '.shared_expert.gate_proj.weight'][first:last])
    assert torch.equal(weights.shared.gate_up[width:], loader._sources[prefix + '.shared_expert.up_proj.weight'][first:last])
    assert torch.equal(weights.router, loader._sources[prefix + '.gate.weight'])
    assert weights.router.dtype == torch.bfloat16


@pytest.mark.parametrize('value,error', [(float('nan'), 'finite'), (.1234567, 'retain its values in BF16')])
def test_feedback_norm_rejects_nonfinite_or_lossy_conversion(stored, value, error):
    loader = loader_for(stored)
    loader._sources['mtp.pre_fc_norm_embedding.weight'][0] = value
    with pytest.raises(ValueError, match=error):
        loader.load_module(module_owner())


def test_module_coverage_is_checked_before_transfers(stored):
    module = module_owner()
    del module.pre_fc_norm_hidden
    with pytest.raises(ValueError, match='module parameters differ'):
        loader_for(stored).load_module(module)


def test_mtp_entrypoint_selects_hf_layout_and_retains_shared_endpoints(stored, monkeypatch):
    from ignition.recipes.qwen38 import mtp_gguf_weights, mtp_weights

    class HostTransferLoader(HFMtpGgufWeightLoader):
        def __enter__(self):
            self._session = SimpleNamespace(flush=lambda: None)
            self._sources = stored[1]
            self._copy = lambda destination, source: destination.copy_(source)
            return self

        def __exit__(self, *args):
            self._session = None
            return False

    # Only transfer/context setup and canonical owner construction are replaced;
    # the real entry point, layout validation, parameter and expert loaders run.
    monkeypatch.setattr(mtp_gguf_weights, 'HFMtpGgufWeightLoader', HostTransferLoader)
    monkeypatch.setattr(mtp_weights, 'create_mtp_modules', lambda manifest: module_owner())
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    embedding, head = object(), object()
    result = mtp_weights.MTPWeights.load_gguf(stored[0], object(), 'cpu',
        embedding_weight=embedding, lm_head_weight=head)
    assert result.embedding is embedding and result.lm_head is head
    assert result.moe.experts.gate_up.qtype == result.moe.experts.down.qtype == 42
    assert result.stored_bytes == 889014272
    assert result.module.layers[0].self_attn.compress_ratio == 0
    assert result.module.pre_fc_norm_hidden.weight[0].item() == -.5
