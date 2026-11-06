# SPDX-License-Identifier: Apache-2.0

"""Real standalone MTP headers and exact precision/memory admission."""

import pytest

gguf = pytest.importorskip('gguf')

from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.preparation.qwen38_gguf import device_weight_bytes
from ignition.preparation.qwen38_mtp_gguf import validate_mtp_gguf
from qualify.qwen38_gguf_box import validate_mtp, inspect_mtp
from tests.preparation.qwen38_mtp_gguf_fixture import inventory, write_hf_mtp


@pytest.fixture(autouse=True)
def q2_reader():
    if not hasattr(gguf.GGMLQuantizationType, 'Q2_0'):
        pytest.skip('requires the PPU GGUF reader with Q2_0 ID42')


def test_standalone_headers_preserve_names_types_and_compressed_budget(tmp_path, capsys):
    path = write_hf_mtp(tmp_path / 'mtp.gguf')
    validate_mtp(path)
    assert inspect_mtp(path) == frozenset()
    assert 'strata.mtp.expert_format: q2_0' in capsys.readouterr().out
    checkpoint = inspect_gguf_checkpoint(path)
    assert validate_mtp_gguf(checkpoint) == 'qwen4exp-mtp'
    assert len(checkpoint.tensors) == 31
    assert {t.component for t in checkpoint.tensors.values()} == {'mtp'}
    assert sum(t.nbytes for t in checkpoint.tensors.values()) == 889014272
    assert device_weight_bytes(checkpoint) == 889014272
    assert checkpoint.tensors['mtp.layers.0.mlp.experts.gate_up_proj'].shape == (512, 1280, 2560)
    assert checkpoint.tensors['mtp.pre_fc_norm_embedding.weight'].encoding == 'F32'
    assert 'qwen4exp.block_count' not in checkpoint.metadata  # No fabricated legacy metadata.


@pytest.mark.parametrize('fault,error', [
    ('missing', 'tensor coverage'), ('extra', 'tensor coverage'), ('shape', 'requires shape'),
    ('dense', 'requires stored BF16'), ('norm', 'requires stored F32'),
    ('plain_expert', 'requires quantized'), ('mixed_experts', 'expert types differ'),
    ('declaration', 'expert types differ'),
])
def test_standalone_boundary_rejects_incompatible_tensors(tmp_path, fault, error):
    entries = inventory()
    by_name = {e['name']: e for e in entries}
    declared = 'q2_0'
    if fault == 'missing':
        entries.pop()
    elif fault == 'extra':
        entries.append(dict(name='token_embd.weight', shape=[1, 2560], encoding='BF16'))
    elif fault == 'shape':
        by_name['mtp.fc_hidden.weight']['shape'] = [2560, 5120]
    elif fault == 'dense':
        by_name['mtp.fc_embedding.weight']['encoding'] = 'Q8_0'
    elif fault == 'norm':
        by_name['mtp.pre_fc_norm_hidden.weight']['encoding'] = 'BF16'
    elif fault == 'plain_expert':
        by_name['mtp.layers.0.mlp.experts.down_proj']['encoding'] = 'BF16'
    elif fault == 'mixed_experts':
        by_name['mtp.layers.0.mlp.experts.down_proj']['encoding'] = 'Q8_0'
    else:
        declared = 'q8_0'
    path = write_hf_mtp(tmp_path / 'bad.gguf', entries=entries, declared=declared)
    with pytest.raises(ValueError, match=error):
        validate_mtp(path)
