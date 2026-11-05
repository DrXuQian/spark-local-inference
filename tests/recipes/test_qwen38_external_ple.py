# SPDX-License-Identifier: Apache-2.0

"""Exact GGUF PLE geometry and native table metadata without executable b12x."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('actlize_la.serving')

from ignition.recipes.qwen38.gguf_ple import GgufPLEGeometry, GgufPLETable, external_layout
from ignition.recipes.qwen38.ple import PLEMetadata, PLEScratch


@pytest.fixture
def checkpoint():
    primes = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
              20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171)
    offsets = (0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
               160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275)
    config = dict(vocab_size=248320, ngram_size=3, heads_per_ngram=8, ple_embed_dim=2560,
                  eos_token_id=[248044, 248046], make_ngram_vocab_size_divisible_by=128)
    metadata = {'qwen4exp.ple.eos_token_id': 248044, 'qwen4exp.ple.head_vocab_sizes': primes,
                'qwen4exp.ple.head_offsets': offsets,
                'qwen4exp.ple.layer_multipliers': (23703573157769, 20109073645365, 8052911324071)}
    return config, metadata


def test_exact_checkpoint_geometry_and_exported_hash_contract(checkpoint):
    config, metadata = checkpoint
    layout = external_layout(config, metadata, 'cpu')
    tensors = GgufPLEGeometry.allocate(layout.geometry, 'cpu')
    assert layout.caps.head_count == 16 and layout.geometry.padded_vocab_size == 320001536
    assert sum(getattr(tensors, k).nbytes for k in ('prime_sizes', 'table_offsets', 'multipliers')) == 280
    for field, key in (('prime_sizes', 'head_vocab_sizes'), ('table_offsets', 'head_offsets'), ('multipliers', 'layer_multipliers')):
        assert tuple(getattr(tensors, field).tolist()) == metadata['qwen4exp.ple.' + key]
    table = GgufPLETable(config, layout, tensors, None, None, None, None,
        ((SimpleNamespace(path='model.gguf', offset=4096, nbytes=320001536 * 90),),))
    exported = table.export_metadata()
    assert exported['pad_token_id'] == 248044 and exported['compressed_vocabulary'] == 248320
    row = exported['tables'][0]
    assert row['rows'] == 320001536 and row['scale_block'] == 32 and row['encoding'] == 'GGUF_IQ4_NL'
    assert row['hash'] == dict(method='EOS_XOR_MODULO', orders=[2] * 8 + [3] * 8,
        multipliers=list(tensors.geometry.multipliers), primes=list(tensors.geometry.prime_sizes),
        offsets=list(tensors.geometry.table_offsets))
    assert table.resident_catalog()['tables'][0]['planes'][0]['row_bytes'] == 90
    rows = PLEMetadata.allocate(17, 3, 'cpu')
    scratch = PLEScratch.allocate(17, 'meta')
    assert rows.convolution_slots.tolist() == [-1] * 3 and rows.committed_history.shape == (3, 2)
    assert scratch.embedding.shape == (17, 2560) and scratch.output.shape == (17, 10240)


@pytest.mark.parametrize('field,value', [('eos_token_id', 248046), ('head_offsets', (1,) * 16),
    ('head_vocab_sizes', (20000000,) * 16), ('layer_multipliers', (2, 3, 5)),
    ('layer_multipliers', ((1 << 63) - 1, 3, 5))])
def test_changed_or_unsafe_checkpoint_geometry_is_rejected(checkpoint, field, value):
    config, metadata = checkpoint
    metadata['qwen4exp.ple.' + field] = value
    with pytest.raises(ValueError):
        external_layout(config, metadata, 'cpu')


@pytest.mark.parametrize('field,value', [('ngram_size', 4), ('heads_per_ngram', 4), ('ple_embed_dim', 160),
                                       ('make_ngram_vocab_size_divisible_by', 1)])
def test_changed_frontend_geometry_is_rejected(checkpoint, field, value):
    config, metadata = checkpoint
    config[field] = value
    with pytest.raises(ValueError):
        external_layout(config, metadata, 'cpu')
