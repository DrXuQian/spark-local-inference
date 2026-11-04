# SPDX-License-Identifier: Apache-2.0

import json

import pytest

gguf = pytest.importorskip('gguf')

from ignition.preparation.checkpoint import CheckpointError
from ignition.preparation.qwen38_gguf import validate_frontend_token_ids


def fixture(tmp_path, *, tokens=('a', 'b', 'ab', '<eos>', '[PAD4]'), merges=('a b',)):
    checkpoint = tmp_path / 'metadata.gguf'
    writer = gguf.GGUFWriter(checkpoint, 'qwen4exp')
    writer.add_token_list(list(tokens))
    if merges:
        writer.add_token_merges(list(merges))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.close()
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text(json.dumps({'model': {'vocab': {'a': 0, 'b': 1, 'ab': 2}, 'merges': [['a', 'b']]},
        'added_tokens': [{'id': 3, 'content': '<eos>'}]}))
    return checkpoint, tokenizer


def test_active_ids_with_gguf_padding_tail(tmp_path):
    validate_frontend_token_ids(*fixture(tmp_path))


@pytest.mark.parametrize('tokens,merges,error', [
    (('b', 'a', 'ab', '<eos>', '[PAD4]'), ('a b',), 'token ID'),
    (('a', 'b', 'ab', '<eos>', 'active-extra'), ('a b',), 'padding row'),
    (('a', 'b', 'ab', '<eos>'), ('b a',), 'BPE merges'),
    (('a', 'b', 'ab'), ('a b',), 'token ID'),
])
def test_frontend_mismatch_is_rejected(tmp_path, tokens, merges, error):
    with pytest.raises(CheckpointError, match=error):
        validate_frontend_token_ids(*fixture(tmp_path, tokens=tokens, merges=merges))
