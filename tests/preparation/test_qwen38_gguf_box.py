# SPDX-License-Identifier: Apache-2.0

"""Box input resolution stays bounded and rejects mismatched frontend assets."""

import hashlib
import importlib.machinery
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qualify import qwen38_gguf_box as box


def files(tmp_path):
    directory = tmp_path / 'UD-Q4_K_XL'
    directory.mkdir()
    for number in (1, 4):
        (directory / f'{box.TARGET_STEM}-{number:05d}-of-00004.gguf').touch()
    mtp = tmp_path / 'MTP' / box.MTP_NAME
    mtp.parent.mkdir()
    mtp.touch()
    return directory, mtp


def test_partial_inputs_need_only_first_and_fourth_shards_and_resolve_shared_mtp(tmp_path):
    directory, mtp = files(tmp_path)
    target, metadata, selected = box.model_files(directory)
    assert target.name.endswith('-00004-of-00004.gguf')
    assert metadata.name.endswith('-00001-of-00004.gguf')
    assert selected == mtp
    (directory / box.MTP_NAME).symlink_to(mtp)
    assert box.model_files(directory)[2] == mtp


def test_missing_and_ambiguous_mtp_require_an_explicit_file(tmp_path):
    directory, mtp = files(tmp_path)
    second = directory / box.MTP_NAME
    second.touch()
    with pytest.raises(ValueError, match='multiple shared-Q8 MTP'):
        box.model_files(directory)
    assert box.model_files(directory, mtp)[2] == mtp
    mtp.unlink()
    second.unlink()
    with pytest.raises(ValueError, match='supply --mtp PATH'):
        box.model_files(directory)


def test_missing_last_shard_is_reported_without_requesting_other_shards(tmp_path):
    directory, mtp = files(tmp_path)
    (directory / f'{box.TARGET_STEM}-00004-of-00004.gguf').unlink()
    with pytest.raises(ValueError, match='00004-of-00004') as error:
        box.model_files(directory, mtp)
    assert '00002-of-00004' not in str(error.value)


def test_frontend_checks_actual_bytes_before_preparation(tmp_path, monkeypatch):
    expected = {}
    for name in box.TEXT_ASSETS:
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(box, 'TEXT_ASSETS', expected)
    assert box.frontend_files(tmp_path) == tmp_path
    (tmp_path / 'tokenizer.json').write_bytes(b'wrong vocabulary')
    with pytest.raises(ValueError, match='tokenizer.json.*pinned Flare frontend'):
        box.frontend_files(tmp_path)


def test_explicit_frontend_download_is_limited_to_pinned_text_files(tmp_path, monkeypatch):
    allowed = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}
    requested = []

    def download(repository, name, *, revision, local_dir):
        assert repository == box.MODEL_ID and revision == box.REVISION
        assert name in allowed
        requested.append(name)
        (Path(local_dir) / name).write_bytes(name.encode())

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(box, 'TEXT_ASSETS', {name: hashlib.sha256(name.encode()).hexdigest()
                                         for name in box.TEXT_ASSETS})
    assert box.frontend_files(tmp_path, fetch=True) == tmp_path
    assert set(requested) == allowed and len(requested) == 4


def test_attention_location_does_not_import_its_host_library(tmp_path, monkeypatch):
    package = tmp_path / 'flash_attn_3'
    package.mkdir()
    (package / '__init__.py').write_text('raise AssertionError("imported before recording")\n')
    extension = package / ('_C' + importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'flash_attn_3', raising=False)
    monkeypatch.delitem(sys.modules, 'flash_attn_3._C', raising=False)
    assert box.installed_attention() == extension
    assert 'flash_attn_3' not in sys.modules and 'flash_attn_3._C' not in sys.modules


def sparse_mtp(path, *, layer=48, shared=True, gate_type='Q2_K', up_type='Q2_K',
               down_type='Q8_0', mtp_compression=0):
    """Real full-shape GGUF headers with sparse zero payloads, never weight arrays."""
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_uint32('qwen4exp.block_count', 49)
    writer.add_uint32('qwen4exp.nextn_predict_layers', 1)
    writer.add_bool('qwen4exp.nextn_shared_target_tensors', shared)
    writer.add_array('qwen4exp.attention.compress_ratios', [4 if i % 4 == 3 else 0 for i in range(48)]
                     + [mtp_compression])
    payload = 0
    for part, shape, encoding in (('gate', (512, 640, 2560), gate_type),
                                  ('up', (512, 640, 2560), up_type),
                                  ('down', (512, 2560, 640), down_type)):
        qtype = gguf.GGMLQuantizationType[encoding]
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        raw_shape = (*shape[:-1], shape[-1] // block * size)
        nbytes = shape[0] * shape[1] * raw_shape[-1]
        writer.add_tensor_info(f'blk.{layer}.ffn_{part}_exps.weight', raw_shape, np.uint8, nbytes, raw_dtype=qtype)
        payload += gguf.GGUFWriter.ggml_pad(nbytes, writer.data_alignment)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    output = writer.fout[0]
    writer.write_padding(output, output.tell())
    output.truncate(output.tell() + payload)
    writer.close()


def test_inspection_reads_types_from_headers_not_filenames(tmp_path, capsys):
    path = tmp_path / 'claims-bf16.gguf'
    sparse_mtp(path)
    box.inspect_mtp(path)
    output = capsys.readouterr().out
    assert 'Q2_K: 2 tensors, 550502400 stored bytes' in output
    assert 'blk.48.ffn_down_exps.weight: Q8_0 (512, 2560, 640)' in output
    box.validate_mtp(path)


@pytest.mark.parametrize('options,error', [
    ({'layer': 0}, 'only blk.48'),
    ({'shared': False}, 'nextn_shared_target_tensors'),
    ({'mtp_compression': 4}, 'uncompressed layer 48'),
    ({'gate_type': 'BF16', 'up_type': 'BF16'}, 'requires quantized blocks'),
    ({'up_type': 'Q8_0'}, 'equal encodings'),
])
def test_mtp_boundary_rejects_incompatible_actual_headers(tmp_path, options, error):
    path = tmp_path / 'mtp-q2_0.gguf'
    sparse_mtp(path, **options)
    with pytest.raises(ValueError, match=error):
        box.validate_mtp(path)


def test_inspect_only_needs_no_frontend_operators_or_patch_application(tmp_path, monkeypatch, capsys):
    from ignition import patches

    directory, mtp = files(tmp_path)
    sparse_mtp(mtp)

    def forbidden(*args, **kwargs):
        pytest.fail('header inspection must not enter export preparation')

    monkeypatch.setattr(box, 'frontend_files', forbidden)
    monkeypatch.setattr(box, 'installed_attention', forbidden)
    monkeypatch.setattr(box.subprocess, 'run', forbidden)
    monkeypatch.setattr(patches, 'apply', forbidden)
    assert box.main(['--model-dir', str(directory), '--mtp', str(mtp), '--inspect-only']) == 0
    assert 'device execution are not qualified' in capsys.readouterr().out


def raw_type_mtp(path, type_id=42, *, truncated=False):
    import gguf
    import numpy as np

    # Use the writer's logical-shape input to preserve an arbitrary raw ID.
    # Six Q2_0 rows of 640 values use 6 * (640/64) * 18 bytes.
    nbytes = 1080
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor_info('blk.48.ffn_down_exps.weight', (2, 3, 640), np.float32,
                           nbytes, raw_dtype=type_id)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    output = writer.fout[0]
    writer.write_padding(output, output.tell())
    output.truncate(output.tell() + nbytes - int(truncated))
    writer.close()


def test_q2_0_headers_report_the_extension_without_modifying_the_reader(tmp_path, capsys):
    import gguf

    path = tmp_path / 'mtp-q2_0.gguf'
    raw_type_mtp(path)
    before = dict(gguf.GGMLQuantizationType.__members__)
    supported = any(int(kind) == 42 for kind in before.values())
    if not supported:
        with pytest.raises(ValueError, match='42.*GGMLQuantizationType'):
            gguf.GGUFReader(path)
    assert box.inspect_mtp(path) == (frozenset() if supported else frozenset({42}))
    output = capsys.readouterr().out
    assert 'Q2_0: 1 tensors, 1080 stored bytes' in output
    assert 'Q2_0 (2, 3, 640), type_id=42' in output
    assert gguf.GGMLQuantizationType.__members__ == before
    if supported:
        assert gguf.GGUFReader(path).tensors[0].tensor_type.name == 'Q2_0'
    else:
        assert 'Q2_K is a different format (type_id=10' in output
        with pytest.raises(ValueError, match='42.*GGMLQuantizationType'):
            gguf.GGUFReader(path)


def test_extended_reader_accepts_full_q2_0_mtp_expert_geometry(tmp_path):
    import gguf

    if not hasattr(gguf.GGMLQuantizationType, 'Q2_0'):
        pytest.skip('requires the PPU GGUF reader wheel')
    path = tmp_path / 'mtp.gguf'
    sparse_mtp(path, gate_type='Q2_0', up_type='Q2_0', down_type='Q2_0')
    box.validate_mtp(path)
    from ignition.preparation.gguf import inspect_gguf_checkpoint

    checkpoint = inspect_gguf_checkpoint(path)
    assert {tensor.encoding for tensor in checkpoint.tensors.values()} == {'Q2_0'}
    assert sum(tensor.nbytes for tensor in checkpoint.tensors.values()) == 707788800


def test_unknown_raw_ids_keep_their_number_without_inventing_a_block_size(tmp_path, capsys):
    path = tmp_path / 'unknown.gguf'
    raw_type_mtp(path, 4242)
    assert box.inspect_mtp(path) == frozenset({4242})
    output = capsys.readouterr().out
    assert 'UNKNOWN_4242: 1 tensors, stored bytes unknown' in output
    assert 'type_id=4242' in output


def test_known_extension_rejects_truncated_payload_extent(tmp_path):
    path = tmp_path / 'truncated.gguf'
    raw_type_mtp(path, truncated=True)
    with pytest.raises(ValueError, match='truncated GGUF tensor storage'):
        box.inspect_mtp(path)


def test_raw_type_inspection_succeeds_but_export_stops_before_preparation(tmp_path, monkeypatch, capsys):
    directory, mtp = files(tmp_path)
    raw_type_mtp(mtp, 4242)

    def forbidden(*args, **kwargs):
        pytest.fail('unknown GGUF types must not enter export preparation')

    monkeypatch.setattr(box, 'validate_mtp', forbidden)
    monkeypatch.setattr(box, 'frontend_files', forbidden)
    monkeypatch.setattr(box, 'installed_attention', forbidden)
    monkeypatch.setattr(box.subprocess, 'run', forbidden)
    arguments = ['--model-dir', str(directory), '--mtp', str(mtp)]
    assert box.main([*arguments, '--inspect-only']) == 0
    with pytest.raises(SystemExit) as error:
        box.main([*arguments, '--operator-target', 'ppu10', '--fetch-frontend',
                  '--blas-library', str(tmp_path / 'unloaded.so'), '--output', str(tmp_path / 'output')])
    assert error.value.code == 2
    assert 'type IDs [4242]; header inspection does not enable execution' in capsys.readouterr().err
