# SPDX-License-Identifier: Apache-2.0

import os
import struct

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from ignition.preparation.checkpoint import CheckpointError
from ignition.preparation.gguf import inspect_gguf_checkpoint


def test_q2_reference_preserves_code_order_and_signed_scales():
    from tests.components.gguf_reference import dequantize

    raw = np.full((6, 18), 0xe4, dtype=np.uint8)
    scales = np.array([1., -2., 0., -0., 2**-24, -2**-24], dtype='<f2')
    raw[:, :2] = scales[:, None].view(np.uint8)
    expected = np.tile(np.array([-1., 0., 1., 2.], dtype=np.float32), 16) * scales.astype(np.float32)[:, None]
    actual = dequantize(raw, 42)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def write_checkpoint(path, tensors, *, split=False, metadata=None, endian=None):
    options = {"split_max_tensors": 1, "small_first_shard": True} if split else {}
    if endian is not None:
        options["endianess"] = endian
    writer = gguf.GGUFWriter(path, "qwen4exp", **options)
    for name, value in (metadata or {}).items():
        if isinstance(value, list):
            writer.add_array(name, value)
        else:
            writer.add_uint32(name, value)
    for name, values, encoding in tensors:
        writer.add_tensor(name, values, raw_dtype=encoding)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_packed_expert_layout_and_bytes(tmp_path):
    path = tmp_path / "experts.gguf"
    raw = np.arange(2 * 3 * 2 * 144, dtype=np.uint8).reshape(2, 3, 288)
    write_checkpoint(path, [("blk.0.ffn_gate_exps.weight", raw, gguf.GGMLQuantizationType.Q4_K)])
    checkpoint = inspect_gguf_checkpoint(path)
    tensor = checkpoint.tensors["blk.0.ffn_gate_exps.weight"]
    assert tensor.shape == (2, 3, 512)
    assert tensor.storage_shape == (2, 3, 2, 144)
    assert tensor.encoding == "Q4_K" and tensor.dtype == "U8"
    assert tensor.component == "decoder_experts"
    assert tensor.nbytes == raw.nbytes
    with path.open("rb") as source:
        source.seek(tensor.offset)
        assert source.read(tensor.nbytes) == raw.tobytes()


@pytest.mark.parametrize("encoding,width,block_bytes", [("Q5_1", 640, 24), ("Q8_0", 640, 34),
                                                        ("IQ4_NL", 160, 18)])
def test_formats_keep_block_encoding_and_non_kquant_widths(tmp_path, encoding, width, block_bytes):
    path = tmp_path / "weight.gguf"
    raw = np.arange(8 * (width // 32) * block_bytes, dtype=np.uint8).reshape(8, -1)
    write_checkpoint(path, [("weight", raw, getattr(gguf.GGMLQuantizationType, encoding))])
    tensor = inspect_gguf_checkpoint(path).tensors["weight"]
    assert tensor.shape == (8, width)
    assert tensor.storage_shape == (8, width // 32, block_bytes)
    assert tensor.nbytes == raw.size
    assert tensor.encoding == encoding


def test_metadata_only_first_shard_and_starting_from_another_shard(tmp_path):
    write_checkpoint(tmp_path / "model.gguf", [
        ("a", np.arange(8, dtype=np.float32), None),
        ("b", np.arange(4, dtype=np.float16), None),
    ], split=True, metadata={"qwen4exp.attention.compress_ratios": [0, 0, 0, 4]})
    paths = sorted(tmp_path.glob("*.gguf"))
    assert len(paths) == 3
    checkpoint = inspect_gguf_checkpoint(paths[-1])
    assert checkpoint.shards[0].tensor_count == 0
    assert tuple(checkpoint.tensors) == ("a", "b")
    assert checkpoint.metadata["qwen4exp.attention.compress_ratios"] == (0, 0, 0, 4)
    assert checkpoint.tensors["a"].storage_shape == (8,)
    assert checkpoint.tensors["b"].dtype == "F16"
    assert checkpoint.index_path is None
    paths[1].unlink()
    with pytest.raises(CheckpointError, match="missing GGUF shards"):
        inspect_gguf_checkpoint(paths[0])


def test_draft_and_ple_memory_roles_follow_checkpoint_metadata(tmp_path):
    path = tmp_path / "model.gguf"
    names = ("blk.47.ffn_gate_exps.weight", "blk.48.ffn_gate_exps.weight",
             "blk.48.nextn.hnorm.weight", "per_layer_token_embd.weight")
    write_checkpoint(path, [(name, np.arange(8, dtype=np.float32), None) for name in names],
                     metadata={"qwen4exp.block_count": 49, "qwen4exp.nextn_predict_layers": 1})
    checkpoint = inspect_gguf_checkpoint(path)
    assert tuple(t.component for t in checkpoint.tensors.values()) == (
        "decoder_experts", "mtp", "mtp", "ple_tables")


def test_bf16_storage_keeps_original_bits(tmp_path):
    path = tmp_path / "bf16.gguf"
    raw = np.array([0x3f80, 0xc000, 0x3e80, 0x0000], dtype=np.uint16).view(np.uint8)
    write_checkpoint(path, [("weight", raw, gguf.GGMLQuantizationType.BF16)])
    tensor = inspect_gguf_checkpoint(path).tensors["weight"]
    assert tensor.shape == tensor.storage_shape == (4,)
    assert tensor.encoding == tensor.dtype == "BF16"
    with path.open("rb") as source:
        source.seek(tensor.offset)
        assert source.read(tensor.nbytes) == raw.tobytes()


def test_ppu_q2_0_reader_preserves_type_geometry_and_raw_bytes(tmp_path):
    qtype = getattr(gguf.GGMLQuantizationType, 'Q2_0', None)
    if qtype is None:
        pytest.skip('requires the PPU GGUF reader wheel')
    assert int(qtype) == 42 and gguf.GGML_QUANT_SIZES[qtype] == (64, 18)
    path = tmp_path / 'q2_0.gguf'
    raw = np.arange(2 * 3 * 10 * 18, dtype=np.uint8).reshape(2, 3, 180)
    name = 'blk.48.ffn_down_exps.weight'
    write_checkpoint(path, [(name, raw, qtype)])
    checkpoint = inspect_gguf_checkpoint(path)
    tensor = checkpoint.tensors[name]
    assert tensor.encoding == 'Q2_0' and tensor.dtype == 'U8'
    assert tensor.shape == (2, 3, 640) and tensor.storage_shape == (2, 3, 10, 18)
    assert tensor.nbytes == 1080
    with path.open('rb') as source:
        source.seek(tensor.offset)
        assert source.read(tensor.nbytes) == raw.tobytes()
    checkpoint.validate_files()


def test_duplicate_across_shards_rejected(tmp_path):
    write_checkpoint(tmp_path / "model.gguf", [
        ("a", np.arange(8, dtype=np.float32), None),
        ("b", np.arange(8, dtype=np.float32), None),
    ], split=True)
    paths = sorted(tmp_path.glob("*.gguf"))
    reader = gguf.GGUFReader(paths[-1], mode="r+")
    reader.tensors[0].field.parts[1][0] = ord("a")
    reader.data.flush()
    del reader
    with pytest.raises(CheckpointError, match="duplicate GGUF tensor across shards"):
        inspect_gguf_checkpoint(paths[0])


def test_overlapping_ranges_rejected(tmp_path):
    path = tmp_path / "overlap.gguf"
    write_checkpoint(path, [(name, np.arange(8, dtype=np.float32), None) for name in ("a", "b")])
    reader = gguf.GGUFReader(path, mode="r+")
    reader.tensors[1].field.parts[-1][0] = 0
    reader.data.flush()
    del reader
    with pytest.raises(CheckpointError, match="overlapping"):
        inspect_gguf_checkpoint(path)


def test_truncation_and_changed_source_rejected(tmp_path):
    path = tmp_path / "weight.gguf"
    write_checkpoint(path, [("weight", np.arange(32, dtype=np.float32), None)])
    checkpoint = inspect_gguf_checkpoint(path)
    checkpoint.validate_files()
    os.truncate(path, path.stat().st_size - 64)
    with pytest.raises(CheckpointError, match="changed after inspection"):
        checkpoint.validate_files()
    with pytest.raises(CheckpointError, match="truncated"):
        inspect_gguf_checkpoint(path)


def test_inconsistent_shard_count_rejected(tmp_path):
    write_checkpoint(tmp_path / "model.gguf", [
        ("a", np.arange(8, dtype=np.float32), None),
        ("b", np.arange(8, dtype=np.float32), None),
    ], split=True)
    paths = sorted(tmp_path.glob("*.gguf"))
    reader = gguf.GGUFReader(paths[-1], mode="r+")
    field = reader.get_field("split.tensors.count")
    field.parts[-1][0] += 1
    reader.data.flush()
    del reader
    with pytest.raises(CheckpointError, match="inconsistent split tensor count"):
        inspect_gguf_checkpoint(paths[0])


def test_big_endian_is_not_reinterpreted_as_cuda_blocks(tmp_path):
    path = tmp_path / "big.gguf"
    write_checkpoint(path, [("weight", np.arange(8, dtype=np.float32), None)], endian=gguf.GGUFEndian.BIG)
    with pytest.raises(CheckpointError, match="little-endian"):
        inspect_gguf_checkpoint(path)


@pytest.mark.parametrize("payload", [b"", b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 1, 0)])
def test_invalid_header(tmp_path, payload):
    path = tmp_path / "bad.gguf"
    path.write_bytes(payload)
    with pytest.raises(CheckpointError):
        inspect_gguf_checkpoint(path)
