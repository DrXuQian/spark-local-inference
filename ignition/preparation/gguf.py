# SPDX-License-Identifier: Apache-2.0

"""Inspect GGUF shards without expanding or reading their weight payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from math import prod
from pathlib import Path
import re
from types import MappingProxyType

from .checkpoint import CheckpointError, IndexedCheckpoint, ShardMetadata, TensorMetadata


_SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z")
_PLAIN = {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}
_VOCAB_FIELDS = {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
                 "tokenizer.ggml.merges"}


def file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True, slots=True)
class GgufTensorMetadata(TensorMetadata):
    """Logical shape is in PyTorch order; storage_shape describes unchanged bytes."""

    encoding: str
    storage_shape: tuple[int, ...]
    role: str

    @property
    def component(self) -> str:
        return self.role


@dataclass(frozen=True, slots=True)
class GgufShardMetadata(ShardMetadata):
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GgufCheckpoint(IndexedCheckpoint):
    metadata: Mapping[str, object]
    vocab_size: int | None

    @property
    def index_path(self) -> None:
        return None

    def validate_files(self) -> None:
        for shard in self.shards:
            if file_identity(self.snapshot / shard.path) != shard.identity:
                raise CheckpointError(f"{shard.path}: GGUF changed after inspection")


def _reader(path):
    from gguf import GGUFEndian, GGUFReader

    try:
        reader = GGUFReader(path, mode="r")
    except (ValueError, KeyError, IndexError, OverflowError) as error:
        raise CheckpointError(f"{path.name}: invalid or truncated GGUF: {error}") from error
    if reader.endianess != GGUFEndian.LITTLE:
        raise CheckpointError(f"{path.name}: CUDA GGUF loading requires little-endian blocks")
    return reader


def _value(reader, key, default=None):
    field = reader.get_field(key)
    return default if field is None else field.contents()


def _split_paths(path, reader):
    count = _value(reader, "split.count", 1)
    number = _value(reader, "split.no", 0)
    if type(count) is not int or not 1 <= count <= 99999 or type(number) is not int or not 0 <= number < count:
        raise CheckpointError(f"{path.name}: invalid GGUF split count or number")
    if count == 1:
        return (path,)
    match = _SPLIT.fullmatch(path.name)
    if match is None or int(match[2]) != number + 1 or int(match[3]) != count:
        raise CheckpointError(f"{path.name}: GGUF split filename disagrees with metadata")
    return tuple(path.with_name(f"{match[1]}-{index + 1:05d}-of-{count:05d}.gguf") for index in range(count))


def _component(name, metadata):
    if name.startswith('mtp.'):
        return 'mtp'
    architecture = metadata.get("general.architecture")
    layers = metadata.get(f"{architecture}.block_count")
    drafts = metadata.get(f"{architecture}.nextn_predict_layers", 0)
    layer = re.match(r"blk\.(\d+)\.", name)
    if (layer is not None and type(layers) is int and type(drafts) is int
            and drafts > 0 and layers - drafts <= int(layer[1]) < layers):
        return "mtp"
    if name == "per_layer_token_embd.weight":
        return "ple_tables"
    if "_exps." in name:
        return "decoder_experts"
    return "decoder_other"


def inspect_gguf_checkpoint(path: Path) -> GgufCheckpoint:
    """Inspect a file or any shard, including a metadata-only first shard.

    The external ``gguf`` reader supplies parsing and block layouts. This layer
    checks shard completeness, logical row boundaries, and nonoverlapping file
    ranges before CUDA allocation. Header validation does not verify weight values.
    Vocabulary payloads stay in the source file; only their count is retained here.
    """
    from gguf import GGML_QUANT_SIZES

    path = Path(path).resolve(strict=True)
    first_identity = file_identity(path)
    first = _reader(path)
    paths = _split_paths(path, first)
    missing = [item.name for item in paths if not item.is_file()]
    if missing:
        raise CheckpointError(f"missing GGUF shards: {', '.join(missing)}")
    tensors, metadata, shards = {}, {}, []
    vocab_size = total_tensors = None
    for index, item in enumerate(paths):
        identity = first_identity if item == path else file_identity(item)
        reader = first if item == path else _reader(item)
        if not 0 < reader.data_offset <= identity[2]:
            raise CheckpointError(f"{item.name}: truncated GGUF header padding")
        if _value(reader, "split.count", 1) != len(paths) or _value(reader, "split.no", 0) != index:
            raise CheckpointError(f"{item.name}: inconsistent GGUF split metadata")
        declared_total = _value(reader, "split.tensors.count")
        if len(paths) > 1 or declared_total is not None:
            if type(declared_total) is not int or declared_total < 0:
                raise CheckpointError(f"{item.name}: invalid split tensor count")
            if total_tensors is not None and declared_total != total_tensors:
                raise CheckpointError(f"{item.name}: inconsistent split tensor count")
            total_tensors = declared_total
        vocab = reader.get_field("tokenizer.ggml.tokens")
        if vocab is not None:
            count = len(vocab.data)
            if vocab_size is not None and count != vocab_size:
                raise CheckpointError(f"{item.name}: inconsistent vocabulary size")
            vocab_size = count
        for key, field in reader.fields.items():
            if key.startswith(("GGUF.", "split.")) or key in _VOCAB_FIELDS:
                continue
            value = field.contents()
            value = tuple(value) if isinstance(value, list) else value
            if key in metadata and value != metadata[key]:
                raise CheckpointError(f"{item.name}: inconsistent GGUF metadata {key}")
            metadata[key] = value

        ranges = []
        for tensor in reader.tensors:
            name, encoding = tensor.name, tensor.tensor_type.name
            shape = tuple(int(d) for d in reversed(tensor.shape))
            block, block_bytes = GGML_QUANT_SIZES[tensor.tensor_type]
            if not shape or len(shape) > 4 or min(shape) <= 0 or shape[-1] % block:
                raise CheckpointError(f"{name}: GGUF rows must contain complete {encoding} blocks")
            nbytes = prod(shape) // block * block_bytes
            offset = int(tensor.data_offset)
            if (nbytes != tensor.n_bytes or tensor.data.nbytes != nbytes
                    or offset < reader.data_offset or offset % reader.alignment
                    or offset + nbytes > identity[2]):
                raise CheckpointError(f"{name}: invalid GGUF tensor byte range")
            if name in tensors:
                raise CheckpointError(f"duplicate GGUF tensor across shards: {name}")
            plain = encoding in _PLAIN
            storage_shape = shape if plain else (*shape[:-1], shape[-1] // block, block_bytes)
            tensors[name] = GgufTensorMetadata(
                name, item.name, offset, nbytes, shape, encoding if plain else "U8",
                encoding, storage_shape, _component(name, metadata))
            ranges.append((offset, nbytes, name))
        end = int(reader.data_offset)
        for offset, nbytes, name in sorted(ranges):
            if offset < end:
                raise CheckpointError(f"{name}: overlapping GGUF tensor ranges")
            end = offset + nbytes
        if identity != file_identity(item):
            raise CheckpointError(f"{item.name}: GGUF changed during inspection")
        digest = hashlib.sha256()
        for start in range(0, reader.data_offset, 1 << 20):
            digest.update(reader.data[start:min(start + (1 << 20), reader.data_offset)])
        shards.append(GgufShardMetadata(item.name, identity[2], int(reader.data_offset),
                                        digest.hexdigest(), len(ranges), identity))
    if total_tensors is not None and len(tensors) != total_tensors:
        raise CheckpointError(f"GGUF tensor count {len(tensors)} differs from declared {total_tensors}")
    return GgufCheckpoint(path.parent, MappingProxyType(tensors), tuple(shards),
                          MappingProxyType(metadata), vocab_size)
