# SPDX-License-Identifier: Apache-2.0

"""Explicit main/draft/frontend inputs for the Qwen GGUF preparation recipe."""

from dataclasses import dataclass
import hashlib
import json
from math import prod
from pathlib import Path
from types import MappingProxyType

from .checkpoint import CheckpointError
from .actlize import ActlizeLibrarySpec
from .quactlize import QuactlizeHelpersSpec, QuactlizeLibrarySpec
from .blas import BlasLibrarySpec
from .gguf import GgufCheckpoint, file_identity, inspect_gguf_checkpoint
from .qwen38 import CONFIG_SHA256, FRONTEND_SHA256, GENERATION_SHA256


@dataclass(frozen=True)
class GgufInputs:
    main: Path
    mtp: Path
    frontend: Path
    read_mode: str
    actlize: ActlizeLibrarySpec | None = None
    quactlize: QuactlizeLibrarySpec | None = None
    dense_blas: BlasLibrarySpec | None = None
    quactlize_helpers: QuactlizeHelpersSpec | None = None

    def operator_identity(self):
        return operator_identity(self.actlize, self.quactlize, self.dense_blas, self.quactlize_helpers)

    def load_operators(self, device):
        attention = None if self.actlize is None else self.actlize.load(device)
        projection = None if self.quactlize is None else self.quactlize.load(device)
        return dict(actlize_library=attention, actlize_qsa_library=attention,
                    gguf_projection_library=projection,
                    dense_blas=None if self.dense_blas is None else self.dense_blas.load(),
                    quactlize_helpers=None if self.quactlize_helpers is None else self.quactlize_helpers.load(device))

    @classmethod
    def read(cls, directory):
        directory = Path(directory).resolve()
        value = json.loads((directory / 'manifest.json').read_text())
        required = {'schema', 'main', 'mtp', 'frontend', 'read_mode'}
        optional = {'actlize_library', 'quactlize_abi_library', 'dense_blas', 'quactlize_helpers_library'}
        if (not isinstance(value, dict) or not required <= value.keys() <= required | optional
                or value['schema'] != 'qwen38-gguf/1'):
            raise CheckpointError('GGUF input manifest requires schema, main, mtp, frontend and read_mode;'
                                  ' actlize_library, quactlize_abi_library, quactlize_helpers_library and dense_blas are optional')
        if value['read_mode'] not in ('auto', 'bounce', 'gds', 'pread'):
            raise CheckpointError('unknown GGUF checkpoint transport')
        paths = []
        for key in ('main', 'mtp', 'frontend'):
            if not isinstance(value[key], str) or not value[key]:
                raise CheckpointError(f'GGUF {key} requires a path')
            paths.append((directory / value[key]).resolve(strict=True))
        if not paths[0].is_file() or not paths[1].is_file() or not paths[2].is_dir():
            raise CheckpointError('GGUF input requires two files and a frontend directory')
        libraries = []
        for key, spec in (('actlize_library', ActlizeLibrarySpec), ('quactlize_abi_library', QuactlizeLibrarySpec),
                          ('quactlize_helpers_library', QuactlizeHelpersSpec)):
            if key in value and (not isinstance(value[key], str) or not value[key]):
                raise CheckpointError(f'{key} requires a native library path')
            libraries.append(None if key not in value else spec.read(directory / value[key]))
        try:
            blas = BlasLibrarySpec.read(directory, value['dense_blas']) if 'dense_blas' in value else None
        except ValueError as exc:
            raise CheckpointError(str(exc)) from exc
        return cls(*paths, value['read_mode'], libraries[0], libraries[1], blas, libraries[2])


def operator_identity(actlize, quactlize, dense_blas=None, quactlize_helpers=None):
    value = None if actlize is None else actlize.identity()
    if value is None and any(spec is not None for spec in (quactlize, dense_blas, quactlize_helpers)):
        value = dict(gdn='b12x', qsa='b12x', kv_dtype='float8_e4m3fn')
    if quactlize is not None:
        value['matmul'] = quactlize.identity()
    if dense_blas is not None:
        value['dense_blas'] = dense_blas.identity()
    if quactlize_helpers is not None:
        value['quactlize_helpers'] = quactlize_helpers.identity()
    return value


@dataclass(frozen=True, slots=True)
class QwenGgufCheckpoint(GgufCheckpoint):
    draft: GgufCheckpoint
    frontend: Path
    metadata_sha256: object
    headers_sha256: str
    read_mode: str
    actlize: ActlizeLibrarySpec | None = None
    quactlize: QuactlizeLibrarySpec | None = None
    dense_blas: BlasLibrarySpec | None = None
    quactlize_helpers: QuactlizeHelpersSpec | None = None

    @property
    def metadata_path(self):
        return self.frontend


def validate_frontend_token_ids(path, tokenizer):
    """Match active IDs and BPE merges; GGUF-only tail rows must be padding."""
    import gguf

    before = file_identity(path)
    reader = gguf.GGUFReader(path)
    field = reader.get_field('tokenizer.ggml.tokens')
    if field is None:
        raise CheckpointError('GGUF metadata shard has no vocabulary')
    tokens = field.contents()
    frontend = json.loads(Path(tokenizer).read_bytes())
    vocabulary = dict(frontend['model']['vocab'])
    for added in frontend.get('added_tokens', ()):
        if added['content'] in vocabulary and vocabulary[added['content']] != added['id']:
            raise CheckpointError('frontend added token conflicts with base vocabulary')
        vocabulary[added['content']] = added['id']
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise CheckpointError('frontend token IDs must form a contiguous vocabulary')
    for token, index in vocabulary.items():
        if index >= len(tokens) or tokens[index] != token:
            raise CheckpointError(f'GGUF token ID {index} differs from the prepared frontend')
    for index in range(len(vocabulary), len(tokens)):
        if tokens[index] != f'[PAD{index}]':
            raise CheckpointError(f'GGUF token ID {index} is not an unused padding row')
    expected = [' '.join(pair) if isinstance(pair, list) else pair
                for pair in frontend['model'].get('merges', ())]
    field = reader.get_field('tokenizer.ggml.merges')
    actual = [] if field is None else field.contents()
    if actual != expected:
        raise CheckpointError('GGUF BPE merges differ from the prepared frontend')
    if file_identity(path) != before:
        raise CheckpointError('GGUF vocabulary changed during frontend validation')


def inspect_qwen38_gguf(inputs: GgufInputs):
    main, draft = inspect_gguf_checkpoint(inputs.main), inspect_gguf_checkpoint(inputs.mtp)
    expected = {
        'embedding_length': 2560, 'context_length': 262144, 'attention.head_count': 24,
        'attention.head_count_kv': 2, 'attention.key_length': 256, 'attention.value_length': 256,
        'expert_count': 512, 'expert_used_count': 10, 'expert_feed_forward_length': 640,
        'expert_shared_feed_forward_length': 640, 'ssm.conv_kernel': 4, 'ssm.state_size': 128,
        'ssm.group_count': 16, 'ssm.time_step_rank': 48, 'ssm.inner_size': 6144,
        'full_attention_interval': 4, 'hyper_connection.count': 4, 'hyper_connection.low_rank': 320,
        'rope.dimension_count': 64, 'rope.dimension_sections': (11, 11, 10, 0), 'rope.freq_base': 10000000.,
        'attention.indexer.head_count': 4, 'attention.indexer.key_length': 128, 'attention.indexer.top_k': 2048,
    }
    for checkpoint, layers in ((main, 48), (draft, 49)):
        if checkpoint.metadata.get('general.architecture') != 'qwen4exp':
            raise CheckpointError('GGUF main and shared MTP must use qwen4exp')
        for name, value in {**expected, 'block_count': layers}.items():
            if checkpoint.metadata.get('qwen4exp.' + name) != value:
                raise CheckpointError(f'GGUF {name} differs from the Qwen recipe geometry')
        if abs(checkpoint.metadata.get('qwen4exp.attention.layer_norm_rms_epsilon', 0.) - 1e-6) > 1e-13:
            raise CheckpointError('GGUF RMS epsilon differs from the Qwen recipe')
        ratios = tuple(4 if i % 4 == 3 else 0 for i in range(layers))
        if checkpoint.metadata.get('qwen4exp.attention.compress_ratios') != ratios:
            raise CheckpointError('GGUF attention compression ratios differ from the prepared layer layout')
    if (draft.metadata.get('qwen4exp.nextn_shared_target_tensors') is not True
            or draft.metadata.get('qwen4exp.nextn_predict_layers') != 1
            or any(not name.startswith('blk.48.') for name in draft.tensors)):
        raise CheckpointError('GGUF draft must share the target embedding/head and contain only layer 48')
    if (main.vocab_size != 248320 or main.metadata.get('qwen4exp.ple.layers') != (1,)
            or main.metadata.get('qwen4exp.ple.ngram_size') != 3
            or main.metadata.get('qwen4exp.ple.heads_per_ngram') != 8
            or main.metadata.get('qwen4exp.ple.conv_kernel') != 4
            or main.metadata.get('qwen4exp.ple.eos_token_id') != 248044
            or main.metadata.get('qwen4exp.embedding_length_per_layer_input') != 160):
        raise CheckpointError('GGUF vocabulary or PLE geometry differs from the Qwen recipe')
    hashes = dict(FRONTEND_SHA256, **{'config.json': CONFIG_SHA256, 'generation_config.json': GENERATION_SHA256})
    for name, expected_hash in hashes.items():
        if hashlib.sha256((inputs.frontend / name).read_bytes()).hexdigest() != expected_hash:
            raise CheckpointError(f'{name}: Qwen GGUF preparation requires the pinned canonical frontend assets')
    validate_frontend_token_ids(main.snapshot / main.shards[0].path, inputs.frontend / 'tokenizer.json')
    headers = hashlib.sha256(''.join(shard.header_sha256 for checkpoint in (main, draft)
                                     for shard in checkpoint.shards).encode()).hexdigest()
    return QwenGgufCheckpoint(main.snapshot, main.tensors, main.shards, main.metadata, main.vocab_size,
                              draft, inputs.frontend, MappingProxyType(hashes), headers, inputs.read_mode,
                              inputs.actlize, inputs.quactlize, inputs.dense_blas, inputs.quactlize_helpers)


def device_weight_bytes(checkpoint):
    """Resident packed experts/embedding and the once-expanded dense matrices."""
    return sum(tensor.nbytes if '_exps.' in name or name == 'token_embd.weight' or tensor.dtype != 'U8'
               else prod(tensor.shape) * 2
               for name, tensor in checkpoint.tensors.items() if tensor.component != 'ple_tables')
