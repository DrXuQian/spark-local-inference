# SPDX-License-Identifier: Apache-2.0

"""Resolve box inputs, then export the existing partial GGUF/MTP fixture on PPU.

Only target blocks 44/47 and shared GGUF MTP weights are loaded. Frontend downloads
are explicit and restricted to four pinned text assets; no weights are fetched.
Select the platform SDK in the shell before starting this command.
"""

import argparse
from collections import Counter
import hashlib
import importlib.machinery
import importlib.util
from math import prod
from pathlib import Path
import shlex
import subprocess
import sys

from ignition.preparation.qwen38 import CONFIG_SHA256, FRONTEND_SHA256, MODEL_ID, REVISION


TARGET_STEM = 'Qwen3.8-Flash-Next-UD-Q4_K_XL'
MTP_NAME = 'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
TEXT_ASSETS = {'config.json': CONFIG_SHA256,
               **{name: FRONTEND_SHA256[name] for name in
                  ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}}


def model_files(directory, mtp=None):
    directory = Path(directory).resolve(strict=True)
    target = directory / f'{TARGET_STEM}-00004-of-00004.gguf'
    metadata = directory / f'{TARGET_STEM}-00001-of-00004.gguf'
    missing = [str(path) for path in (target, metadata) if not path.is_file()]
    if mtp is None:
        candidates = tuple(root / MTP_NAME for root in
                           (directory, directory / 'MTP', directory.parent, directory.parent / 'MTP'))
        found = {path.resolve() for path in candidates if path.is_file()}
        if len(found) > 1:
            raise ValueError('multiple shared-Q8 MTP files found; select one with --mtp: '
                             + ', '.join(map(str, sorted(found))))
        if not found:
            missing.append('shared-Q8 MTP; supply --mtp PATH (searched '
                           + ', '.join(map(str, candidates)) + ')')
        else:
            mtp = found.pop()
    else:
        mtp = Path(mtp).resolve(strict=True)
        if not mtp.is_file():
            missing.append(str(mtp))
    if missing:
        raise ValueError('missing qualification inputs:\n  ' + '\n  '.join(missing))
    return target, metadata, mtp


def frontend_files(directory, *, fetch=False):
    directory = Path(directory).resolve()
    if fetch:
        from huggingface_hub import hf_hub_download

        for name in TEXT_ASSETS:
            hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=directory)
    for name, expected in TEXT_ASSETS.items():
        path = directory / name
        with path.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{path}: does not match the pinned Flare frontend')
    return directory


def inspect_mtp(path):
    """Report stored types and layer names from headers, without decoding weights."""
    import gguf

    class HeaderReader(gguf.GGUFReader):
        def _build_tensors(self, start_offs, fields):
            # The pinned reader parses raw type IDs before constructing payload
            # views. Diagnostics need those fields even for an unknown encoding.
            self.tensor_fields = fields

    reader = HeaderReader(path, mode='r')
    print(f'MTP file: {path} ({path.stat().st_size} bytes)', flush=True)
    for key in ('general.architecture', 'general.file_type', 'qwen4exp.block_count',
                'qwen4exp.nextn_predict_layers', 'qwen4exp.nextn_shared_target_tensors'):
        field = reader.get_field(key)
        print(f'  {key}: {None if field is None else field.contents()}', flush=True)
    counts, sizes, headers, unreadable, names = Counter(), Counter(), [], set(), set()
    for field in reader.tensor_fields:
        shape = tuple(int(n) for n in reversed(field.parts[3]))
        type_id, offset = int(field.parts[4][0]), reader.data_offset + int(field.parts[5][0])
        if field.name in names or not shape or len(shape) > 4 or min(shape) <= 0:
            raise ValueError(f'{field.name}: invalid or duplicate GGUF tensor header')
        names.add(field.name)
        try:
            qtype = gguf.GGMLQuantizationType(type_id)
            encoding = qtype.name
            layout = gguf.GGML_QUANT_SIZES[qtype]
        except (ValueError, KeyError):
            unreadable.add(type_id)
            encoding, layout = f'UNKNOWN_{type_id}', None
            if type_id == 42:
                # DrXuQian/llama.cpp 356a231c1: ggml-common.h block_q2_0
                # and gguf-py/constants.py define 64 values in 18 bytes.
                # This labels header storage only; it does not add a decoder.
                encoding, layout = 'Q2_0', (64, 18)
        nbytes = None
        if layout is not None:
            block, width = layout
            if shape[-1] % block:
                raise ValueError(f'{field.name}: incomplete {encoding} blocks in a row')
            nbytes = prod(shape) // block * width
            sizes[encoding] += nbytes
        if offset < reader.data_offset or offset >= reader.data.size or offset % reader.alignment:
            raise ValueError(f'{field.name}: invalid GGUF tensor offset')
        counts[encoding] += 1
        headers.append((field.name, encoding, shape, type_id, offset, nbytes))
    ordered = sorted(headers, key=lambda header: header[4])
    for index, (name, _, _, _, offset, nbytes) in enumerate(ordered):
        end = ordered[index + 1][4] if index + 1 < len(ordered) else reader.data.size
        if end <= offset or (nbytes is not None and offset + nbytes > end):
            raise ValueError(f'{name}: overlapping or truncated GGUF tensor storage')
    for encoding in sorted(counts):
        size = f'{sizes[encoding]} stored bytes' if encoding in sizes else 'stored bytes unknown'
        print(f'  {encoding}: {counts[encoding]} tensors, {size}', flush=True)
    print(f'  total tensors: {len(headers)}', flush=True)
    for name, encoding, shape, type_id, offset, _ in headers[:64]:
        print(f'  {name}: {encoding} {shape}, type_id={type_id}, data_offset={offset}', flush=True)
    if len(headers) > 64:
        print(f'  ... {len(headers) - 64} additional tensor names omitted', flush=True)
    if unreadable:
        print(f'  installed gguf cannot load type IDs: {sorted(unreadable)}; header inspection only', flush=True)
    if 42 in unreadable:
        print('  type_id=42 follows the llama.cpp PPU Q2_0 extension: 64 values/18 bytes; '
              'Q2_K is a different format (type_id=10, 256 values/84 bytes).', flush=True)
    return frozenset(unreadable)


def validate_mtp(path):
    """Reject incompatible shared-MTP storage before allocating target weights."""
    from ignition.preparation.gguf import inspect_gguf_checkpoint

    checkpoint = inspect_gguf_checkpoint(path)
    expected = {'general.architecture': 'qwen4exp', 'qwen4exp.block_count': 49,
                'qwen4exp.nextn_predict_layers': 1, 'qwen4exp.nextn_shared_target_tensors': True}
    for name, value in expected.items():
        actual = checkpoint.metadata.get(name)
        if actual != value or type(actual) is not type(value):
            raise ValueError(f'MTP {name}={actual!r}; the current shared-MTP fixture requires {value!r}')
    if not checkpoint.tensors or any(not name.startswith('blk.48.') for name in checkpoint.tensors):
        raise ValueError('the current shared-MTP fixture requires only blk.48.* tensors')
    ratios = checkpoint.metadata.get('qwen4exp.attention.compress_ratios')
    if not isinstance(ratios, tuple) or len(ratios) != 49 or ratios[48] != 0:
        raise ValueError('MTP requires an uncompressed layer 48 in attention.compress_ratios')
    for part, shape in (('gate', (512, 640, 2560)), ('up', (512, 640, 2560)),
                        ('down', (512, 2560, 640))):
        name = f'blk.48.ffn_{part}_exps.weight'
        tensor = checkpoint.tensors.get(name)
        if tensor is None or tensor.shape != shape:
            raise ValueError(f'MTP {name} requires shape {shape}')
        if tensor.dtype != 'U8':
            raise ValueError(f'MTP {name} uses {tensor.encoding}; the current GGUF expert path requires quantized blocks')
    # Full matrix/type admission still belongs to the selected operator library.
    gate, up = (checkpoint.tensors[f'blk.48.ffn_{part}_exps.weight'] for part in ('gate', 'up'))
    if gate.encoding != up.encoding:
        raise ValueError('MTP gate/up experts require equal encodings for the existing fused load')


def installed_attention():
    # Locate the extension without importing it before the module recorder starts.
    package = importlib.util.find_spec('flash_attn_3')
    extension = None if package is None or not package.submodule_search_locations else \
        importlib.machinery.PathFinder.find_spec('flash_attn_3._C', package.submodule_search_locations)
    if extension is None or extension.origin is None:
        raise ValueError('installed flash_attn_3._C extension is missing')
    return Path(extension.origin).resolve(strict=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True, help='existing UD-Q4_K_XL directory')
    parser.add_argument('--mtp', type=Path, help='explicit shared MTP GGUF; types are read from its header')
    parser.add_argument('--inspect-only', action='store_true', help='report MTP headers without downloading, patching or loading operators')
    frontend = parser.add_mutually_exclusive_group()
    frontend.add_argument('--frontend', type=Path, help='existing pinned config/tokenizer/template directory')
    frontend.add_argument('--fetch-frontend', action='store_true', help='download only four pinned text assets')
    parser.add_argument('--operator-target', choices=('ppu10', 'ppu15'))
    parser.add_argument('--blas-library', type=Path, help='selected SDK libcublas.so')
    parser.add_argument('--output', type=Path, help='new package work directory on a large disk')
    args = parser.parse_args(argv)
    if not args.inspect_only:
        missing = [f'--{name.replace("_", "-")}' for name in ('operator_target', 'blas_library', 'output')
                   if getattr(args, name) is None]
        if args.frontend is None and not args.fetch_frontend:
            missing.append('--frontend or --fetch-frontend')
        if missing:
            parser.error('export requires ' + ', '.join(missing))
    try:
        target, metadata, mtp = model_files(args.model_dir, args.mtp)
        unreadable = inspect_mtp(mtp)
        if args.inspect_only:
            print('MTP headers inspected; operator compatibility and device execution are not qualified.', flush=True)
            return 0
        if unreadable:
            raise ValueError(f'export requires a GGUF reader and operator implementation for type IDs '
                             f'{sorted(unreadable)}; header inspection does not enable execution')
        validate_mtp(mtp)
        blas = args.blas_library.resolve(strict=True)
        if not blas.is_file():
            raise ValueError(f'not a BLAS library file: {blas}')
        attention = installed_attention()
        output = args.output.resolve()
        if (output / 'package').exists():
            raise ValueError(f'{output / "package"} already exists; select a new --output')
        frontend = frontend_files(args.frontend or output / 'frontend', fetch=args.fetch_frontend)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for name, path in (('target blocks 44/47', target), ('metadata', metadata), ('shared GGUF MTP', mtp),
                       ('frontend', frontend), ('BLAS', blas), ('FA3', attention)):
        print(f'{name}: {path}', flush=True)

    from ignition.patches import apply

    apply(operator_target=args.operator_target)
    command = [sys.executable, '-X', 'faulthandler', '-u', '-m', 'qualify.qwen38_gguf',
               '--target-shard', str(target), '--metadata', str(metadata), '--mtp', str(mtp),
               '--frontend', str(frontend), '--output', str(output), '--layers', '44', '47',
               '--operator-target', args.operator_target, '--module-resource-abi', 'hgpti-bare',
               '--actlize-qsa', '--dense-blas-library', str(blas), '--dense-blas-api', 'cublas',
               '--flash-attention3-library', str(attention)]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode:
        return result.returncode if result.returncode > 0 else 128 - result.returncode
    print(f'Partial GGUF/MTP export process exited successfully: {output / "package"}\n'
          'Native request/replay checks and complete-round timing remain to be run.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
