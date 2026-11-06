# SPDX-License-Identifier: Apache-2.0

"""Sparse files matching Strata's independently recorded 31-tensor inventory.

The fixture names/shapes come from Strata's 2026-10-07 RTX5080 MTP manifest;
encodings follow tools/mtp_pack.py at 6780211b. No model weights are downloaded.
"""

import json
from pathlib import Path

import gguf
import numpy as np


def inventory():
    return json.loads((Path(__file__).parents[1] / 'fixtures/qwen38_hf_mtp_gguf.json').read_text())


def write_hf_mtp(path, *, entries=None, declared='q2_0', values=None):
    entries = inventory() if entries is None else entries
    writer = gguf.GGUFWriter(path, 'qwen4exp-mtp')
    writer.add_string('strata.mtp.expert_format', declared)
    offsets, payload = {}, 0
    for entry in entries:
        name, shape = entry['name'], tuple(entry['shape'])
        kind = gguf.GGMLQuantizationType[entry['encoding']]
        block, size = gguf.GGML_QUANT_SIZES[kind]
        nbytes = int(np.prod(shape)) // block * size
        raw_shape = (*shape[:-1], shape[-1] // block * size) if block > 1 else shape
        dtype = np.uint8 if block > 1 else np.float32 if kind.name == 'F32' else np.uint16
        writer.add_tensor_info(name, raw_shape, dtype, nbytes, raw_dtype=kind)
        offsets[name] = payload
        payload += gguf.GGUFWriter.ggml_pad(nbytes, writer.data_alignment)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    output = writer.fout[0]
    writer.write_padding(output, output.tell())
    start = output.tell()
    output.truncate(start + payload)
    for name, data in (values or {}).items():
        output.seek(start + offsets[name])
        output.write(data.tobytes())
    writer.close()
    return path
