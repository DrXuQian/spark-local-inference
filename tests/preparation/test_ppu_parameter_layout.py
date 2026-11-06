# SPDX-License-Identifier: Apache-2.0

"""Compiled PPU launch ABI metadata; no driver or device is required."""

import os
import struct
import subprocess
from pathlib import Path

import pytest
from flare_heliopause import ppu_parameter_sizes

from ignition.preparation.module_images import strip_debug


def entry(tag, payload):
    return struct.pack('<HH', tag, len(payload)) + payload


def argument(index, size, dtype=8, kind=3, location=6):
    return entry(0x102, struct.pack('<IIIBBBB', location, size, index, dtype, kind, 0, 0))


HEADER = entry(0x101, struct.pack('<II', 0, 1))
HIDDEN = argument(0xffff, 2, dtype=12, kind=14, location=0)


def image(metadata, *, machine=270, abi=4, duplicate=False):
    names = b'\0.shstrtab\0.strtab\0.symtab\0.text.kernel.probe\0.hg_info.probe\0'
    strings = b'\0probe\0'
    symbols = bytes(24) + struct.pack('<IBBHQQ', 1, 0x12, 0x20, 4, 0, 8)
    sections = [(b'', 0, 0, b'', 0, 0), (b'.shstrtab', 3, 0, names, 0, 0),
                (b'.strtab', 3, 0, strings, 0, 0), (b'.symtab', 2, 0, symbols, 2, 24),
                (b'.text.kernel.probe', 1, 6, bytes(8), 0, 0),
                (b'.hg_info.probe', 1, 0, metadata, 0, 0)]
    if duplicate:
        sections.append(sections[-1])
    data = bytearray(64 + 64 * len(sections))
    data[:9] = b'\x7fELF\x02\x01\x01' + bytes([96, abi])
    struct.pack_into('<HHI', data, 16, 2, machine, 0x221)
    struct.pack_into('<Q', data, 40, 64)
    struct.pack_into('<HHHHHH', data, 52, 64, 56, 0, 64, len(sections), 1)
    for i, (name, kind, flags, payload, link, stride) in enumerate(sections):
        struct.pack_into('<IIQQQQIIQQ', data, 64 + 64 * i,
                         names.index(name) if name else 0, kind, flags, 0,
                         len(data), len(payload), link, 0, 1, stride)
        data.extend(payload)
    return bytes(data)


def test_ppu_pointer_width_is_independent_of_pointee_and_device_location():
    # Actual HGCC uses dtype=12 for a uint64_t*; Triton uses dtype=4 for void*.
    data = HEADER + HIDDEN + argument(1, 1) + argument(0, 2, dtype=4, kind=2, location=14)
    assert ppu_parameter_sizes(image(data), 'probe') == [8, 4]
    assert ppu_parameter_sizes(strip_debug(image(data)), 'probe') == [8, 4]


def test_ppu_by_value_objects_spilled_scalars_and_hidden_parameters():
    data = HEADER + HIDDEN + argument(0, 3, dtype=4, kind=1, location=0)
    data += argument(1, 24, dtype=4, kind=1, location=8)
    data += argument(2, 8, dtype=12, kind=1, location=32)
    data += argument(3, 2, dtype=14)
    assert ppu_parameter_sizes(image(data), 'probe') == [3, 24, 8, 8]
    assert ppu_parameter_sizes(image(HEADER + HIDDEN), 'probe') == []
    assert ppu_parameter_sizes(image(data, machine=190), 'probe') is None


@pytest.mark.parametrize('metadata,error', [
    (b'', 'lacks its function header'),
    (HEADER + HEADER, 'function metadata header'),
    (HEADER + argument(0, 1)[:-1], 'truncated'),
    (HEADER + b'\x02\x01', 'extent'),
    (HEADER + entry(0x102, bytes(15)), 'argument record size'),
    (HEADER + entry(0x202, bytes(16)), 'record version'),
    (HEADER + argument(1, 1), 'not contiguous'),
    (HEADER + argument(0, 1) * 2, 'duplicate'),
    (HEADER + argument(4096, 1), 'ordinal'),
    (HEADER + argument(0, 0), 'extent'),
    (HEADER + argument(0xffff, 2, dtype=12, kind=2), 'ordinal'),
    (HEADER + argument(0, 1, dtype=12, kind=2), 'delivery kind'),
    (HEADER + argument(0, 1, dtype=4), 'register argument type'),
    (HEADER + argument(0, 2, dtype=8), 'register argument type'),
    (HEADER + argument(0, 1, kind=19), 'delivery kind'),
])
def test_invalid_or_unqualified_parameter_metadata_fails(metadata, error):
    with pytest.raises(ValueError, match=error):
        ppu_parameter_sizes(image(metadata), 'probe')


def test_missing_duplicate_or_invalid_elf_metadata_is_not_a_parameterless_kernel():
    with pytest.raises(ValueError, match='one defined kernel symbol'):
        ppu_parameter_sizes(image(HEADER), 'absent')
    with pytest.raises(ValueError, match='duplicate'):
        ppu_parameter_sizes(image(HEADER, duplicate=True), 'probe')
    with pytest.raises(ValueError, match='metadata ABI'):
        ppu_parameter_sizes(image(HEADER, abi=5), 'probe')
    data = bytearray(image(HEADER))
    struct.pack_into('<Q', data, 64 + 5 * 64 + 32, len(data))
    with pytest.raises(ValueError, match='section extent'):
        ppu_parameter_sizes(bytes(data), 'probe')


@pytest.mark.parametrize('target', ['ppu_10', 'ppu_15'])
def test_hgcc_parameter_sizes_and_debug_stripping(tmp_path, target):
    compiler = os.environ.get('FLARE_TEST_HGCC')
    if not compiler:
        pytest.skip('set FLARE_TEST_HGCC to the SDK hgcc for offline compiler qualification')
    source = Path(__file__).parent / 'fixtures/ppu_parameter_layout.cu'
    output = tmp_path / 'parameters.hgbin'
    subprocess.run([compiler, '--hgbin', '--gpu-architecture=' + target, '-O2', str(source),
                    '-o', str(output)], check=True, capture_output=True)
    for data in (output.read_bytes(), strip_debug(output.read_bytes())):
        for symbol, sizes in {'empty': [], 'structs': [3, 24, 16, 4],
                              'unused': [8, 4, 3, 8], 'spilled': [8] * 24,
                              'words': [4, 4, 8, 8], 'pointers': [8] * 24}.items():
            assert ppu_parameter_sizes(data, symbol) == sizes
        # Sub-word SREG delivery is not a host-argument byte size: fail closed.
        with pytest.raises(ValueError, match='register argument type'):
            ppu_parameter_sizes(data, 'scalars')
