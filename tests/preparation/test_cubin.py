# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cubin debug stripping preserves executable layouts and original CUDA symbols."""

import struct
import os
from pathlib import Path

import pytest

from ignition.preparation.module_images import global_symbols, strip_debug


def cubin():
    names = b"\0.shstrtab\0.strtab\0.symtab\0.text.descriptive_kernel\0.nv.global\0.debug_info\0.comment\0.nv.info.descriptive_kernel\0.nv.merc.symtab\0"
    strings = b"\0/private/source/descriptive_kernel\0weight_layout\0__assertfail\0.nv.reservedSmem.offset0\0"
    symbol = lambda name, info, section, size=0: struct.pack(
        "<IBBHQQ", strings.index(name) if name else 0, info, 0, section, 0, size)
    symbols = b"".join((
        bytes(24), symbol(b"/private/source", 4, 0xfff1),
        symbol(b"descriptive_kernel", 0x12, 4, 16),
        symbol(b"weight_layout", 0x11, 5, 8),
        symbol(b"__assertfail", 0x12, 0),
        symbol(b".nv.reservedSmem.offset0", 0x11, 5, 8),
    ))
    entries = [
        (b"", 0, 0, b"", 0, 0),
        (b".shstrtab", 3, 0, names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, symbols, 2, 24),
        (b".text.descriptive_kernel", 1, 6, bytes(range(16)), 0, 0),
        (b".nv.global", 1, 3, b"constant", 0, 0),
        (b".debug_info", 1, 0, b"/private/kernel.py\0", 0, 0),
        (b".comment", 1, 0, b"compiler build identity\0", 0, 0),
        (b".nv.info.descriptive_kernel", 0x70000000, 0, b"cuda metadata", 3, 0),
        (b".nv.merc.symtab", 0x70000085, 0x10000000, symbols, 2, 24),
    ]
    result = bytearray(64 + len(entries) * 64)
    result[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HH", result, 16, 2, 190)
    struct.pack_into("<Q", result, 40, 64)
    struct.pack_into("<HHHHHH", result, 52, 64, 56, 0, 64, len(entries), 1)
    for i, (name, kind, flags, data, link, stride) in enumerate(entries):
        struct.pack_into("<IIQQQQIIQQ", result, 64 + i * 64,
                         names.index(name) if name else 0, kind, flags, 0,
                         len(result), len(data), link, 0, 1, stride)
        result.extend(data)
    return bytes(result)


def sections(image):
    offset = struct.unpack_from("<Q", image, 40)[0]
    size, count = struct.unpack_from("<HH", image, 58)
    return [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * size) for i in range(count)]


def test_debug_stripping_preserves_code_metadata_and_symbol_names():
    original = cubin()
    clean = strip_debug(original)
    for name in (b"descriptive_kernel", b"weight_layout", b"__assertfail"):
        assert name + b"\0" in clean
    assert b"compiler build identity" not in clean
    assert global_symbols(clean) == [".nv.reservedSmem.offset0", "weight_layout"]
    before, after = sections(original), sections(clean)
    assert len(before) == len(after)
    for index in (1, 2, 3, 4, 5, 8, 9):
        assert before[index] == after[index]
        start, size = before[index][4:6]
        assert original[start:start + size] == clean[start:start + size]
    for index in (6, 7):
        assert after[index][5] == 0
        start, size = before[index][4:6]
        assert clean[start:start + size] == bytes(size)


@pytest.mark.parametrize("mutation,error", [
    (lambda b: struct.pack_into("<Q", b, 64 + 6 * 64 + 8, 2), "device-allocated"),
    (lambda b: struct.pack_into("<Q", b, 64 + 6 * 64 + 24, sections(bytes(b))[4][4]), "overlaps executable metadata"),
    (lambda b: struct.pack_into("<I", b, 64 + 3 * 64 + 40, 100), "symbol table"),
    (lambda b: struct.pack_into("<Q", b, 64 + 3 * 64 + 56, 0), "symbol table"),
    (lambda b: struct.pack_into("<Q", b, 64 + 2 * 64 + 32, len(b)), "overlaps executable metadata"),
])
def test_invalid_metadata_cannot_overwrite_executable_bytes(mutation, error):
    image = bytearray(cubin())
    mutation(image)
    with pytest.raises(ValueError, match=error):
        strip_debug(bytes(image))


@pytest.mark.parametrize('name,kind,flags', [
    (b'.data', 1, 3), (b'.data.foo', 1, 3), (b'.bss', 8, 3), (b'.rodata', 1, 2),
])
def test_ppu_data_globals_require_native_machine_and_allocated_storage(name, kind, flags):
    image = bytearray(cubin())
    offset = image.index(b'.nv.global\0')
    image[offset:offset + 10] = name.ljust(10, b'\0')
    struct.pack_into('<I', image, 64 + 5 * 64 + 4, kind)
    struct.pack_into('<Q', image, 64 + 5 * 64 + 8, flags)
    for machine in (62, 190):
        struct.pack_into('<H', image, 18, machine)
        assert global_symbols(bytes(image)) == []
    struct.pack_into('<H', image, 18, 270)
    assert global_symbols(bytes(image)) == ['.nv.reservedSmem.offset0', 'weight_layout']
    struct.pack_into('<Q', image, 64 + 5 * 64 + 8, 0)
    assert global_symbols(bytes(image)) == []


def test_actual_ppu_module_preserves_globals_and_executable_sections():
    path = os.environ.get('FLARE_TEST_PPU_ELF')
    if path is None:
        pytest.skip('compile fixtures/ppu_module_globals.cu with the PPU SDK and supply FLARE_TEST_PPU_ELF')
    original = Path(path).read_bytes()
    assert struct.unpack_from('<H', original, 18)[0] == 270
    wanted = {'qwen_export_counter', 'qwen_export_zero', 'qwen_export_constants', 'qwen_export_pointer'}
    assert wanted <= set(global_symbols(original))
    clean = strip_debug(original)
    assert global_symbols(clean) == global_symbols(original)
    before, after = sections(original), sections(clean)
    assert len(before) == len(after)
    names = before[struct.unpack_from('<H', original, 62)[0]]
    strings = original[names[4]:names[4] + names[5]]
    retained = set()
    for a, b in zip(before, after, strict=True):
        name = strings[a[0]:].split(b'\0', 1)[0]
        if b'.debug' in name or name == b'.comment':
            assert b[5] == 0
            continue
        retained.add(name)
        assert a == b, name
        if a[1] != 8:  # NOBITS owns storage without bytes in the module image.
            start, size = a[4:6]
            assert original[start:start + size] == clean[start:start + size], name
    assert {b'.data', b'.bss', b'.hg_info', b'.rela.data'} <= retained
