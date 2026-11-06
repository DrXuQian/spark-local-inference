/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! ELF transformations that preserve CUDA section and symbol indices.

use super::{ExportResult, invalid};
use std::collections::BTreeSet;

use helios_core::runtime_package::kernel_image::{Elf, number};

fn put(image: &mut [u8], offset: usize, value: u64, size: usize) {
    image[offset..offset + size].copy_from_slice(&value.to_le_bytes()[..size]);
}
fn overlap(a: usize, size: usize, b: usize, length: usize) -> bool {
    size > 0
        && length > 0
        && (a as u128) < b as u128 + length as u128
        && (b as u128) < a as u128 + size as u128
}
pub(crate) fn strip_global_source(symbol: &str, data: &[u8]) -> ExportResult<Vec<u8>> {
    if !symbol.starts_with("assertFile_") {
        return Ok(data.to_vec());
    }
    if data.last() != Some(&0) {
        return Err(invalid(
            "CUDA assertion filename must be a terminated string",
        ));
    }
    let mut result = vec![0; data.len()];
    let length = 6.min(data.len() - 1);
    result[..length].copy_from_slice(&b"kernel"[..length]);
    Ok(result)
}

pub(crate) fn strip_debug(image: &[u8]) -> ExportResult<Vec<u8>> {
    let elf = Elf::parse(image)?;
    let mut debug = BTreeSet::new();
    for (i, s) in elf.sections.iter().enumerate() {
        let name = elf.string(elf.names, s.name)?;
        if [b".debug".as_slice(), b".zdebug", b".nv_debug"]
            .iter()
            .any(|needle| name.windows(needle.len()).any(|w| w == *needle))
            || name == b".comment"
        {
            debug.insert(i);
        }
    }
    let mut result = image.to_vec();
    for &i in &debug {
        let s = &elf.sections[i];
        if s.flags & 2 != 0
            || s.offset
                .checked_add(s.size)
                .is_none_or(|end| end > image.len())
        {
            return Err(invalid("invalid or device-allocated cubin debug section"));
        }
        for (j, t) in elf.sections.iter().enumerate() {
            if !debug.contains(&j) && t.kind != 8 && overlap(s.offset, s.size, t.offset, t.size) {
                return Err(invalid("cubin debug section overlaps executable metadata"));
            }
        }
        result[s.offset..s.offset + s.size].fill(0);
        put(&mut result, elf.offset + i * 64 + 32, 0, 8);
    }
    for (_, _, name, kind, index, value, size) in elf.symbols(true)? {
        if kind != 1
            || !name.starts_with(b"assertFile_")
            || index == 0
            || index >= elf.sections.len()
        {
            continue;
        }
        let s = &elf.sections[index];
        if s.kind == 8
            || value
                .checked_add(size as u64)
                .is_none_or(|end| end > s.size as u64)
        {
            return Err(invalid("invalid CUDA assertion filename symbol"));
        }
        elf.contents(index)?;
        let start = s.offset + value as usize;
        let clean = strip_global_source(std::str::from_utf8(name)?, &image[start..start + size])?;
        result[start..start + size].copy_from_slice(&clean);
    }
    Ok(result)
}

pub(crate) fn global_symbols(image: &[u8]) -> ExportResult<Vec<String>> {
    let elf = Elf::parse(image)?;
    // SDK 2.1.1 native PPU images use machine 270 and ordinary allocated
    // data sections, including NOBITS for zero-initialized device globals.
    let ppu = number(image, 18, 2)? == 270;
    let mut result = BTreeSet::new();
    for (_, _, name, kind, index, _, size) in elf.symbols(false)? {
        if kind != 1 || size == 0 || index == 0 || index >= elf.sections.len() {
            continue;
        }
        let section_name = elf.string(elf.names, elf.sections[index].name)?;
        let section = &elf.sections[index];
        let ppu_data = ppu
            && section.flags & 2 != 0
            && matches!(section.kind, 1 | 8)
            && [b".data".as_slice(), b".bss", b".rodata"].iter().any(|name| {
                section_name == *name
                    || section_name.strip_prefix(*name).is_some_and(|tail| tail.starts_with(b"."))
            });
        if section_name.starts_with(b".nv.global")
            || section_name.starts_with(b".nv.constant")
            || ppu_data
        {
            result.insert(std::str::from_utf8(name)?.to_owned());
        }
    }
    Ok(result.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubin() -> Vec<u8> {
        let names = b"\0.strtab\0.symtab\0.text.descriptive_kernel\0.shstrtab\0.nv.global\0.debug_info\0.nv.info.descriptive_kernel\0";
        let strings = b"\0/private/source/descriptive_kernel\0weight_layout\0__assertfail\0.nv.reservedSmem.offset0\0";
        let find =
            |data: &[u8], name: &[u8]| data.windows(name.len()).position(|w| w == name).unwrap();
        let mut symbols = vec![0; 24];
        for (name, kind, index, size) in [
            (b"/private/source".as_slice(), 4u8, 0xfff1u64, 0),
            (b"descriptive_kernel", 0x12, 3, 16),
            (b"weight_layout", 0x11, 5, 8),
            (b"__assertfail", 0x12, 0, 0),
            (b".nv.reservedSmem.offset0", 0x11, 5, 8),
        ] {
            let mut symbol = vec![0; 24];
            put(&mut symbol, 0, find(strings, name) as u64, 4);
            symbol[4] = kind;
            put(&mut symbol, 6, index, 2);
            put(&mut symbol, 16, size, 8);
            symbols.extend(symbol);
        }
        let entries = [
            (b"".as_slice(), 0, 0, vec![], 0, 0),
            (b".strtab", 3, 0, strings.to_vec(), 0, 0),
            (b".symtab", 2, 0, symbols, 1, 24),
            (b".text.descriptive_kernel", 1, 6, (0..16).collect(), 0, 0),
            (b".shstrtab", 3, 0, names.to_vec(), 0, 0),
            (b".nv.global", 1, 3, b"constant".to_vec(), 0, 0),
            (b".debug_info", 1, 0, b"/private/kernel.py\0".to_vec(), 0, 0),
            (
                b".nv.info.descriptive_kernel",
                0x70000000,
                0,
                b"cuda metadata".to_vec(),
                2,
                0,
            ),
        ];
        let mut image = vec![0; 64 + entries.len() * 64];
        image[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        put(&mut image, 40, 64, 8);
        put(&mut image, 54, 56, 2);
        put(&mut image, 58, 64, 2);
        put(&mut image, 60, entries.len() as u64, 2);
        put(&mut image, 62, 4, 2);
        for (i, (name, kind, flags, data, link, stride)) in entries.iter().enumerate() {
            let p = 64 + i * 64;
            put(
                &mut image,
                p,
                if name.is_empty() {
                    0
                } else {
                    find(names, name) as u64
                },
                4,
            );
            put(&mut image, p + 4, *kind, 4);
            put(&mut image, p + 8, *flags, 8);
            let offset = image.len();
            put(&mut image, p + 24, offset as u64, 8);
            put(&mut image, p + 32, data.len() as u64, 8);
            put(&mut image, p + 40, *link, 4);
            put(&mut image, p + 56, *stride, 8);
            image.extend(data);
        }
        image
    }
    #[test]
    fn debug_stripping_preserves_executable_layout_and_symbol_names() {
        let original = cubin();
        let clean = strip_debug(&original).unwrap();
        for name in [b"descriptive_kernel".as_slice(), b"weight_layout"] {
            assert!(clean.windows(name.len()).any(|bytes| bytes == name));
        }
        assert_eq!(global_symbols(&clean).unwrap(), vec![".nv.reservedSmem.offset0", "weight_layout"]);
        let before = Elf::parse(&original).unwrap();
        let after = Elf::parse(&clean).unwrap();
        for i in [1, 2, 3, 4, 5, 7] {
            assert_eq!(before.contents(i).unwrap(), after.contents(i).unwrap());
            assert_eq!(before.sections[i].offset, after.sections[i].offset);
        }
        assert_eq!(after.sections[6].size, 0);
        assert!(clean[before.sections[6].offset..before.sections[6].offset + before.sections[6].size]
            .iter().all(|byte| *byte == 0));
    }
    #[test]
    fn debug_stripping_redacts_embedded_assertion_filenames() {
        let mut image = cubin();
        let symbol = image
            .windows(13)
            .position(|w| w == b"weight_layout")
            .unwrap();
        image[symbol..symbol + 13].copy_from_slice(b"assertFile_1\0");
        assert!(
            strip_debug(&image)
                .unwrap_err()
                .to_string()
                .contains("terminated string")
        );
        let start = Elf::parse(&image).unwrap().sections[5].offset;
        image[start..start + 8].copy_from_slice(b"/tmp/a\0\0");
        let clean = strip_debug(&image).unwrap();
        assert_eq!(&clean[start..start + 8], b"kernel\0\0");
    }

    #[test]
    fn assertion_source_redaction_preserves_size_and_termination() {
        for size in 1..20 {
            let mut source = vec![b'x'; size];
            source[size - 1] = 0;
            let clean = strip_global_source("assertFile_1", &source).unwrap();
            assert_eq!(clean.len(), size);
            assert_eq!(*clean.last().unwrap(), 0);
            assert!(clean.starts_with(&b"kernel"[..6.min(size - 1)]));
        }
        assert!(strip_global_source("assertFile_1", b"unterminated").is_err());
        assert_eq!(
            strip_global_source("other", b"/private/file").unwrap(),
            b"/private/file"
        );
    }
}
