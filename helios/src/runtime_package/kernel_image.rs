/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiled kernel images shared by graph export and native package loading.

use super::format::invalid;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Section {
    pub name: usize,
    pub kind: u32,
    pub flags: u64,
    pub offset: usize,
    pub size: usize,
    link: usize,
    stride: usize,
}
pub struct Elf<'a> {
    image: &'a [u8],
    pub offset: usize,
    pub names: usize,
    pub sections: Vec<Section>,
}
pub fn number(image: &[u8], offset: usize, size: usize) -> Result<u64> {
    if size > 8 {
        return Err(invalid("invalid cubin integer width"));
    }
    let bytes = image
        .get(
            offset
                ..offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("invalid cubin extent"))?,
        )
        .ok_or_else(|| invalid("invalid cubin extent"))?;
    let mut value = [0; 8];
    value[..size].copy_from_slice(bytes);
    Ok(u64::from_le_bytes(value))
}
impl<'a> Elf<'a> {
    pub fn parse(image: &'a [u8]) -> Result<Self> {
        if image.len() < 64 || &image[..6] != b"\x7fELF\x02\x01" {
            return Err(invalid("expected an ELF64 little-endian cubin"));
        }
        let offset = number(image, 40, 8)? as usize;
        let count = number(image, 60, 2)? as usize;
        let names = number(image, 62, 2)? as usize;
        if number(image, 58, 2)? != 64
            || names == 0
            || names >= count
            || offset
                .checked_add(count * 64)
                .is_none_or(|end| end > image.len())
        {
            return Err(invalid("invalid cubin ELF section table"));
        }
        let mut sections = Vec::with_capacity(count);
        for i in 0..count {
            let p = offset + i * 64;
            sections.push(Section {
                name: number(image, p, 4)? as usize,
                kind: number(image, p + 4, 4)? as u32,
                flags: number(image, p + 8, 8)?,
                offset: number(image, p + 24, 8)? as usize,
                size: number(image, p + 32, 8)? as usize,
                link: number(image, p + 40, 4)? as usize,
                stride: number(image, p + 56, 8)? as usize,
            });
        }
        Ok(Self {
            image,
            offset,
            names,
            sections,
        })
    }
    pub fn contents(&self, index: usize) -> Result<&'a [u8]> {
        let s = self
            .sections
            .get(index)
            .ok_or_else(|| invalid("invalid cubin section reference"))?;
        if s.kind == 8 {
            return Err(invalid("invalid cubin section extent"));
        }
        self.image
            .get(
                s.offset
                    ..s.offset
                        .checked_add(s.size)
                        .ok_or_else(|| invalid("invalid cubin section extent"))?,
            )
            .ok_or_else(|| invalid("invalid cubin section extent"))
    }
    pub fn string(&self, table: usize, index: usize) -> Result<&'a [u8]> {
        if self.sections.get(table).is_none_or(|s| s.kind != 3) {
            return Err(invalid("invalid cubin string reference"));
        }
        let data = self
            .contents(table)?
            .get(index..)
            .ok_or_else(|| invalid("invalid cubin string reference"))?;
        let end = data
            .iter()
            .position(|b| *b == 0)
            .ok_or_else(|| invalid("unterminated cubin string"))?;
        Ok(&data[..end])
    }
    pub fn symbols(
        &self,
        alternate: bool,
    ) -> Result<Vec<(usize, usize, &'a [u8], u8, usize, u64, usize)>> {
        let mut result = Vec::new();
        for (index, s) in self.sections.iter().enumerate() {
            if s.kind != 2 && !(alternate && s.kind == 0x70000085) {
                continue;
            }
            if s.link == 0 || s.link >= self.sections.len() || s.stride != 24 || s.size % 24 != 0 {
                return Err(invalid("invalid cubin symbol table"));
            }
            self.contents(index)
                .map_err(|_| invalid("invalid cubin symbol table extent"))?;
            for p in (s.offset..s.offset + s.size).step_by(24) {
                result.push((
                    p,
                    s.link,
                    self.string(s.link, number(self.image, p, 4)? as usize)?,
                    self.image[p + 4] & 15,
                    number(self.image, p + 6, 2)? as usize,
                    number(self.image, p + 8, 8)?,
                    number(self.image, p + 16, 8)? as usize,
                ));
            }
        }
        Ok(result)
    }
}

pub fn is_ppu(image: &[u8]) -> bool {
    image.starts_with(b"\x7fELF\x02\x01") && image.get(18..20) == Some(&[0x0e, 0x01])
}

/// Host argument widths from the compiled PPU ABI-4 image. SREG/KM locations
/// describe device delivery, not offsets in CUDA's packed launch buffer.
/// Keep pointer classification separate: untyped aggregates remain scanned.
pub fn ppu_parameter_sizes(image: &[u8], symbol: &str) -> Result<Option<Vec<usize>>> {
    if !is_ppu(image) {
        return Ok(None);
    }
    if image.get(7..9) != Some(&[96, 4]) {
        return Err(invalid(
            "unsupported PPU kernel metadata ABI; need OSABI 96, ABI 4",
        ));
    }
    let elf = Elf::parse(image)?;
    let mut functions = 0;
    for (_, _, name, kind, section, _, _) in elf.symbols(false)? {
        if name == symbol.as_bytes() && kind == 2 && section != 0 {
            if elf.sections.get(section).is_none_or(|s| s.flags & 4 == 0) {
                return Err(invalid("PPU kernel symbol has no executable section"));
            }
            functions += 1;
        }
    }
    if functions != 1 {
        return Err(invalid(format!(
            "PPU image needs one defined kernel symbol {symbol}"
        )));
    }
    let name = format!(".hg_info.{symbol}");
    let mut metadata = None;
    for (index, section) in elf.sections.iter().enumerate() {
        if elf.string(elf.names, section.name)? == name.as_bytes() {
            if section.kind != 1 || metadata.is_some() {
                return Err(invalid("invalid or duplicate PPU kernel metadata section"));
            }
            metadata = Some(elf.contents(index)?);
        }
    }
    let data = metadata.ok_or_else(|| invalid(format!("PPU image lacks {name}")))?;
    ppu_arguments(data)
        .map(Some)
        .map_err(|e| invalid(format!("{name}: {e}")))
}

fn ppu_arguments(mut data: &[u8]) -> Result<Vec<usize>> {
    let mut header = false;
    let mut parameters = std::collections::BTreeMap::new();
    while !data.is_empty() {
        // ABI-4 entries: u16 tag, u16 payload length, then the payload.
        let tag = number(data, 0, 2)?;
        let size = number(data, 2, 2)? as usize;
        let payload = data
            .get(4..4 + size)
            .ok_or_else(|| invalid("truncated PPU metadata record"))?;
        data = &data[4 + size..];
        if tag >> 8 != 1 {
            return Err(invalid("unsupported PPU metadata record version"));
        }
        if tag == 0x101 {
            if header || size != 8 {
                return Err(invalid("invalid PPU function metadata header"));
            }
            header = true;
        }
        if tag != 0x102 {
            continue;
        }
        if size != 16 {
            return Err(invalid("invalid PPU argument record size"));
        }
        let count = number(payload, 4, 4)? as usize;
        let ordinal = number(payload, 8, 4)? as usize;
        let (dtype, kind) = (payload[12], payload[13]);
        if ordinal == 0xffff && kind >= 4 {
            continue; // Driver-supplied hidden parameters are not launch arguments.
        }
        if ordinal >= 4096 || count == 0 {
            return Err(invalid("invalid PPU argument ordinal or extent"));
        }
        let width = match kind {
            0 if count == 8 => 8, // Pointer in global kernel-parameter storage.
            1 => count,           // By-value object, including spilled scalar, in bytes.
            2 if count == 2 => 8, // Pointer in two 32-bit SREGs; dtype is pointee type.
            3 => match (dtype, count) {
                (8..=10, 1) => 4,
                (12..=14, 2) => 8,
                _ => {
                    return Err(invalid(format!(
                        "unsupported PPU register argument type {dtype}, words {count}; \
                     require an implemented cuFuncGetParamInfo for this ABI"
                    )));
                }
            },
            _ => return Err(invalid("unsupported PPU argument delivery kind or width")),
        };
        if parameters.insert(ordinal, width).is_some() {
            return Err(invalid("duplicate PPU argument ordinal"));
        }
    }
    if !header {
        return Err(invalid("PPU metadata lacks its function header"));
    }
    parameters
        .into_iter()
        .enumerate()
        .map(|(index, (ordinal, width))| {
            if index != ordinal {
                return Err(invalid("PPU argument ordinals are not contiguous"));
            }
            Ok(width)
        })
        .collect()
}
