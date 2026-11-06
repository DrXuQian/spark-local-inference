/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Readable cache-entry names: `<platform>-tp<N>-<UTC timestamp>`.
//!
//! Names are display metadata only. Payload and package identities remain
//! content hashes and signatures; renaming an entry never changes them.

use std::time::{SystemTime, UNIX_EPOCH};

/// Conventional platform labels retained for existing package names.
/// Other devices use their PCI identity; use [`is_platform`] for validation.
pub const PLATFORMS: [&str; 2] = ["gb10", "gb202"];

/// GB202 device identifiers used by the existing conventional label.
pub const GB202_DEVICES: [u32; 9] = [
    0x2b85, 0x2b87, 0x2b8c, 0x2bb1, 0x2bb2, 0x2bb3, 0x2bb4, 0x2bb5, 0x2bb9,
];

/// GB10 device identifiers. `0x2e12` is the DGX Spark part; `0x2e03` is the
/// RTX Spark N1X part, which NVML reports for the same SM121 platform.
pub const GB10_DEVICES: [u32; 2] = [0x2e12, 0x2e03];

/// Stable label for NVML's `device<<16 | vendor` identity, independent of bus
/// location and marketing names. A label does not establish executable compatibility.
pub fn platform_from_pci(pci_id: u32) -> Option<String> {
    let vendor = pci_id & 0xffff;
    let device = pci_id >> 16;
    if vendor == 0 || vendor == 0xffff || device == 0xffff {
        return None;
    }
    if vendor == 0x10de {
        if GB10_DEVICES.contains(&device) {
            return Some("gb10".into());
        }
        if GB202_DEVICES.contains(&device) {
            return Some("gb202".into());
        }
    }
    Some(format!("pci-{vendor:04x}-{device:04x}"))
}

/// Recognize conventional labels and canonical PCI identities without admitting
/// arbitrary path components or treating an SM architecture as a device identity.
pub fn is_platform(platform: &str) -> bool {
    if PLATFORMS.contains(&platform) {
        return true;
    }
    let Some((vendor, device)) = platform
        .strip_prefix("pci-")
        .and_then(|id| id.split_once('-'))
    else {
        return false;
    };
    if vendor.len() != 4 || device.len() != 4 {
        return false;
    }
    let (Ok(vendor), Ok(device)) = (
        u16::from_str_radix(vendor, 16),
        u16::from_str_radix(device, 16),
    ) else {
        return false;
    };
    platform_from_pci(u32::from(device) << 16 | u32::from(vendor)).as_deref() == Some(platform)
}

/// UTC `YYYYMMDDHHMM` for one instant, without external crates.
pub fn utc_timestamp(at: SystemTime) -> String {
    let seconds = at
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0);
    let days = (seconds / 86_400) as i64;
    let minutes = (seconds % 86_400) / 60;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}{:02}{:02}",
        minutes / 60,
        minutes % 60
    )
}

/// Base entry name for one exported deployment.
pub fn candidate(platform: &str, tensor_parallel: u32, at: SystemTime) -> Option<String> {
    if !is_platform(platform) || tensor_parallel == 0 {
        return None;
    }
    Some(format!(
        "{platform}-tp{tensor_parallel}-{}",
        utc_timestamp(at)
    ))
}

/// Compatibility label for packages that lack an authenticated device identity.
///
/// This does not infer a PCI identity. It keeps older
/// packages with only SM or custom deployment identifiers loadable without
/// guessing a device from architecture or the local host.
pub fn unknown_device_candidate(sizes: &[u32], at: SystemTime) -> Option<String> {
    if sizes.is_empty() || sizes.iter().any(|size| *size == 0) {
        return None;
    }
    let label = sizes
        .iter()
        .map(|size| format!("tp{size}"))
        .collect::<Vec<_>>()
        .join("+");
    Some(format!("unknown-device-{label}-{}", utc_timestamp(at)))
}

/// Base entry name for one package's authenticated deployment sizes.
///
/// Sizes are sorted and deduplicated. An unsupported or absent device identity
/// uses the explicit `unknown-device` label; architecture and the local host
/// never supply one.
pub fn deployment_candidate(
    platform: Option<&str>,
    sizes: &[u32],
    at: SystemTime,
) -> Option<String> {
    let mut sizes = sizes.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    if sizes.is_empty() || sizes.iter().any(|size| *size == 0) {
        return None;
    }
    let Some(platform) = platform.filter(|platform| is_platform(platform)) else {
        return unknown_device_candidate(&sizes, at);
    };
    if let [size] = sizes.as_slice() {
        return candidate(platform, *size, at);
    }
    let label = sizes
        .iter()
        .map(|size| format!("tp{size}"))
        .collect::<Vec<_>>()
        .join("+");
    Some(format!("{platform}-{label}-{}", utc_timestamp(at)))
}

/// Same-minute collision label: the second attempt is `-2`, then `-3`, and so on.
pub fn collision(base: &str, ordinal: u32) -> Option<String> {
    (ordinal >= 2).then(|| format!("{base}-{ordinal}"))
}

/// Howard Hinnant's civil-from-days algorithm, for UTC fields only.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    #[test]
    fn timestamps_are_utc_and_minute_resolved() {
        assert_eq!(utc_timestamp(UNIX_EPOCH), "197001010000");
        // 2026-10-17T09:23:00Z
        assert_eq!(utc_timestamp(at(1_792_228_980)), "202610170923");
        // Leap day.
        assert_eq!(utc_timestamp(at(1_709_164_800)), "202402290000");
        // One minute later, same day.
        assert_eq!(utc_timestamp(at(1_792_229_040)), "202610170924");
    }

    #[test]
    fn deployments_name_by_platform_and_tensor_parallel() {
        assert_eq!(
            candidate("gb202", 1, at(1_792_228_980)).as_deref(),
            Some("gb202-tp1-202610170923")
        );
        assert_eq!(
            candidate("gb10", 8, at(1_792_228_980)).as_deref(),
            Some("gb10-tp8-202610170923")
        );
        assert!(candidate("sm120", 1, at(0)).is_none());
        assert!(candidate("gb202", 0, at(0)).is_none());
    }

    #[test]
    fn pci_ids_select_the_platform_identifier() {
        for device in GB10_DEVICES {
            assert_eq!(
                platform_from_pci(device << 16 | 0x10de).as_deref(),
                Some("gb10"),
                "{device:#06x}"
            );
        }
        for device in GB202_DEVICES {
            assert_eq!(
                platform_from_pci(device << 16 | 0x10de).as_deref(),
                Some("gb202"),
                "{device:#06x}"
            );
        }
        for (id, expected) in [
            (0x6001_1ded, "pci-1ded-6001"),
            (0x6002_1ded, "pci-1ded-6002"),
            (0x2bbc_1af4, "pci-1af4-2bbc"),
            (0x1e02_10de, "pci-10de-1e02"),
        ] {
            assert_eq!(platform_from_pci(id).as_deref(), Some(expected));
            assert!(is_platform(expected));
            assert_eq!(
                candidate(expected, 1, at(0)).unwrap(),
                format!("{expected}-tp1-197001010000")
            );
            assert_eq!(
                deployment_candidate(Some(expected), &[2, 1], at(0)).unwrap(),
                format!("{expected}-tp1+tp2-197001010000")
            );
        }
        for id in [0, 0x6001_0000, 0x6001_ffff, 0xffff_1ded] {
            assert_eq!(platform_from_pci(id), None);
        }
    }

    #[test]
    fn pci_labels_require_canonical_device_ids_and_safe_names() {
        for invalid in [
            "pci-1ded-6001/../x",
            "pci-1ded-6001-extra",
            "pci-1DED-6001",
            "pci-1ded-601",
            "pci-0000-6001",
            "pci-ffff-6001",
            "pci-1ded-ffff",
            "pci-10de-2e12",
            "sm80",
            "ppu10",
            "",
            "pci-+ded-6001",
        ] {
            assert!(!is_platform(invalid), "{invalid}");
            assert!(candidate(invalid, 1, at(0)).is_none(), "{invalid}");
        }
    }

    #[test]
    fn unknown_device_labels_are_explicit_and_never_use_sm_names() {
        let base = unknown_device_candidate(&[1], at(1_792_228_980)).unwrap();
        assert_eq!(base, "unknown-device-tp1-202610170923");
        assert!(!base.contains("sm120"));
        let union = unknown_device_candidate(&[1, 2], at(1_792_228_980)).unwrap();
        assert_eq!(union, "unknown-device-tp1+tp2-202610170923");
        assert!(unknown_device_candidate(&[], at(0)).is_none());
        assert!(unknown_device_candidate(&[0], at(0)).is_none());
    }

    #[test]
    fn deployments_name_by_signed_platform_and_sorted_union_sizes() {
        let at = at(1_792_228_980);
        assert_eq!(
            deployment_candidate(Some("gb202"), &[2, 1], at).as_deref(),
            Some("gb202-tp1+tp2-202610170923")
        );
        assert_eq!(
            deployment_candidate(Some("sm120"), &[1], at).as_deref(),
            Some("unknown-device-tp1-202610170923")
        );
        assert_eq!(
            deployment_candidate(None, &[2, 1], at).as_deref(),
            Some("unknown-device-tp1+tp2-202610170923")
        );
        assert!(deployment_candidate(None, &[], at).is_none());
        assert!(deployment_candidate(Some("gb202"), &[0], at).is_none());
    }

    #[test]
    fn same_minute_collisions_get_numeric_suffixes() {
        let base = candidate("gb202", 1, at(1_792_228_980)).unwrap();
        assert!(collision(&base, 1).is_none());
        assert_eq!(collision(&base, 2).unwrap(), format!("{base}-2"));
        assert_eq!(collision(&base, 3).unwrap(), format!("{base}-3"));
    }
}
