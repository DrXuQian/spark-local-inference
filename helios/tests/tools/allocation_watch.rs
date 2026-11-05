/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Count CUDA allocation calls without relying on system-wide free memory.

use std::{
    ffi::{CStr, c_char, c_int, c_void},
    sync::atomic::{AtomicUsize, Ordering},
};

type Handle = *mut c_void;
type Callback = unsafe extern "C" fn(Handle, c_int, u32, *const CallbackData);

#[repr(C)]
struct CallbackData {
    site: c_int,
    name: *const c_char,
}

fn allocation_call(name: &[u8]) -> bool {
    // The HGPTI CUDA facade keeps hg-prefixed callback names, as the native
    // export recorder already observes. Match both spellings of the same APIs.
    let Some(name) = name
        .strip_prefix(b"cu")
        .or_else(|| name.strip_prefix(b"hg"))
    else {
        return false;
    };
    [
        b"MemAlloc".as_slice(),
        b"MemCreate",
        b"MemMap",
        b"MemHostAlloc",
        b"MemHostRegister",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

unsafe extern "C" fn record(user: Handle, domain: c_int, _: u32, data: *const CallbackData) {
    if domain != 1 || data.is_null() {
        return;
    }
    // CUPTI owns the callback prefix and keeps its function name alive during the call.
    let data = unsafe { &*data };
    if data.site != 0 || data.name.is_null() {
        return;
    }
    let name = unsafe { CStr::from_ptr(data.name) }.to_bytes();
    if allocation_call(name) {
        unsafe { &*user.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct AllocationWatch {
    _library: libloading::Library,
    subscriber: Handle,
    unsubscribe: unsafe extern "C" fn(Handle) -> c_int,
    calls: Box<AtomicUsize>,
}

impl AllocationWatch {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let library = libloading::Library::new("libcupti.so")?;
            let subscribe = library
                .get::<unsafe extern "C" fn(*mut Handle, Callback, Handle) -> c_int>(
                    b"cuptiSubscribe\0",
                )?;
            let enable = library
                .get::<unsafe extern "C" fn(u32, Handle, c_int) -> c_int>(b"cuptiEnableDomain\0")?;
            let unsubscribe =
                *library.get::<unsafe extern "C" fn(Handle) -> c_int>(b"cuptiUnsubscribe\0")?;
            let calls = Box::new(AtomicUsize::new(0));
            let mut subscriber = std::ptr::null_mut();
            assert_eq!(
                subscribe(
                    &mut subscriber,
                    record,
                    (&*calls as *const AtomicUsize).cast_mut().cast()
                ),
                0
            );
            assert_eq!(enable(1, subscriber, 1), 0);
            Ok(Self {
                _library: library,
                subscriber,
                unsubscribe,
                calls,
            })
        }
    }

    pub fn count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Drop for AllocationWatch {
    fn drop(&mut self) {
        unsafe {
            (self.unsubscribe)(self.subscriber);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_and_hgpti_entries_count_the_same_allocation_calls() {
        for prefix in ["cu", "hg"] {
            let calls = AtomicUsize::new(0);
            for name in [
                "MemAlloc_v2",
                "MemCreate",
                "MemMap",
                "MemHostAlloc",
                "MemHostRegister_v2",
            ] {
                let name = std::ffi::CString::new(format!("{prefix}{name}")).unwrap();
                let mut data = CallbackData {
                    site: 0,
                    name: name.as_ptr(),
                };
                let user = (&calls as *const AtomicUsize).cast_mut().cast();
                unsafe { record(user, 1, 0, &data) };
                data.site = 1;
                unsafe { record(user, 1, 0, &data) };
            }
            assert_eq!(calls.load(Ordering::Relaxed), 5);
        }
    }

    #[test]
    fn non_allocating_and_unrecognized_callbacks_do_not_count() {
        for name in [
            b"hgMemFree".as_slice(),
            b"cuMemcpyHtoD",
            b"hgLaunchKernel",
            b"otherMemAlloc",
        ] {
            assert!(!allocation_call(name));
        }
    }
}
