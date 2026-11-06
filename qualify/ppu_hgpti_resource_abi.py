# SPDX-License-Identifier: Apache-2.0

"""Reproduce the SDK 2.2.1 HGPTI module-resource ABI mismatch without a device.

This diagnostic invokes that SDK's exported, internal HGtool resource dispatcher
with synthetic module events. It uses the real CUDA facade subscription and the
real HGPTI dispatcher, not Flare's recorder. The HGtool event layout/IDs below are
specific to SDK 2.2.1; this is not a public SDK API or a device qualification.
Activation errors remain visible; synthetic delivery cannot validate activation.
"""

import argparse
import ctypes as C
import os
import struct
from pathlib import Path


def bind(library, name, result, arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, arguments
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    args = parser.parse_args()
    cupti = C.CDLL(str(args.sdk / 'CUDA_SDK/lib64/libcupti.so.13'))
    hgpti = C.CDLL(str(args.sdk / 'lib/libhgpti.so'))
    callback_type = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_uint32, C.c_void_p)
    context, module = 0x12340000, 0x56780000
    image = C.create_string_buffer(b'\x7fELF' + bytes(60))
    address, length = C.addressof(image), 64
    # SDK 2.2.1's HGtool event: context, module, reserved, size, reserved, image.
    event = (C.c_uint64 * 6)(context, module, 0, length, 0, address)
    observed, failures = [], []
    memory = os.open('/proc/self/mem', os.O_RDONLY)

    def words(pointer):
        # Malformed metadata must report an error instead of dereferencing an
        # arbitrary address. No driver/runtime API is called from the callback.
        return struct.unpack('<3Q', os.pread(memory, 24, pointer))

    @callback_type
    def callback(user, domain, cbid, data):
        try:
            if domain != 3 or cbid not in (6, 7):
                raise ValueError(f'unexpected callback {domain}/{cbid}')
            raw = words(data)
            observed.append(cbid)
            print(f'callback domain={domain} id={cbid}: {[hex(x) for x in raw]}', flush=True)
            if raw[1:] == (length, address):
                raise ValueError('bare ModuleResourceData; missing ResourceData wrapper and context')
            if raw[0] != context:
                raise ValueError(f'context {raw[0]:#x} != {context:#x}')
            descriptor = words(raw[2])
            if descriptor[1:] != (length, address):
                raise ValueError(f'wrong descriptor: {descriptor!r}')
            if os.pread(memory, length, descriptor[2]) != image.raw[:length]:
                raise ValueError('module bytes changed')
        except (OSError, ValueError, TypeError, OverflowError, struct.error) as error:
            failures.append(str(error))
            print(f'FAIL: {error}', flush=True)

    subscribe = bind(cupti, 'cuptiSubscribe', C.c_int,
                     [C.POINTER(C.c_void_p), callback_type, C.c_void_p])
    enable = bind(cupti, 'cuptiEnableCallback', C.c_int,
                  [C.c_uint32, C.c_void_p, C.c_int, C.c_uint32])
    unsubscribe = bind(cupti, 'cuptiUnsubscribe', C.c_int, [C.c_void_p])
    dispatch = bind(hgpti, '_ZN5hgpti29HgtoolResourceCallbackHandlerEPvjPKv', None,
                    [C.c_void_p, C.c_uint32, C.c_void_p])
    subscriber = C.c_void_p()
    try:
        status = subscribe(C.byref(subscriber), callback, None)
        print(f'cuptiSubscribe={status}', flush=True)
        if status:
            raise RuntimeError(f'cuptiSubscribe failed: {status}')
        try:
            for cbid in (6, 7):
                status = enable(1, subscriber, 3, cbid)
                print(f'cuptiEnableCallback(3, {cbid})={status}', flush=True)
            print('Synthetic dispatcher calls only; activation/device delivery are untested.', flush=True)
            for event_id in (7, 8, 9):  # HGtool load, unload-starting, unload-finished.
                dispatch(None, event_id, C.byref(event))
        finally:
            status = unsubscribe(subscriber)
            print(f'cuptiUnsubscribe={status}', flush=True)
            if status:
                failures.append(f'cuptiUnsubscribe failed: {status}')
    finally:
        os.close(memory)
    if observed != [6, 7]:
        failures.append(f'missing or reordered module events: {observed}')
    if failures:
        raise SystemExit(1)
    print('Synthetic module resource ABI passed; device export/replay remain untested.', flush=True)


if __name__ == '__main__':
    main()
