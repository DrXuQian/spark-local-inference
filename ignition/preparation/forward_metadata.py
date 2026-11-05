# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Serialize prepared Python forwards without tensor data or CUDA resources."""

import ctypes
import io
import pickle
import socket
import sys
import types
from contextvars import ContextVar
from threading import Lock, RLock, local

import torch
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary

_MODE = ContextVar('flare_forward_metadata_mode')
_TENSOR_KINDS = ContextVar('flare_forward_tensor_kinds', default=None)
_TENSOR_KEYS = ContextVar('flare_forward_tensor_dispatch_keys', default=None)
_LOCK_TYPES = (type(Lock()), type(RLock()))


class _Unavailable:
    def __init__(self, description):
        self.description = description

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'forward tracing called process-local resource {self.description}')

    def __getattr__(self, name):
        # Unpickling probes special methods such as __setstate__ before restoring the description.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        description = self.__dict__.get('description', 'unknown')
        raise RuntimeError(f'forward tracing accessed process-local resource {description}.{name}')


def _storage(nbytes):
    return torch.UntypedStorage(nbytes, device='meta')


class _Storage:
    def __init__(self, nbytes):
        self.nbytes = nbytes

    def __reduce__(self):
        return _storage, (self.nbytes,)


def _tensor(storage, shape, stride, offset, dtype, device, requires_grad, kind, dispatch_keys=None):
    with torch._C._DisableTorchDispatch():
        meta = torch.empty(0, dtype=dtype, device='meta').set_(storage, offset, shape, stride)
        meta.requires_grad_(requires_grad)
    value = FakeTensor(_MODE.get(), meta, device)
    if dispatch_keys is not None:
        _TENSOR_KEYS.get()[value] = dispatch_keys
    if kind is torch.nn.Parameter:
        value._is_param = True
    _TENSOR_KINDS.get()[value] = kind
    return value


def _object(cls):
    return object.__new__(cls)


def _set_state(value, state):
    value.__dict__.update(state)


def _mapping_proxy(values):
    return types.MappingProxyType(values)


def _loaded_types(module, *names):
    """Resource objects imply their defining modules are already loaded.

    Metadata serialization must not initialize an unused compiler/operator
    package merely to identify objects which cannot occur in this forward.
    """
    namespace = vars(sys.modules[module]) if module in sys.modules else {}
    return tuple(value for name in names if isinstance(value := namespace.get(name), type))


class _ForwardPickler(pickle.Pickler):
    def __init__(self, file):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.storages = {}
        self.sessions = (*_loaded_types('b12x.preparation', 'PreparationSession'),
                         *_loaded_types('ignition.preparation.external_session', 'ExternalPreparationSession'))
        self.devices = _loaded_types('b12x.preparation.device', 'DetectedDevice')
        self.slots = _loaded_types('b12x.preparation.types', '_PreparedSlot')
        self.prepared = _loaded_types('b12x.preparation.types', '_Prepared')
        self.triton_functions = _loaded_types('triton.runtime', 'JITFunction')
        self.ffi_objects = _loaded_types('tvm_ffi.core', 'CObject')
        self.compiler_resources = (
            *_loaded_types('triton.compiler', 'CompiledKernel'),
            *_loaded_types('cutlass.base_dsl.jit_executor', 'JitCompiledFunction'),
            *_loaded_types('b12x._lib.compile_plan', 'CompiledCuTeProgram', 'DeferredCuTeKernel', 'DeferredTritonKernel'),
            *_loaded_types('ignition.preparation.package_storage', 'PackageStorage'),
            *_loaded_types('ignition.preparation.peer_storage', 'PeerStorage'))

    def reducer_override(self, value):
        from .device_binding import restore_device, restore_prepared_device

        if isinstance(value, self.sessions):
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if isinstance(value, _Unavailable):
            return _Unavailable, (value.description,)
        # A worker may trace another local rank's forward on its own GPU; the artifact relocates when bound.
        if isinstance(value, torch.device):
            return restore_device, (value.type, value.index)
        if isinstance(value, self.devices):
            return restore_prepared_device, (type(value), dict(vars(value)))
        if isinstance(value, torch.Tensor):
            if type(value) not in (torch.Tensor, torch.nn.Parameter, FakeTensor):
                raise TypeError(f'unsupported forward tensor subclass: {type(value)}')
            storage = value.untyped_storage()
            owner = self.storages.setdefault(storage._cdata, _Storage(storage.nbytes()))
            keys = torch._C._dispatch_keys(value).raw_repr()
            if isinstance(value, FakeTensor):
                recorded = _TENSOR_KEYS.get()
                keys = None if value.dispatch_keys is None else value.dispatch_keys.raw_repr()
                keys = keys if recorded is None else recorded.get(value, keys)
            return _tensor, (owner, tuple(value.shape), value.stride(), value.storage_offset(),
                             value.dtype, value.device, value.requires_grad, type(value),
                             keys)
        if isinstance(value, self.slots):
            # Tracing needs selected layouts; normal plan pickling clears prepared state.
            return _object, (type(value),), dict(vars(value)), None, None, _set_state
        if isinstance(value, self.prepared):
            state = dict(vars(value))
            for name in ('retained', 'owners', 'closers', 'users'):
                state[name] = _Unavailable(f'b12x prepared {name}')
            return _object, (type(value),), state, None, None, _set_state
        if isinstance(value, self.triton_functions):
            from .torch_aot_api import deserialize_triton_kernel, serialize_triton_kernel

            return deserialize_triton_kernel, (serialize_triton_kernel(value),)
        if isinstance(value, types.MappingProxyType):
            return _mapping_proxy, (dict(value),)
        if isinstance(value, self.ffi_objects) or (not isinstance(value, type) and '<locals>' in type(value).__qualname__):
            # FlashInfer's TVM-FFI kernel libraries and its locally defined kernel runners are process-local.
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if isinstance(value, (torch.cuda.Stream, torch.cuda.Event, torch.cuda.CUDAGraph, torch.cuda.MemPool,
                              torch.distributed.ProcessGroup, socket.socket, ctypes.CDLL, ctypes._Pointer,
                              ctypes._CFuncPtr, ctypes.Structure, ctypes.Array, ctypes._SimpleCData,
                              *self.compiler_resources, local, *self.sessions, *_LOCK_TYPES)):
            # Preparation sessions, allocators and peer storage retain process-local contexts, not forward
            # metadata; collective runtimes keep peer storage for teardown.
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if not isinstance(value, type) and (hasattr(type(value), '__del__')
                or type(value).__module__.startswith('cutlass._mlir.')):
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if isinstance(value, types.BuiltinMethodType) and isinstance(value.__self__, (*_LOCK_TYPES, local)):
            return _Unavailable, (f'{type(value.__self__).__qualname__}.{value.__name__}',)
        if isinstance(value, types.FunctionType) and (
                '<locals>' in value.__qualname__ or value.__name__ == '<lambda>'):
            return _Unavailable, (value.__qualname__,)
        if isinstance(value, types.ModuleType):
            import importlib
            return importlib.import_module, (value.__name__,)
        if (type(value).__module__.startswith(('b12x.', 'ignition.'))
                and hasattr(type(value), '__getattr__') and hasattr(value, '__dict__')):
            return _object, (type(value),), dict(vars(value)), None, None, _set_state
        return NotImplemented


class _ForwardUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('transformers_modules.'):
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        return super().find_class(module, name)


def dumps(forward):
    buffer = io.BytesIO()
    _ForwardPickler(buffer).dump(forward)
    return buffer.getvalue()


def loads(payload):
    token = _MODE.set(FakeTensorMode(allow_non_fake_inputs=True))
    _TENSOR_KINDS.set(WeakIdKeyDictionary())
    _TENSOR_KEYS.set(WeakIdKeyDictionary())
    try:
        return _ForwardUnpickler(io.BytesIO(payload)).load()
    finally:
        _MODE.reset(token)
