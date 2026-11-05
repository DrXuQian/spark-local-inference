# SPDX-License-Identifier: Apache-2.0

"""The two Dynamo artifact protocols used by Flare and platform Torch 2.9."""

import importlib

import torch

__all__ = ['TORCH29', 'SerializableCallable', 'deserialize_triton_kernel', 'serialize_triton_kernel', 'guard_locals']

TORCH29 = torch.__version__.split('+')[0].split('.')[:2] == ['2', '9']

if TORCH29:
    from torch._dynamo.aot_compile import SerializableCallable as SerializableCallable

    def serialize_triton_kernel(kernel):
        function = getattr(kernel, 'fn', None)
        module = getattr(function, '__module__', None)
        name = getattr(function, '__name__', None)
        if module is None or name is None:
            raise TypeError('a prepared Triton kernel must have an importable function')
        return module, name

    def deserialize_triton_kernel(identity):
        module, name = identity
        return getattr(importlib.import_module(module), name)
else:
    from torch._dynamo.aot_compile_types import (
        SerializableCallable as SerializableCallable,
        _deserialize_triton_kernel as deserialize_triton_kernel,
        _serialize_triton_kernel as serialize_triton_kernel,
    )


def guard_locals(artifact, *args):
    if TORCH29:
        from torch._dynamo.aot_compile import bind_locals

        return bind_locals(artifact._artifacts.signature, *args)
    return artifact.prepare_f_locals(*args)
