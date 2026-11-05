# SPDX-License-Identifier: Apache-2.0

"""Torch 2.9 compiler artifacts for fixed buffers and guarded forward catalogs."""

import copy
import pickle
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class StandaloneArtifact:
    def __init__(self, artifact, serialized=None):
        self.artifact = artifact
        self.serialized = serialized

    @classmethod
    def compile(cls, module, inputs, options):
        from torch._inductor import standalone_compile

        if torch.__version__.split('+')[0].split('.')[:2] != ['2', '9']:
            raise RuntimeError('this standalone artifact adapter requires platform Torch 2.9')
        return cls(standalone_compile(module, inputs, dynamic_shapes='from_example_inputs',
                                     options={'config_patches': dict(options)}))

    @classmethod
    def load(cls, data):
        from torch._inductor.standalone_compile import CompiledArtifact

        with TemporaryDirectory(prefix='flare-torch29-') as directory:
            path = Path(directory) / 'forward.bin'
            path.write_bytes(data)
            artifact = CompiledArtifact.load(path=str(path))
        return cls(artifact, zlib.compress(data, level=1))

    def serialize(self):
        if self.serialized is None:
            with TemporaryDirectory(prefix='flare-torch29-') as directory:
                path = Path(directory) / 'forward.bin'
                self.artifact.save(path=str(path))
                self.serialized = zlib.compress(path.read_bytes(), level=1)
            # The callable retains live kernels; keep only one compressed copy
            # of the portable cache payload after publication.
            self.artifact._artifacts = None
        return zlib.decompress(self.serialized)

    def __call__(self, *inputs):
        return self.artifact(*inputs)


class BundledArtifact:
    """Keep the AOT entry editable until the catalog freezes its Triton winners."""

    def __init__(self, entry, function):
        self.entry = entry
        self.function = function

    @classmethod
    def compile(cls, module, inputs, options):
        from torch._functorch import config
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache, BundledAOTAutogradCacheEntry
        from torch._inductor import standalone_compile

        with config.patch(bundled_autograd_cache=True, strict_autograd_cache=True):
            artifact = standalone_compile(module, inputs, dynamic_shapes='from_tracing_context',
                                          options={'config_patches': dict(options)})
            if artifact._artifacts is None:
                raise RuntimeError('Torch 2.9 did not retain the compiled forward artifact')
            _, inventory = artifact._artifacts
            key, = inventory.aot_autograd_artifacts
            # Dynamo and StorageBackend retain the guards. This is the same
            # cache lookup used by CompiledArtifact.load, before binding inputs.
            with torch._inductor.config.patch(unsafe_skip_cache_dynamic_shape_guards=True):
                entry = AOTAutogradCache._lookup(key, local=True, remote=False, args=[],
                                                cache_info={}, aot_config=None)
        if not isinstance(entry, BundledAOTAutogradCacheEntry) or entry.compiled_bw is not None:
            raise RuntimeError('Torch 2.9 did not produce a bundled inference forward')
        artifact._artifacts = None
        return cls(entry, artifact)

    @classmethod
    def load_entry(cls, entry):
        from torch._guards import TracingContext, tracing
        from torch._inductor.cudagraph_utils import BoxedDeviceIndex
        from torch._inductor.utils import BoxedBool
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # Torch's BundledAOTAutogradCacheArtifact delegates guards to Dynamo.
        # StorageBackend additionally checks exact views and storage aliases.
        entry.guards_expr = None
        with tracing(TracingContext(FakeTensorMode(shape_env=ShapeEnv()))):
            function = entry.wrap_post_compile([], entry.sanitized_aot_config,
                {'cudagraphs': BoxedBool(False), 'boxed_forward_device_index': BoxedDeviceIndex(None)})
        result = cls(entry, lambda *args: function(list(args)))
        result._flare_graph = entry.compiled_fw.result
        return result

    def serialize(self):
        entry = copy.copy(self.entry)
        entry.compiled_fw = copy.copy(entry.compiled_fw)
        entry.pre_save()
        # Match the newer artifact envelope so winner freezing, relocation and
        # runtime kernel-table validation have one representation.
        return pickle.dumps((entry, {}, {}))

    def __call__(self, *inputs):
        return self.function(*inputs)
