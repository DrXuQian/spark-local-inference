# SPDX-License-Identifier: Apache-2.0

"""Metadata workers do not import unused operator/compiler libraries."""

import ctypes
import io
import subprocess
import sys
from types import SimpleNamespace

import torch

from ignition.components.ppu_gguf_projection import PreparedPpuGgufProjection
from ignition.components.quactlize_helpers import PreparedQuactlizeHelpers
from ignition.preparation.aot import _GuardPickler, _guard_object
from ignition.preparation.forward_metadata import _Unavailable, dumps, loads
from ignition.preparation.package_storage import PackageStorage


def test_external_projection_metadata_keeps_aliases_and_excludes_native_owner():
    # Exercise the real metadata serializer without claiming device execution.
    projection = PreparedPpuGgufProjection.__new__(PreparedPpuGgufProjection)
    pool = torch.empty(4096, dtype=torch.uint8)
    projection.raw, projection.output = pool[:512], pool[1024:2048].view(torch.bfloat16).reshape(2, 256)
    projection.workspace = pool[2048:]
    projection.ids = torch.empty(2, 1, dtype=torch.int32)
    projection.device_error = torch.zeros((), dtype=torch.int32)
    projection.device_errors = (projection.device_error,)
    projection.handle, projection.owner = 123, ctypes.CDLL(None)
    helpers = PreparedQuactlizeHelpers.__new__(PreparedQuactlizeHelpers)
    helpers.handle, helpers.owner = 987, projection.owner
    storage = PackageStorage.__new__(PackageStorage)
    storage.callback = lambda: None
    restored = loads(dumps(SimpleNamespace(projection=projection, helpers=helpers, storage=storage)))
    assert 'owner' not in vars(restored.projection) and vars(restored.helpers) == {'handle': 987}
    assert restored.projection.handle == 123
    assert restored.projection.device_error is restored.projection.device_errors[0]
    assert isinstance(restored.storage, _Unavailable)
    for field in ('raw', 'output', 'workspace'):
        original, value = getattr(projection, field), getattr(restored.projection, field)
        assert (value.shape, value.stride(), value.storage_offset(), value.dtype) == (
            original.shape, original.stride(), original.storage_offset(), original.dtype)
    assert len({getattr(restored.projection, field).untyped_storage()._cdata
                for field in ('raw', 'output', 'workspace')}) == 1


def test_guard_pickler_keeps_external_handle_without_importing_b12x():
    helper = PreparedQuactlizeHelpers.__new__(PreparedQuactlizeHelpers)
    helper.handle, helper.owner = 9876543, ctypes.CDLL(None)
    pickler = _GuardPickler({id(helper.handle): helper.handle}, {}, {}, {}, io.BytesIO())
    restore, args, state = pickler.reducer_override(helper)
    assert restore is _guard_object and args == (PreparedQuactlizeHelpers,)
    assert state['handle'] == helper.handle
    assert type(state['owner']).__name__ == '_Missing'


def test_fresh_metadata_process_never_imports_b12x_cute_or_peer_storage():
    code = '''
import ctypes, importlib.abc, sys, torch
class RejectUnused(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('b12x', 'cutlass', 'cuda.tile', 'ignition.preparation.peer_storage')):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, RejectUnused())
from ignition.preparation.forward_metadata import dumps, loads, _Unavailable
from ignition.preparation.aot import _GuardPickler
from ignition.components.quactlize_helpers import PreparedQuactlizeHelpers
import io
from ignition.recipes.qwen38.attention import QSAScratch
from ignition.recipes.qwen38.mtp import MTPScratch
from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader
from ignition.recipes.qwen38.mtp_weights import MTPWeights
from ignition.recipes.qwen38.decoder import ModelWeights, DecoderScratch
qsa = QSAScratch.allocate(17, 'meta')
mtp = MTPScratch.allocate(17, 2, 'meta')
assert qsa.projected_qkv.shape == (17, 13312)
assert mtp.residual.shape == (17, 10240) and mtp.logits.shape == (2, 248320)
tensor = torch.empty(3, 11)
values = loads(dumps(dict(a=tensor, b=tensor[1:, 3:], native=ctypes.CDLL(None))))
assert values['a'].untyped_storage()._cdata == values['b'].untyped_storage()._cdata
assert values['b'].stride() == (11, 1) and values['b'].storage_offset() == 14
assert isinstance(values['native'], _Unavailable)
helper = PreparedQuactlizeHelpers.__new__(PreparedQuactlizeHelpers)
helper.handle = 567890
result = _GuardPickler({id(helper.handle): helper.handle}, {}, {}, {}, io.BytesIO()).reducer_override(helper)
assert result[2]['handle'] == helper.handle
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
