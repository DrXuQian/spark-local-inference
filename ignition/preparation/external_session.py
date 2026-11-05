# SPDX-License-Identifier: Apache-2.0

"""Finalize explicitly prepared external operators before compiling a catalog."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import get_ident

import torch

from .scratch import ScratchArena


@dataclass(frozen=True)
class _Namespace:
    values: dict

    def to_dict(self):
        return dict(self.values)


class ExternalPreparationSession:
    """One fixed QUactlize/actlizeLA/BLAS/FA3 catalog with caller-owned scratch.

    Native libraries prepare their own handles. This session finalizes the
    complete scratch inventory; ForwardCache still compiles, warms and freezes
    generated kernels before any graph is captured.
    """

    def __init__(self, *, device, checkpoint, namespace, compile_workers=None,
                 headroom_bytes=0, autotune_effort='default', lock_gpu_clocks=False):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError('external preparation requires a CUDA-compatible device and checkpoint identity')
        if autotune_effort not in ('off', 'default', 'audit'):
            raise ValueError('autotune effort must be off, default or audit')
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise ValueError('preparation headroom must be a nonnegative byte count')
        self.namespace = _Namespace(dict(namespace or {}, checkpoint=checkpoint))
        self.requested_compile_workers = compile_workers
        self.headroom_bytes = headroom_bytes
        self.autotune, self.lock_gpu_clocks = autotune_effort != 'off', lock_gpu_clocks
        self._preparation_thread = get_ident()
        self._scratch_arena = None
        self._capturing = False
        self.state = 'collecting'

    def _check_thread(self):
        if get_ident() != self._preparation_thread:
            raise RuntimeError('external preparation requires its owning thread')

    def register_scratch(self, arena):
        self._check_thread()
        if self.state != 'collecting' or self._scratch_arena is not None:
            raise RuntimeError('declare exactly one scratch arena before finalizing external operators')
        if not isinstance(arena, ScratchArena) or arena.device != self.device:
            raise ValueError('external operators must share their device scratch arena')
        self._scratch_arena = arena

    def tune(self, *, show_progress=False, console=None):
        self._check_thread()
        if self.state != 'collecting' or self._scratch_arena is None:
            raise RuntimeError('external operator finalization requires a declared, unfinished scratch arena')
        self.state = 'finalizing'
        arena = self._scratch_arena
        finalizers = len(arena._finalizers)
        try:
            if arena._parallel_plans:
                raise ValueError('external operators require explicitly disjoint scratch, not b12x parallel plans')
            arena.freeze()
            arena.finalize()
        except BaseException:
            self.state = 'failed'
            raise
        self.state = 'ready'
        return {'selection_counts': {}, 'external_binding_finalizers': finalizers,
                'scratch_bytes': arena.nbytes}

    @contextmanager
    def capture(self):
        self._check_thread()
        if self.state not in ('ready', 'frozen') or self._capturing:
            raise RuntimeError('capture requires finalized external operators and no active capture')
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False

    def freeze(self):
        self._check_thread()
        if self.state != 'ready' or self._capturing:
            raise RuntimeError('freeze requires completed external preparation outside capture')
        self.state = 'frozen'

    def close(self):
        self._check_thread()
        if self.state == 'closed':
            return
        if self._capturing:
            raise RuntimeError('cannot close external preparation during capture')
        try:
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.state = 'failed'
            raise
        if self._scratch_arena is not None:
            self._scratch_arena._finalizers.clear()
        self._scratch_arena = None
        self.dense_blas = None
        self.quactlize_helpers = self.actlize_helpers = self.flash_attention3 = None
        self.state = 'closed'

    def __enter__(self):
        self._check_thread()
        if self.state != 'collecting':
            raise RuntimeError('enter external preparation before finalization')
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f'external preparation cleanup failed: {error!r}')
