# SPDX-License-Identifier: Apache-2.0

"""External workspace survives the same trim/freeze/finalize used after tuning."""

import pytest
import torch

from ignition.preparation.scratch import ScratchArena


def test_external_views_survive_zero_tuned_scratch_and_rebind_after_trim():
    arena = ScratchArena('cpu', overlap=False)
    specs = (((3,), torch.int32), ((2, 7), torch.bfloat16))
    arena.require_views(*specs)
    arena.reserve(4096)  # Temporary candidate capacity is discarded after tuning.
    previous = arena.tensor
    views = []
    arena.register_finalizer(lambda: views.extend(arena.views(*specs)))
    arena.trim(0)
    assert arena.external_nbytes == arena.nbytes == 512
    assert arena.tensor is not previous
    arena.freeze()
    address = arena.tensor.data_ptr()
    arena.finalize()
    views[0].fill_(17); views[1].fill_(29)
    assert torch.all(views[0] == 17) and torch.all(views[1] == 29)
    assert [value.data_ptr() - address for value in views] == [0, 256]
    assert arena.tensor.data_ptr() == address
    with pytest.raises(RuntimeError, match='frozen'):
        arena.require_views(*specs)


def test_serial_requirements_take_maximum_and_legacy_trim_still_releases_storage():
    arena = ScratchArena('cpu')
    arena.require_views(((257,), torch.uint8))
    arena.require_views(((256,), torch.uint8))
    assert arena.external_nbytes == 512
    arena.require_views(((257,), torch.uint8), ((1,), torch.float32))
    assert arena.external_nbytes == 768
    arena.trim(0)
    assert arena.nbytes == 768
    legacy = ScratchArena('cpu')
    legacy.views(((1024,), torch.float32))
    legacy.trim(0)
    assert legacy.nbytes == legacy.external_nbytes == 0


@pytest.mark.parametrize('shape,dtype,error', [((-1,), torch.float32, ValueError),
    ((True,), torch.float32, ValueError), ((4,), 'float32', TypeError)])
def test_invalid_declarations_leave_capacity_unchanged(shape, dtype, error):
    arena = ScratchArena('cpu')
    with pytest.raises(error):
        arena.require_views((shape, dtype))
    assert arena.nbytes == arena.external_nbytes == 0


@pytest.mark.parametrize('target', ['ppu10', 'ppu15', 'cuda_sm120'])
def test_actual_actlize_workspace_specs_finalize_without_b12x(target):
    serving = pytest.importorskip('actlize_la.serving')
    arena = ScratchArena('cpu', overlap=False)
    retained = []
    for specs in (serving.gdn_workspace_specs(2, 8, mode='verify', target=target),
                  serving.gdn_workspace_specs(2, 17, mode='prefill', target=target),
                  serving.qsa_workspace_specs(8, 2, 24)):
        values = tuple(specs.values())
        arena.require_views(*values)
        arena.register_finalizer(lambda values=values: retained.append(arena.views(*values)))
    expected = arena.external_nbytes
    arena.trim(0); arena.freeze(); arena.finalize()
    assert arena.nbytes == expected and len(retained) == 3
