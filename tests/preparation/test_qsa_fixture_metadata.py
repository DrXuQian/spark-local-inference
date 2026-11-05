# SPDX-License-Identifier: Apache-2.0

"""Exercise fixture geometry up to native binding without running a device."""

import pytest
import torch

pytest.importorskip('actlize_la.serving')

from qualify.actlize_qsa_export import Fixture


@pytest.mark.parametrize('rows,requests', [(1, 1), (7, 2), (128, 2)])
def test_fixture_capacities_reach_the_native_boundary(monkeypatch, rows, requests):
    from actlize_la.serving import flare

    class ReachedNativeBoundary(Exception):
        pass

    def inspect_binding(library, **values):
        assert library is None
        metadata, workspace = values['metadata'], values['workspace']
        assert values['query'].shape == (rows, 24, 256)
        assert metadata.main_block_table.shape[0] == requests
        assert metadata.query_start_loc.shape == (requests + 1,)
        assert metadata.raw_state_slot_ids.shape == (requests,)
        assert workspace['errors'].shape == (requests,)
        assert workspace['selected'].shape == (rows, 2051)
        raise ReachedNativeBoundary

    # Stop only at the unexecuted native boundary. Actual Flare metadata
    # allocation/validation and library workspace geometry run unchanged.
    monkeypatch.setattr(flare, 'bind_qsa', inspect_binding)
    with pytest.raises(ReachedNativeBoundary):
        Fixture(None, rows=rows, device='cpu')


@pytest.mark.parametrize('rows,requests', [(1, 1), (7, 2), (128, 2)])
def test_interval_metadata_fills_exact_request_capacity(rows, requests):
    from ignition.recipes.qwen38.attention_state import QSAMetadata

    f = Fixture.__new__(Fixture)
    f.requests, f.state = requests, None
    f.metadata = QSAMetadata.allocate(tokens=rows, batch=requests, context=256, device='cpu')
    f.live = torch.empty(1, dtype=torch.int32)
    f.perm = torch.tensor([1, 0, 3, 2])
    f.inputs = {name: torch.empty(rows, 4) for name in ('query', 'key', 'value', 'index_query', 'raw_index_key')}
    f.data = {name: torch.randn(256, 4) for name in ('query', 'key', 'value', 'index_query', 'raw')}
    f.positions = torch.arange(256)[:, None].expand(-1, 3)
    for width in (rows, 0, 1):
        f.interval(63, width, 1)
        assert f.metadata.query_start_loc.tolist() == [0] + [width] * requests
        assert f.metadata.sequence_lengths.tolist() == [63 + width] + [0] * (requests - 1)
        assert f.metadata.num_accepted_tokens.tolist() == [1] + [0] * (requests - 1)
        assert f.metadata.raw_state_slot_ids.tolist() == [1] + [-1] * (requests - 1)
        assert f.live.item() == (requests if width else 0)
        assert f.metadata.request_ids.tolist() == [0] * width + [-1] * (rows - width)
