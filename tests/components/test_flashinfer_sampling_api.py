# SPDX-License-Identifier: Apache-2.0

"""Reject incomplete operator packages before allocating preparation scratch."""

import sys
from types import ModuleType

import pytest
import torch

from ignition.components.flashinfer_sampling import probability_filter_workspace, require_prepared_sampling_api


def test_old_sampling_package_reports_both_missing_interfaces_without_allocation(monkeypatch):
    package = ModuleType('flashinfer')
    package.sampling = ModuleType('flashinfer.sampling')
    package.sampling.chain_speculative_sampling = lambda draft, ids, target: None
    monkeypatch.setitem(sys.modules, 'flashinfer', package)
    def forbidden(*args, **kwargs):
        pytest.fail('incompatible sampling API must be rejected before device allocation')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(RuntimeError) as error:
        probability_filter_workspace(4, 248320, 'cuda:0')
    assert '_top_p_renorm_workspace_size' in str(error.value)
    assert 'row_independent_rng' in str(error.value)


def test_api_inspection_does_not_compile_or_load_a_device(monkeypatch):
    package = ModuleType('flashinfer')
    sampling = package.sampling = ModuleType('flashinfer.sampling')
    sampling._top_p_renorm_workspace_size = lambda rows, vocabulary, deterministic: 256
    sampling.chain_speculative_sampling = lambda draft, ids, target, row_independent_rng=False: None
    def forbidden(*args, **kwargs):
        pytest.fail('API inspection must leave native preparation to the preparation batch')
    sampling.get_sampling_module = forbidden
    monkeypatch.setitem(sys.modules, 'flashinfer', package)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    assert require_prepared_sampling_api() is sampling
