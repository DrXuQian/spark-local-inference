# SPDX-License-Identifier: Apache-2.0

"""The GGUF export consumer drains prepared graphs even when package writing fails."""

import pytest

from tests.recipes.test_qwen38_bootstrap import model_fixture


@pytest.mark.parametrize('fails', [False, True])
def test_fixture_export_closes_prepared_owners_on_success_and_error(monkeypatch, tmp_path, fails):
    from ignition.preparation import runtime_package
    from qualify.qwen38_gguf import export_fixture

    model, events, _ = model_fixture(monkeypatch)
    images, storage, publisher = object(), object(), object()
    output = tmp_path / 'package'
    failure = RuntimeError('injected package metadata failure')

    def write(owner, recorder, pools, path, **kwargs):
        assert owner is model and not owner.closed and owner.graphs and owner.resources
        assert recorder is images and pools is storage and path == output
        assert kwargs == dict(options={}, publisher=publisher)
        events.append('write')
        if fails:
            raise failure

    monkeypatch.setattr(runtime_package, 'export_model', write)
    if fails:
        with pytest.raises(RuntimeError) as caught:
            export_fixture(model, images, storage, output, publisher=publisher)
        assert caught.value is failure
    else:
        export_fixture(model, images, storage, output, publisher=publisher)
    assert events == ['write', 'synchronize', 'graph', 'session', 'table']
    assert model.closed and not model.resources and not model.graphs
