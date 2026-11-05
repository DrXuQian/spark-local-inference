# SPDX-License-Identifier: Apache-2.0

"""The box runner must not treat skipped or empty device suites as success."""

import pytest

from qualify.ppu_box import parser, pytest_result, resolve_sources, device_suites


@pytest.mark.parametrize('tests,failures,errors,skipped,passed', [
    (4, 0, 0, 0, True), (0, 0, 0, 0, False), (4, 0, 0, 4, False),
    (4, 1, 0, 0, False), (4, 0, 1, 0, False), (4, 0, 0, 1, False)])
def test_device_result_requires_executed_passing_cases(tmp_path, tests, failures, errors, skipped, passed):
    path = tmp_path / 'pytest.xml'
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
                    f'errors="{errors}" skipped="{skipped}"/></testsuites>')
    counts, actual = pytest_result(path)
    assert actual is passed and counts['tests'] == tests


def test_installed_wheel_runner_needs_no_quactlize_checkout_or_manual_dso(tmp_path):
    from pathlib import Path

    (tmp_path / 'actlize_la/serving').mkdir(parents=True)
    library = tmp_path / 'operator.so'
    library.touch()
    args = parser().parse_args(['--actlize-source', str(tmp_path), '--actlize-library', str(library),
        '--actlize-target', 'ppu10', '--blas-library', str(library)])
    assert resolve_sources(args) == (tmp_path,)
    suites = device_suites(args, Path.cwd(), tmp_path / 'results')
    names = {suite[0] for suite in suites}
    assert {'framework', 'gguf-packed', 'gguf-expert-chain', 'gguf-read', 'compiled-catalog'} <= names
    assert 'quactlize' not in names
    args.quactlize_source = tmp_path
    with pytest.raises(ValueError, match='requires --quactlize-library'):
        resolve_sources(args)
    (tmp_path / 'quactlize').mkdir()
    (tmp_path / 'quactlize/serving.py').touch()
    args.quactlize_library = library
    assert len(resolve_sources(args)) == 2
    assert 'quactlize' in {suite[0] for suite in device_suites(args, Path.cwd(), tmp_path / 'results')}


def test_both_installed_packages_keep_framework_gates_without_operator_sources(tmp_path):
    from pathlib import Path

    library = tmp_path / 'blas.so'
    library.touch()
    args = parser().parse_args(['--actlize-target', 'ppu10', '--blas-library', str(library),
                               '--blas-api', 'cublas'])
    assert resolve_sources(args) == ()
    suites = device_suites(args, Path.cwd(), tmp_path / 'results')
    assert {'framework', 'gguf-packed', 'gguf-expert-chain', 'compiled-catalog', 'qsa', 'sampling'} <= {s[0] for s in suites}
    assert not {'quactlize', 'actlize', 'actlize-aux'} & {s[0] for s in suites}
    assert 'tests/components/test_actlize_gdn_sequence_cuda.py' in suites[0][1]
    assert 'tests/components/test_actlize_mixing_cuda.py' in suites[0][1]
    assert 'tests/components/test_actlize_ple_cuda.py' in suites[0][1]
    args.actlize_source = tmp_path
    with pytest.raises(ValueError, match='--actlize-source requires --actlize-library'):
        resolve_sources(args)
    (tmp_path / 'actlize_la/serving').mkdir(parents=True)
    args.actlize_library = library
    assert resolve_sources(args) == (tmp_path,)
    assert {'actlize', 'actlize-aux'} <= {s[0] for s in device_suites(args, Path.cwd(), tmp_path / 'results')}
