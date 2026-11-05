# SPDX-License-Identifier: Apache-2.0

"""The box runner must not treat skipped or empty device suites as success."""

import pytest

from qualify.ppu_box import pytest_result


@pytest.mark.parametrize('tests,failures,errors,skipped,passed', [
    (4, 0, 0, 0, True), (0, 0, 0, 0, False), (4, 0, 0, 4, False),
    (4, 1, 0, 0, False), (4, 0, 1, 0, False), (4, 0, 0, 1, False)])
def test_device_result_requires_executed_passing_cases(tmp_path, tests, failures, errors, skipped, passed):
    path = tmp_path / 'pytest.xml'
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
                    f'errors="{errors}" skipped="{skipped}"/></testsuites>')
    counts, actual = pytest_result(path)
    assert actual is passed and counts['tests'] == tests
