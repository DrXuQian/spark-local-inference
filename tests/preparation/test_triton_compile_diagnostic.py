# SPDX-License-Identifier: Apache-2.0

"""Diagnostic coverage uses real subprocesses, not PPU kernel execution."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from qualify.triton_compile import AssemblyLog, excerpt, kernel_source


def test_extracts_only_the_named_kernel_without_importing_the_model(tmp_path):
    source = tmp_path / 'generated.py'
    code = 'raise AssertionError("must not run while extracting")\n'
    source.write_text('raise AssertionError("must not import the model")\n'
        f'first = async_compile.triton("first", {code!r})\n'
        f'second = async_compile.triton("second", {code!r}, device_str="cuda")\n')
    assert kernel_source(source, 3) == ('second', code)
    with pytest.raises(ValueError, match='must identify one'):
        kernel_source(source, 1)
    source.write_text('x = async_compile.triton("first", get_source())\n')
    with pytest.raises(ValueError):
        kernel_source(source, 1)


@pytest.mark.parametrize('mode', ['success', 'nonzero', 'missing'])
def test_real_triton_assembler_path_retains_logs_and_preserves_failures(tmp_path, monkeypatch, mode):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler
    from triton.runtime.errors import PTXASError

    # Exercise real make_cubin and file cleanup with a small executable taking
    # the assembler's actual argv. This verifies diagnostics, not codegen.
    assembler = tmp_path / 'ptxas'
    assembler.write_text(f'#!{sys.executable}\n'
        'import sys\nfrom pathlib import Path\n'
        'print("frontend output")\n'
        'print("backend diagnostic", file=sys.stderr)\n'
        + ('Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"native binary")\n'
           if mode == 'success' else '')
        + ('sys.exit(9)\n' if mode == 'nonzero' else ''))
    assembler.chmod(0o755)
    monkeypatch.setattr(compiler, 'get_ptxas', lambda *args: SimpleNamespace(path=str(assembler)))
    directory = tmp_path / 'diagnostic'
    directory.mkdir()
    recorder = AssemblyLog(subprocess, directory)
    monkeypatch.setattr(compiler, 'subprocess', recorder)
    backend = compiler.CUDABackend(GPUTarget('cuda', 80, 32))
    options = backend.parse_options({})
    source = '// assembler input\n'
    if mode == 'success':
        assert backend.make_cubin(source, {}, options, 80) == b'native binary'
    else:
        with pytest.raises(PTXASError if mode == 'nonzero' else FileNotFoundError):
            backend.make_cubin(source, {}, options, 80)
    attempt = directory / 'assembler-001'
    assert recorder.count == 1
    assert (attempt / 'input.ptx').read_text() == source
    assert (attempt / 'stdout.txt').read_text() == 'frontend output\n'
    assert (attempt / 'stderr.txt').read_text() == 'backend diagnostic\n'
    status = (attempt / 'status.txt').read_text()
    assert f'returncode={9 if mode == "nonzero" else 0} ' in status
    assert f'artifact_present={mode == "success"}' in status
    command = (attempt / 'command.txt').read_text()
    assert str(assembler) in command and '--gpu-name=sm_80' in command
    # Triton removes temporary source/logs on the zero-exit path; our copies
    # must survive, including the reported missing-output failure.
    import shlex
    arguments = shlex.split(command)
    original = Path(next(arg for arg in arguments if arg.endswith('.ptx')))
    if mode != 'nonzero':
        assert not original.exists()
    original.unlink(missing_ok=True)


def test_subprocess_version_probe_and_failure_excerpt_are_bounded(tmp_path):
    recorder = AssemblyLog(subprocess, tmp_path)
    result = recorder.run([sys.executable, '-c', 'print("release 13.0")'], capture_output=True, check=True)
    assert result.stdout == b'release 13.0\n' and recorder.count == 0
    lines = excerpt('banner\nLLVM ERROR: unsupported instruction\n' + 'details\n' * 30)
    assert len(lines) == 12 and lines[0] == 'LLVM ERROR: unsupported instruction'


def test_ppu_translated_input_survives_temporary_file_cleanup(tmp_path):
    source = tmp_path / 'kernel.ptx.trans'
    source.write_bytes(b'target triple = "ppu"\n')
    recorder = AssemblyLog(subprocess, tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        recorder.run([sys.executable, '-c', 'import sys; sys.exit(7)',
                      str(source), '-o', str(tmp_path / 'kernel.ptx.o')], check=True)
    source.unlink()
    assert (tmp_path / 'assembler-001/input.ptx.trans').read_bytes() == b'target triple = "ppu"\n'
