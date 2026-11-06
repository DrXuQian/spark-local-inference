# SPDX-License-Identifier: Apache-2.0

"""Recompile one generated Inductor kernel and preserve assembler diagnostics.

Run on the preparation host against its generated Python cache file. Only the
selected kernel's literal source is loaded; the enclosing model is not imported.
This diagnostic uses Inductor's compile-worker path without launching kernels.
"""

import argparse
import ast
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback


def kernel_source(path, line):
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = [node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'async_compile'
        and node.func.attr == 'triton' and node.lineno <= line <= node.end_lineno]
    if len(matches) != 1 or len(matches[0].args) < 2:
        raise ValueError(f'{path}:{line} must identify one async_compile.triton call')
    name, source = (ast.literal_eval(node) for node in matches[0].args[:2])
    if not isinstance(name, str) or not name.isidentifier() or not isinstance(source, str):
        raise ValueError('the generated kernel name and source must be string literals')
    return name, source


class AssemblyLog:
    """Observe this backend's subprocess calls before it deletes temporary logs."""

    def __init__(self, original, directory):
        self.original, self.directory, self.count = original, directory, 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, command, **options):
        arguments = shlex.split(command) if isinstance(command, str) else list(map(str, command))
        if '-o' not in arguments:
            return self.original.run(command, **options)
        self.count += 1
        attempt = self.directory / f'assembler-{self.count:03}'
        attempt.mkdir()
        (attempt / 'command.txt').write_text(shlex.join(arguments) + '\n')
        working = Path(options.get('cwd') or Path.cwd())
        binary = working / arguments[arguments.index('-o') + 1]
        for argument in arguments:
            path = working / argument
            if path.suffix in ('.ptx', '.ll', '.bc', '.trans') and path.is_file():
                # PPU's formatter appends .trans to the temporary .ptx name.
                shutil.copyfile(path, attempt / ('input' + ''.join(path.suffixes)))
        result = None
        stdout = None
        # Triton redirects stderr into a temporary file but inherits stdout.
        # Preserve both, including a compiler wrapper that exits zero without
        # producing its requested binary. Do not change check or codegen flags.
        if options.get('stdout') is None and not options.get('capture_output'):
            stdout = (attempt / 'stdout.txt').open('wb')
            options = dict(options, stdout=stdout)
        try:
            result = self.original.run(command, **options)
            return result
        except subprocess.CalledProcessError as error:
            result = error
            raise
        finally:
            if stdout is not None:
                stdout.close()
            for name in ('stdout', 'stderr'):
                value = getattr(result, name, None)
                sink = options.get(name)
                filename = getattr(sink, 'name', None)
                if name == 'stdout' and stdout is not None:
                    continue
                if isinstance(filename, (str, Path)) and Path(filename).is_file():
                    shutil.copyfile(filename, attempt / f'{name}.txt')
                elif value is not None:
                    (attempt / f'{name}.txt').write_bytes(value.encode() if isinstance(value, str) else value)
            present = binary.is_file()
            (attempt / 'status.txt').write_text(
                f'returncode={getattr(result, "returncode", None)} '
                f'artifact_present={present} artifact_bytes={binary.stat().st_size if present else 0}\n')


def compile_kernel(directory, name):
    import torch
    import triton
    from triton import knobs
    from triton.backends.nvidia import compiler
    from torch._inductor.codecache import _load_triton_kernel_from_source
    from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path

    print(f'torch={torch.__version__} triton={triton.__version__}', flush=True)
    print(f'backend={compiler.__file__}', flush=True)
    _set_triton_ptxas_path()
    original = compiler.subprocess
    compiler.subprocess = AssemblyLog(original, directory)
    try:
        with knobs.compilation.scope():
            # Recompile the selected source without deleting shared caches.
            knobs.compilation.always_compile = True
            kernel = _load_triton_kernel_from_source(name, (directory / 'kernel.py').read_text())
            kernel.precompile(warm_cache_only=True)
    finally:
        compiler.subprocess = original


def excerpt(text, count=12):
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines)
        if re.search(r'error|fatal|cannot select|unsupported|not supported|failed', line, re.I)), 0)
    return [line[:300] for line in lines[start:start + count]]


def diagnose(source, line, output=None):
    name, code = kernel_source(source, line)
    if output is None:
        output = Path(tempfile.mkdtemp(prefix='flare-triton-compile-'))
    else:
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    (output / 'kernel.py').write_text(code)
    with (output / 'compile.log').open('wb') as log:
        result = subprocess.run([sys.executable, '-m', 'qualify.triton_compile',
            '_worker', str(output), name], stdout=log, stderr=subprocess.STDOUT)
    print(f'kernel={name}')
    print(f'compiler_process_returncode={result.returncode}')
    attempts = sorted(output.glob('assembler-*'))
    selected = next((path for path in attempts
        if not (path / 'status.txt').read_text().startswith('returncode=0 artifact_present=True ')),
        attempts[-1] if attempts else None)
    if selected is not None:
        print((selected / 'command.txt').read_text().strip())
        print((selected / 'status.txt').read_text().strip())
        diagnostic = '\n'.join(path.read_text(errors='replace') for name in ('stdout.txt', 'stderr.txt')
                               if (path := selected / name).exists())
        print('\n'.join(excerpt(diagnostic)))
    else:
        print('No assembler command observed; compilation stopped earlier or used another backend.')
        print('\n'.join(excerpt((output / 'compile.log').read_text(errors='replace'))))
    print(f'Full compiler log, source and assembler inputs: {output}')
    if result.returncode == 0:
        print('Compile-only completed; device execution and the full catalog remain untested.')
    return 0 if result.returncode == 0 else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        try:
            compile_kernel(Path(argv[1]), argv[2])
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Inductor-generated Python file from the traceback')
    parser.add_argument('--line', type=int, required=True, help='line of the failing async_compile.triton call')
    parser.add_argument('--output', type=Path, help='fresh diagnostic directory; default: a new temporary directory')
    args = parser.parse_args(argv)
    try:
        return diagnose(args.source, args.line, args.output)
    except (OSError, ValueError, SyntaxError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
