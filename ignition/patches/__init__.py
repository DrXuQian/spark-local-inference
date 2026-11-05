# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the pinned PyTorch Inductor and Triton fixes before preparation imports."""

import os
import stat
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files
from pathlib import Path

from ignition.errors import PreparationError

_VERSIONS = {"torch": "2.14.1+cu132", "triton": "3.8.0"}
_PATCHES = (
    "torch-inductor-triton-cache-filenames.patch",
    "torch-inductor-cooperative-static-launch.patch",
    "triton-platform-key-pointer-width.patch",
)
_PPU_VERSIONS = {"torch": "2.9.0", "triton": "3.5.0"}
_PPU_PATCHES = (
    "torch29-inductor-triton-cache-filenames.patch",
    "triton-platform-key-pointer-width.patch",
)


def _hunks(patch):
    """Read file names and exact replacement blocks from the bundled unified diffs."""
    path = None
    before = after = None
    for line in patch.splitlines(keepends=True):
        if line.startswith(("diff --git ", "@@ ")):
            if before is not None:
                yield path, "".join(before), "".join(after)
            before = after = None
            if line.startswith("diff --git "):
                _, _, source, target = line.split()
                allowed = target.startswith("b/torch/_inductor/") or target == "b/triton/runtime/build.py"
                if source[2:] != target[2:] or not allowed:
                    raise PreparationError(
                        "Preparation patches must modify existing torch/_inductor files or triton/runtime/build.py")
                path = target[2:]
            else:
                before, after = [], []
        elif before is not None:
            if line.startswith((" ", "-")):
                before.append(line[1:])
            if line.startswith((" ", "+")):
                after.append(line[1:])
    if before is not None:
        yield path, "".join(before), "".join(after)


def apply(*, operator_target=None) -> tuple[Path, ...]:
    """Validate bundled patches, then atomically replace changed installation files.

    Matching uses complete hunk context without fuzz. Already applied
    hunks are accepted, including installations with only one patch applied.
    Replacing files breaks wheel-cache hardlinks instead of modifying shared bytes.
    """
    if operator_target not in (None, 'cuda_sm120', 'ppu10', 'ppu15'):
        raise PreparationError('unknown preparation operator target')
    ppu = operator_target in ('ppu10', 'ppu15')
    versions, selected_patches = (_PPU_VERSIONS, _PPU_PATCHES) if ppu else (_VERSIONS, _PATCHES)
    installation = ('Keep the platform Torch2.9.0/Triton3.5.0 artifacts; do not install the NVIDIA prepare extra.'
                    if ppu else 'Install the locked preparation dependencies with uv sync --locked --extra prepare.')
    try:
        packages = {}
        for name, expected in versions.items():
            try:
                package = distribution(name)
            except PackageNotFoundError:
                if not ppu and name == "triton" and sys.platform == "win32":
                    continue
                raise
            version = package.version.partition('+')[0] if ppu else package.version
            if version != expected:
                raise PreparationError(
                    f"Preparation patches require {name}=={expected}; found {package.version}. "
                    + installation)
            packages[name] = package

        sources, patched, modules = {}, {}, {}
        for name in selected_patches:
            for relative, before, after in _hunks(files(__package__).joinpath(name).read_text(encoding="utf-8")):
                package = relative.split("/", 1)[0]
                if package not in packages:
                    continue
                path = Path(packages[package].locate_file(relative))
                modules[path] = relative.removesuffix(".py").replace("/", ".")
                if path not in sources:
                    sources[path] = path.read_text(encoding="utf-8")
                    patched[path] = sources[path]
                source = patched[path]
                counts = source.count(before), source.count(after)
                if counts == (1, 0):
                    patched[path] = source.replace(before, after, 1)
                elif counts != (0, 1):
                    remedy = ('Platform source differs from the inspected patch context; preserve the installed artifacts.'
                              if ppu else f'Reinstall {package} with uv sync --locked --extra prepare --reinstall-package {package}.')
                    raise PreparationError(
                        f"Cannot apply {name} to {path}: patch context is missing or ambiguous. "
                        + remedy)

        changed = tuple(path for path in sources if patched[path] != sources[path])
        for path in changed:
            module = modules[path]
            if module in sys.modules:
                raise PreparationError(f"Cannot patch imported module {module}; restart Python before preparation.")
        for path in changed:
            # Keep replacement files beside their targets for atomic publication on one filesystem.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    output.write(patched[path].encode("utf-8"))
                    output.close()
                    temporary.chmod(stat.S_IMODE(path.stat().st_mode))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return changed
    except PackageNotFoundError as error:
        raise PreparationError(
            "Preparation patches require the preparation dependencies; "
            + installation) from error
    except OSError as error:
        raise PreparationError(f"Cannot apply preparation patches in the preparation environment: {error}") from error
