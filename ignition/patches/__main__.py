# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply bundled Inductor and Triton fixes in the selected Python environment."""

import argparse
import sys

from ignition.errors import PreparationError
from . import apply


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-target', choices=('cuda_sm120', 'ppu10', 'ppu15'),
                        help='select the matching dependency versions; default: locked NVIDIA environment')
    args = parser.parse_args(argv)
    try:
        changed = apply(operator_target=args.operator_target)
        for path in changed:
            print(f"Patched {path}")
        if not changed:
            print('Preparation patches already applied')
        return 0
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
