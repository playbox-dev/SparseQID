# SPDX-License-Identifier: Apache-2.0

"""Unified SparseQID command-line interface."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
    "extract": "build the JPEG frame cache",
    "train": "train the identity model",
    "infer": "run tracking inference",
    "visualize": "render tracked boxes as MP4 clips",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sqid", description=__doc__)
    commands = result.add_subparsers(dest="command", metavar="COMMAND")
    for name, description in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    if not arguments:
        root.print_help()
        return
    namespace, remaining = root.parse_known_args(arguments)
    if namespace.command is None:
        root.parse_args(arguments)
        return
    module = importlib.import_module(f".{namespace.command}", __package__)
    module.main(remaining, prog=f"sqid {namespace.command}")


if __name__ == "__main__":
    main()
