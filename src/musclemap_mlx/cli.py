"""Local asset inspection CLI; no model execution or automatic downloads."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from musclemap_mlx.assets import inspect_assets


def main(argv: Sequence[str] | None = None) -> int:
    """Print JSON on success and an actionable typed error on failure."""
    parser = argparse.ArgumentParser(
        prog="musclemap-mlx", description="MuscleMap v1.4 native MLX development intake"
    )
    parser.add_argument("--version", action="version", version=version("musclemap-mlx"))
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-assets", help="Verify downloaded publisher assets")
    inspect.add_argument("--model-dir", type=Path, required=True)
    commands.add_parser("status", help="Report native implementation status")
    args = parser.parse_args(argv)
    try:
        result = (
            inspect_assets(args.model_dir)
            if args.command == "inspect-assets"
            else {
                "native_inference": "unimplemented",
                "medmlx_registered": False,
                "weights_converted": False,
            }
        )
    except (OSError, ValueError) as error:
        print(f"{parser.prog}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
