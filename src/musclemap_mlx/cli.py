"""Command line for MuscleMap v1.4: optional weights download, then segmentation.

Every subcommand prints one JSON object on stdout. Errors print one line on stderr
and exit 1; usage errors exit 2. medmlx.org reads this parser with ``ast`` to
check published commands, so each subcommand uses literal flags and choices.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit status."""
    parser = argparse.ArgumentParser(
        prog="musclemap-mlx",
        description="MuscleMap v1.4 whole-body segmentation with native MLX",
    )
    parser.add_argument("--version", action="version", version=version("musclemap-mlx"))
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download converted weights from the Hub")
    download.add_argument(
        "--repo",
        default="MedMLX/musclemap-mlx",
        help="Hugging Face repository (default: %(default)s)",
    )
    download.add_argument(
        "--revision", type=_commit, help="Override the reviewed 40-character commit"
    )
    download.add_argument(
        "--model-dir", type=Path, help="Destination folder; omit to use the cache"
    )

    segment = commands.add_parser("segment", help="Segment one 3D NIfTI volume")
    segment.add_argument("--input", required=True, type=Path, help="3D NIfTI image")
    segment.add_argument(
        "--model-dir", type=Path, help="Folder containing model.safetensors"
    )
    segment.add_argument("--output-dir", required=True, type=Path)
    segment.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Positive axial slices per independent pass; omit for automatic",
    )
    segment.add_argument(
        "--overlap",
        type=int,
        default=90,
        help="Gaussian slice-window overlap percentage in [0, 99] (default: %(default)s)",
    )
    segment.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = _download(args.repo, args.revision, args.model_dir)
        else:
            from musclemap_mlx.medmlx import run

            result = run(
                image_path=args.input,
                weights_path=args.model_dir,
                output_dir=args.output_dir,
                chunk_size=args.chunk_size,
                overlap_percent=args.overlap,
                overwrite=args.overwrite,
            )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"{parser.prog}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _commit(value: str) -> str:
    if not _COMMIT.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a full 40-character lowercase commit hash")
    return value


def _download(repo: str, revision: str | None, model_dir: Path | None) -> dict[str, object]:
    """Download one Hub commit into ``model_dir`` and record it for revision lookup."""
    from httpx import HTTPError
    from medmlx_core.errors import AssetNotReadyError
    from medmlx_core.weights import weights_dir, weights_revision

    if revision is None:
        if repo != "MedMLX/musclemap-mlx":
            raise ValueError("Custom repositories require --revision with a full reviewed commit")
        cached = weights_dir("musclemap")
        revision = weights_revision(cached)
        if revision is None:
            raise AssetNotReadyError("The completed MuscleMap cache has no revision")
        import shutil

        folder = cached if model_dir is None else model_dir.expanduser().absolute()
        if folder != cached:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".medmlx-revision").unlink(missing_ok=True)
            for source in cached.iterdir():
                if source.is_file() and not source.name.startswith("."):
                    shutil.copyfile(source, folder / source.name)
            (folder / ".medmlx-revision").write_text(revision + "\n", encoding="utf-8")
        return {"model_dir": folder.as_posix(), "repo": repo, "revision": revision}
    if model_dir is None:
        raise ValueError("An explicit --revision requires --model-dir")
    hub = import_module("huggingface_hub")  # installed by medmlx-core
    folder = model_dir.expanduser().absolute()
    marker = folder / ".medmlx-revision"
    marker.unlink(missing_ok=True)
    try:
        hub.snapshot_download(
            repo,
            revision=revision,
            local_dir=folder,
            allow_patterns=[
                "model.safetensors",
                "config.json",
                "contrast_agnostic_wholebody_model.json",
                "LICENSE",
                "NOTICE",
                "labels.json",
                "README.md",
                "conversion.json",
            ],
        )
    except (HTTPError, OSError) as error:
        raise AssetNotReadyError(
            f"Could not download {repo}@{revision}: {error}", hint="Rerun to resume."
        ) from error
    marker.write_text(revision + "\n", encoding="utf-8")
    return {"model_dir": folder.as_posix(), "repo": repo, "revision": revision}


if __name__ == "__main__":
    raise SystemExit(main())
