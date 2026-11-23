"""Command line for MuscleMap v1.4: optional weights download, then segmentation.

Every subcommand prints one JSON object on stdout. Errors print one line on stderr
and exit 1; usage errors exit 2. The ``segment`` options are generated from the
``run`` signature and its docstring.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from medmlx_core.cli import add_options, call, execute

_COMMIT = re.compile(r"[0-9a-f]{40}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit status."""
    from musclemap_mlx.medmlx import run

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
    add_options(segment, run)

    args = parser.parse_args(argv)
    if args.command == "download":
        return execute(parser.prog, lambda: _download(args.repo, args.revision, args.model_dir))
    return execute(parser.prog, lambda: call(run, args))


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
