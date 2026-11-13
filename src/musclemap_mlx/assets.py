"""Check publisher asset completeness without deserializing checkpoint data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from musclemap_mlx.config import CONFIG_NAME, MODEL_ID

ASSETS = {
    "contrast_agnostic_wholebody_model.pth": 104884148,
    "contrast_agnostic_wholebody_model.json": 10555,
    "LICENSE": 1087,
}


def verify_assets(folder: Path) -> dict[str, str]:
    """Check publisher file sizes and return each asset's absolute path.

    Raises:
        ValueError: An asset is missing or incomplete for Zenodo record 21929873.
    """
    paths: dict[str, str] = {}
    for name, size in ASSETS.items():
        path = folder / name
        if not path.is_file():
            raise ValueError(f"Missing asset: {path}; obtain it from Zenodo record 21929873")
        if path.stat().st_size != size:
            raise ValueError(f"Unexpected size for {name}; expected {size} bytes")
        paths[name] = path.absolute().as_posix()
    return paths


def inspect_assets(folder: Path) -> dict[str, object]:
    """Check asset completeness and return the original configuration as JSON data."""
    verify_assets(folder)
    config = cast(object, json.loads((folder / CONFIG_NAME).read_text(encoding="utf-8")))
    return {
        "status": "assets-complete",
        "model_id": MODEL_ID,
        "record": "https://zenodo.org/records/21929873",
        "configuration": config,
        "checkpoint_deserialized": False,
    }
