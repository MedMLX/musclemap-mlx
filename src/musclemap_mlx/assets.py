"""Inspect publisher assets without deserializing the checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

ASSETS = {
    "contrast_agnostic_wholebody_model.pth": (104884148, "910b722aeb641c380404c99ec6d1af97"),
    "contrast_agnostic_wholebody_model.json": (10555, "b586ac488b2e40a4e8624a9a1c52d6b5"),
    "LICENSE": (1087, "e388417c03884fc1359c86602aa735f4"),
}
CONFIG = "contrast_agnostic_wholebody_model.json"


def inspect_assets(folder: Path) -> dict[str, object]:
    """Verify publisher bytes and return the original configuration as JSON data."""
    for name, (size, expected) in ASSETS.items():
        path = folder / name
        if not path.is_file():
            raise ValueError(f"Missing asset: {path}; obtain it from Zenodo record 21929873")
        if path.stat().st_size != size:
            raise ValueError(f"Unexpected size for {name}; expected {size} bytes")
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Publisher MD5 mismatch for {name}")
    config = cast(object, json.loads((folder / CONFIG).read_text()))
    return {
        "status": "assets-verified-native-inference-unimplemented",
        "record": "https://zenodo.org/records/21929873",
        "configuration": config,
        "checkpoint_deserialized": False,
    }
