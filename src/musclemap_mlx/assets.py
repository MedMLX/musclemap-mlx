"""Verify the publisher assets without deserializing executable checkpoint data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from musclemap_mlx.config import CONFIG_NAME, MODEL_ID

ASSETS = {
    "contrast_agnostic_wholebody_model.pth": (104884148, "910b722aeb641c380404c99ec6d1af97"),
    "contrast_agnostic_wholebody_model.json": (10555, "b586ac488b2e40a4e8624a9a1c52d6b5"),
    "LICENSE": (1087, "e388417c03884fc1359c86602aa735f4"),
}
SHA256 = {
    "contrast_agnostic_wholebody_model.pth": (
        "45dfa2843d2e0b1fd842152d6a79bfd4bcb90899c076ceb6346c59da9a79a16c"
    ),
    "contrast_agnostic_wholebody_model.json": (
        "82c74f854ab74d8770e6e2b9a240bd23ccc646a7b71b13e6cfd8f9919d2acb26"
    ),
    "LICENSE": "89ce1faf8c253b24fd549b2f2c3f4a74b0aaa89ba442cd23a066f3e6248b2758",
}


def verify_assets(folder: Path) -> dict[str, str]:
    """Verify publisher sizes and digests and return each asset's absolute path.

    Raises:
        ValueError: An asset is missing or its bytes differ from Zenodo record
            21929873.
    """
    paths: dict[str, str] = {}
    for name, (size, expected_md5) in ASSETS.items():
        path = folder / name
        if not path.is_file():
            raise ValueError(f"Missing asset: {path}; obtain it from Zenodo record 21929873")
        if path.stat().st_size != size:
            raise ValueError(f"Unexpected size for {name}; expected {size} bytes")
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                md5.update(chunk)
                sha256.update(chunk)
        if md5.hexdigest() != expected_md5:
            raise ValueError(f"Publisher MD5 mismatch for {name}")
        if sha256.hexdigest() != SHA256[name]:
            raise ValueError(f"Publisher SHA-256 mismatch for {name}")
        paths[name] = path.absolute().as_posix()
    return paths


def inspect_assets(folder: Path) -> dict[str, object]:
    """Verify publisher bytes and return the original configuration as JSON data."""
    verify_assets(folder)
    config = cast(object, json.loads((folder / CONFIG_NAME).read_text(encoding="utf-8")))
    return {
        "status": "assets-verified",
        "model_id": MODEL_ID,
        "record": "https://zenodo.org/records/21929873",
        "configuration": config,
        "checkpoint_deserialized": False,
    }
