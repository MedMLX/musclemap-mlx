"""Publisher asset verification without deserializing the checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from musclemap_mlx import assets


def _records(folder: Path, payloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    records: dict[str, int] = {}
    for name, payload in payloads.items():
        (folder / name).write_bytes(payload)
        records[name] = len(payload)
    monkeypatch.setattr(assets, "ASSETS", records)


def test_missing_asset_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Zenodo record 21929873"):
        assets.verify_assets(tmp_path)


def test_incomplete_asset_is_rejected(tmp_path: Path) -> None:
    name = next(iter(assets.ASSETS))
    (tmp_path / name).write_bytes(b"partial")
    with pytest.raises(ValueError, match="Unexpected size"):
        assets.verify_assets(tmp_path)


def test_tiny_verified_assets_keep_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records(
        tmp_path,
        {
            "task.pth": b"abc",
            assets.CONFIG_NAME: b'{"labels": {"0": "background", "7": "muscle"}}',
            "LICENSE": b"mit",
        },
        monkeypatch,
    )
    result = assets.inspect_assets(tmp_path)
    assert result["status"] == "assets-complete"
    assert result["checkpoint_deserialized"] is False
    assert result["configuration"] == {"labels": {"0": "background", "7": "muscle"}}
