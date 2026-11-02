"""Boundary checks for the development-intake CLI."""

import hashlib
import json
from pathlib import Path

import pytest

from musclemap_mlx import assets
from musclemap_mlx.assets import ASSETS, inspect_assets
from musclemap_mlx.cli import main


def test_status_does_not_advertise_inference(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["native_inference"] == "unimplemented"
    assert result["medmlx_registered"] is False


def test_missing_assets_are_actionable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect-assets", "--model-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Zenodo record 21929873" in captured.err


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    name = next(iter(ASSETS))
    (tmp_path / name).write_bytes(b"partial")
    with pytest.raises(ValueError, match="Unexpected size"):
        inspect_assets(tmp_path)


def test_no_inference_command_is_advertised() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["segment"])
    assert caught.value.code == 2


@pytest.fixture
def tiny_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Supply tiny publisher records while retaining real file IO and hashing."""
    payloads = {
        "task.pth": b"abc",
        assets.CONFIG: b'{"labels": {"0": "background", "7": "muscle"}}',
    }
    records: dict[str, tuple[int, str]] = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        records[name] = (len(payload), hashlib.md5(payload, usedforsecurity=False).hexdigest())
    monkeypatch.setattr(assets, "ASSETS", records)
    return tmp_path


def test_verified_assets_preserve_original_labels(tiny_assets: Path) -> None:
    assert inspect_assets(tiny_assets) == {
        "status": "assets-verified-native-inference-unimplemented",
        "record": "https://zenodo.org/records/21929873",
        "configuration": {"labels": {"0": "background", "7": "muscle"}},
        "checkpoint_deserialized": False,
    }


def test_same_size_corrupt_asset_is_rejected(tiny_assets: Path) -> None:
    (tiny_assets / "task.pth").write_bytes(b"xyz")
    with pytest.raises(ValueError, match=r"Publisher MD5 mismatch for task\.pth"):
        inspect_assets(tiny_assets)
