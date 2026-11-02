"""Boundary checks for the development-intake CLI."""
import json
from pathlib import Path

import pytest

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
