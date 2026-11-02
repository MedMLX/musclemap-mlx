"""Console-script contract for the download and segment subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musclemap_mlx.cli import main
from musclemap_mlx.geometry import nib as nifti_io


def _image(path: Path) -> None:
    nifti_io.save(
        nifti_io.Nifti1Image(np.ones((8, 9, 2), dtype=np.float32), np.eye(4)),
        str(path),
    )


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert "segment" in capsys.readouterr().out


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip()


def test_missing_segment_arguments_exit_two() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["segment"])
    assert caught.value.code == 2


def test_unknown_status_command_is_not_advertised() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["status"])
    assert caught.value.code == 2


def test_short_revision_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["download", "--revision", "abc", "--model-dir", str(tmp_path)])
    assert caught.value.code == 2


def test_segment_missing_input_reports_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "segment",
            "--input",
            str(tmp_path / "missing.nii.gz"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "InvalidInputError" in captured.err
    assert "does not exist" in captured.err


def test_segment_without_weights_reports_asset_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)
    code = main(
        [
            "segment",
            "--input",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "AssetNotReadyError" in captured.err
    assert "converted weights" in captured.err


def test_download_rejects_unpinned_revision_before_network(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["download", "--revision", "0" * 39, "--model-dir", str(tmp_path)])
    assert caught.value.code == 2


def test_segment_result_is_json_when_stubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)

    def fake_run(**_kwargs: object) -> dict[str, object]:
        return {"outputs": {"mask": "/tmp/mask.nii.gz"}, "details": {"requires_human_review": True}}

    monkeypatch.setattr("musclemap_mlx.medmlx.run", fake_run)
    code = main(
        [
            "segment",
            "--input",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["details"]["requires_human_review"] is True
