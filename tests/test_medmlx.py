"""MedMLX declaration and preflight rejection contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.volumes import write_volume

from musclemap_mlx import medmlx


def _image(path: Path) -> None:
    write_volume(path, np.ones((8, 9, 2), dtype=np.float32), np.eye(4), dtype=np.float32)


def test_declaration_preserves_labels_and_setup_boundary() -> None:
    assert medmlx.MODEL["id"] == "musclemap"
    assert medmlx.MODEL["task"] == "segmentation"
    assert medmlx.MODEL["setup_supported"] is True
    labels = cast(dict[str, str], medmlx.MODEL["labels"])
    assert len(labels) == 114
    assert labels["0"] == "background"
    assert labels["1101"] == "left levator scapulae"
    assert labels["8222"] == "right plantaris"


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="does not exist"):
        medmlx.run(image_path=tmp_path / "missing.nii.gz", output_dir=tmp_path / "out")


def test_non_bool_overwrite_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)
    with pytest.raises(InvalidInputError, match="overwrite"):
        medmlx.run(
            image_path=image,
            output_dir=tmp_path / "out",
            overwrite=cast(bool, "false"),
        )


def test_existing_output_is_rejected_before_weights(tmp_path: Path) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)
    output = tmp_path / "out"
    output.mkdir()
    (output / medmlx.OUTPUT_NAME).write_bytes(b"kept")
    with pytest.raises(InvalidInputError, match="overwrite=true"):
        medmlx.run(image_path=image, output_dir=output)


def test_missing_weights_is_an_asset_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)

    def fail_mlx() -> None:
        raise AssertionError("MLX was imported before weights were resolved")

    monkeypatch.setattr(medmlx, "import_mlx", fail_mlx)
    def missing_weights(_model_id: str, _path: object, _name: str) -> Path:
        raise AssetNotReadyError("MuscleMap converted weights unavailable")

    monkeypatch.setattr(medmlx, "weights_file", missing_weights)
    with pytest.raises(AssetNotReadyError, match="converted weights"):
        medmlx.run(image_path=image, output_dir=tmp_path / "out")


def test_missing_converted_file_is_an_asset_error(tmp_path: Path) -> None:
    image = tmp_path / "image.nii.gz"
    _image(image)
    with pytest.raises(AssetNotReadyError, match="does not exist"):
        medmlx.run(
            image_path=image,
            weights_path=tmp_path / "weights",
            output_dir=tmp_path / "out",
        )
