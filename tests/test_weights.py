"""Converted-weight path and inventory admission contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from medmlx_core.errors import AssetNotReadyError

from musclemap_mlx.config import default_spec
from musclemap_mlx.network import MuscleMapRuntime
from musclemap_mlx.weights import load_network, resolve_weights_file


def test_resolve_folder_and_file(tmp_path: Path) -> None:
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")
    assert resolve_weights_file(folder) == folder / "model.safetensors"
    assert resolve_weights_file(folder / "model.safetensors") == folder / "model.safetensors"


def test_missing_and_unconverted_paths_are_assets_errors(tmp_path: Path) -> None:
    with pytest.raises(AssetNotReadyError, match="weights not found"):
        resolve_weights_file(tmp_path / "missing")
    (tmp_path / "model.pth").write_bytes(b"x")
    with pytest.raises(AssetNotReadyError, match="unconverted"):
        resolve_weights_file(tmp_path)


@pytest.mark.metal
def test_wrong_inventory_is_rejected(tmp_path: Path, mx: MuscleMapRuntime) -> None:
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), {"wrong": mx.zeros((1,))})
    with pytest.raises(AssetNotReadyError, match="inventory differs"):
        load_network(path, default_spec())
