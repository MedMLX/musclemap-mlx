"""Converted-weight inventory admission contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from medmlx_core.errors import AssetNotReadyError
from medmlx_core.typing import MlxRuntime

from musclemap_mlx.config import default_spec
from musclemap_mlx.weights import load_network


@pytest.mark.metal
def test_wrong_inventory_is_rejected(tmp_path: Path, mx: MlxRuntime) -> None:
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), {"wrong": mx.zeros((1,))})
    with pytest.raises(AssetNotReadyError, match="tensor names differ"):
        load_network(path, default_spec())
