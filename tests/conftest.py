"""Shared native runtime and opt-in reference paths."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from musclemap_mlx.network import MuscleMapRuntime, as_musclemap_runtime


@pytest.fixture(scope="session")
def mx() -> MuscleMapRuntime:
    """Provide the admitted MLX Metal runtime, skipping non-Apple hosts."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MuscleMap native tests require macOS arm64 with MLX Metal")
    from medmlx_core.runtime import import_mlx

    return as_musclemap_runtime(import_mlx())


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """Return the opt-in recorded MONAI reference directory."""
    value = _env_path("MEDMLX_TEST_REFERENCE")
    if value is None:
        pytest.skip("Set MEDMLX_TEST_REFERENCE to the recorded MONAI fixture folder")
    return value


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    """Return the opt-in converted safetensors folder."""
    value = _env_path("MEDMLX_TEST_WEIGHTS")
    if value is None:
        pytest.skip("Set MEDMLX_TEST_WEIGHTS to a folder containing model.safetensors")
    return value


def _env_path(name: str) -> Path | None:
    import os

    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser().absolute()
    return path
