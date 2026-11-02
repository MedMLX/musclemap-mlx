"""MuscleMap v1.4 whole-body segmentation with native MLX on Apple Silicon."""

from __future__ import annotations

from importlib.metadata import version


def __getattr__(name: str) -> str:
    if name == "__version__":
        return version("musclemap-mlx")
    raise AttributeError(name)
