"""MuscleMap v1.4 whole-body segmentation with native MLX on Apple Silicon."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from musclemap_mlx.medmlx import run as run

__all__ = ["run"]


def __getattr__(name: str) -> object:
    """Resolve ``__version__`` and ``run`` on first access, keeping MLX out of the import."""
    if name == "__version__":
        return version("musclemap-mlx")
    if name == "run":
        from musclemap_mlx.medmlx import run

        return run
    raise AttributeError(name)
