"""MuscleMap native-port development intake; no inference implementation yet."""

from importlib.metadata import version


def __getattr__(name: str) -> str:
    if name == "__version__":
        return version("musclemap-mlx")
    raise AttributeError(name)
