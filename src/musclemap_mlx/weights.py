"""Strict native safetensors admission for the converted MuscleMap checkpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from medmlx_core.errors import AssetNotReadyError
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import ArrayFactory

from musclemap_mlx.config import (
    CONVERTED_NAME,
    MODEL_ID,
    ModelSpec,
    ParameterInventory,
    expected_parameters,
)
from musclemap_mlx.network import MuscleMapRuntime, MuscleMapUNet, as_musclemap_runtime

if TYPE_CHECKING:
    pass

type Float32Array = npt.NDArray[np.float32]

_WEIGHTS_HINT = (
    "Run musclemap-mlx download or pass a folder containing model.safetensors as weights_path."
)


def resolve_weights_file(weights_path: str | os.PathLike[str]) -> Path:
    """Return the ``model.safetensors`` file inside a local weights folder.

    A direct file path is also accepted for local experimentation, but the folder
    form is the documented MedMLX one.

    Raises:
        AssetNotReadyError: The path is missing, is not a safetensors file, or the
            folder does not contain ``model.safetensors``.
    """
    path = Path(weights_path).expanduser().absolute()
    if path.is_dir():
        path = path / CONVERTED_NAME
    if not path.is_file():
        raise AssetNotReadyError(
            f"MuscleMap weights not found: {path}",
            reason=f"missing {path}",
            hint=_WEIGHTS_HINT,
        )
    if path.suffix != ".safetensors":
        raise AssetNotReadyError(
            f"MuscleMap weights must be a .safetensors file, got {path.name}",
            reason=f"unconverted {path.name}",
            hint=_WEIGHTS_HINT,
        )
    return path


def load_network(
    weights_path: str | os.PathLike[str],
    spec: ModelSpec,
    *,
    inventory: ParameterInventory | None = None,
) -> MuscleMapUNet:
    """Load and validate the converted tensors, then construct the MLX graph.

    The file must contain exactly the expected 46 float32 tensors with the
    published shapes and finite values. Torch and MONAI are not imported.

    Raises:
        AssetNotReadyError: The file cannot be loaded or its inventory differs.
        MissingDependencyError: MLX or Metal is unavailable.
    """
    path = resolve_weights_file(weights_path)
    runtime: MuscleMapRuntime = as_musclemap_runtime(import_mlx())
    expected = inventory or expected_parameters(spec)
    try:
        payload = runtime.load(str(path), format="safetensors")
    except (OSError, ValueError) as error:
        raise AssetNotReadyError(
            f"Could not read MuscleMap safetensors: {path}",
            reason=str(error),
            hint=_WEIGHTS_HINT,
        ) from error
    if not isinstance(payload, dict):
        raise AssetNotReadyError(
            "MuscleMap safetensors must contain a tensor mapping",
            reason="non-mapping checkpoint",
            hint=_WEIGHTS_HINT,
        )
    arrays = _admit(payload, expected)
    make_array = cast(ArrayFactory, runtime.array)
    weights = {name: make_array(value) for name, value in arrays.items()}
    runtime.eval(list(weights.values()))
    return MuscleMapUNet(weights, spec, expected, mx=runtime)


def _admit(payload: Mapping[str, object], expected: ParameterInventory) -> dict[str, Float32Array]:
    names = set(payload)
    missing = sorted(set(expected.shapes) - names)
    extra = sorted(names - set(expected.shapes))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing[:4])}")
        if extra:
            details.append(f"unexpected {', '.join(extra[:4])}")
        raise AssetNotReadyError(
            f"MuscleMap checkpoint inventory differs ({'; '.join(details)})",
            reason="mismatched tensor inventory",
            hint=_WEIGHTS_HINT,
        )
    admitted: dict[str, Float32Array] = {}
    for name, shape in expected.shapes.items():
        array = np.asarray(payload[name])
        if array.shape != shape:
            raise AssetNotReadyError(
                f"MuscleMap tensor {name} has shape {array.shape}, expected {shape}",
                reason="mismatched tensor shape",
                hint=_WEIGHTS_HINT,
            )
        if array.dtype != np.float32:
            raise AssetNotReadyError(
                f"MuscleMap tensor {name} has dtype {array.dtype}, expected float32",
                reason="mismatched tensor dtype",
                hint=_WEIGHTS_HINT,
            )
        if not np.isfinite(array).all():
            raise AssetNotReadyError(
                f"MuscleMap tensor {name} contains non-finite values",
                reason="non-finite tensor",
                hint=_WEIGHTS_HINT,
            )
        admitted[name] = np.ascontiguousarray(array)
    return admitted


def weights_inventory(spec: ModelSpec) -> ParameterInventory:
    """Expose the expected inventory for tests and conversion checks."""
    return expected_parameters(spec)


__all__ = [
    "MODEL_ID",
    "load_network",
    "resolve_weights_file",
    "weights_inventory",
]
