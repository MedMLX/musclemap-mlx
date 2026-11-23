"""Strict native safetensors admission for the converted MuscleMap checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime
from medmlx_core.weights import load_tensors

from musclemap_mlx.config import (
    MODEL_ID,
    ModelSpec,
    ParameterInventory,
    expected_parameters,
)
from musclemap_mlx.network import MuscleMapUNet


def load_network(
    path: Path,
    spec: ModelSpec,
    *,
    inventory: ParameterInventory | None = None,
) -> MuscleMapUNet:
    """Load and validate the converted tensors, then construct the MLX graph.

    ``path`` is the ``model.safetensors`` file. It must contain exactly the
    expected 46 float32 tensors with the published shapes and finite values.
    Torch and MONAI are not imported.

    Raises:
        AssetNotReadyError: The file cannot be loaded or its inventory differs.
        MissingDependencyError: MLX or Metal is unavailable.
    """
    runtime = cast(MlxRuntime, import_mlx())
    expected = inventory or expected_parameters(spec)
    weights = load_tensors(path, model_id=MODEL_ID, label="MuscleMap", shapes=expected.shapes)
    runtime.eval(list(weights.values()))
    return MuscleMapUNet(weights, spec, expected, mx=runtime)


def weights_inventory(spec: ModelSpec) -> ParameterInventory:
    """Expose the expected inventory for tests and conversion checks."""
    return expected_parameters(spec)


__all__ = [
    "MODEL_ID",
    "load_network",
    "weights_inventory",
]
