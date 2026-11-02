"""MedMLX declaration and runner for MuscleMap v1.4 whole-body segmentation.

The publisher's checkpoint is a 2D MONAI UNet with one input channel, 114 output
channels and instance-normalized residual stages. The runner reads a 3D NIfTI,
reorients it to RAS, resamples to 1 mm in-plane while keeping the original slice
spacing, normalizes nonzero voxels, crops the positive foreground with a 20-voxel
margin, pads to 256x256, runs 2D slices through the native MLX graph, inverts the
geometry on the logits, takes the channel argmax and restores the original 113
anatomical IDs. It is research software, not a diagnostic device.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    ModelExecutionError,
)
from medmlx_core.runtime import import_mlx
from medmlx_core.weights import weights_revision

from musclemap_mlx.config import (
    CONFIG_NAME,
    MODEL_ID,
    ModelSpec,
    default_spec,
    load_model_config,
)

OUTPUT_NAME = "mask.nii.gz"
DEFAULT_OVERLAP_PERCENT = 90
_WEIGHTS_HINT = (
    "No converted MuscleMap weights are hosted yet. Convert the publisher checkpoint "
    "locally to model.safetensors and pass its folder as weights_path."
)

_SPEC = default_spec()
MODEL = {
    "id": MODEL_ID,
    "label": "MuscleMap v1.4 whole-body",
    "task": "segmentation",
    "run": "musclemap_mlx.medmlx:run",
    "summary": (
        "Segment 113 whole-body muscle and bone structures in MRI or CT with the "
        "contrast-agnostic MuscleMap v1.4 2D MONAI UNet."
    ),
    "setup_supported": False,
    "labels": {str(item["value"]): str(item["name"]) for item in _SPEC.label_details},
}


def run(
    *,
    image_path: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
    chunk_size: int | None = None,
    overlap_percent: int = DEFAULT_OVERLAP_PERCENT,
    overwrite: bool = False,
) -> dict[str, object]:
    """Segment one whole-body MRI or CT volume with MuscleMap v1.4.

    The input is reoriented to RAS, resampled to 1 mm in the first two axes while
    keeping the source slice spacing, nonzero-normalized, cropped to the positive
    foreground with a 20-voxel margin and padded to at least 256x256x1. Each axial
    slice runs through the 114-channel native MLX UNet with 90% Gaussian window
    overlap by default. Logits are resampled back to the reader grid before argmax,
    and the largest 6-connected component of each anatomical label is kept. The
    output uses the original 113 anatomical IDs, not channel indices.

    Args:
        image_path: 3D NIfTI (``.nii`` or ``.nii.gz``) in millimetres. MR or CT
            contrast is accepted; the model is contrast-agnostic. Header slope and
            intercept are applied on read.
        weights_path: Folder containing ``model.safetensors`` from the local
            converter. There is no hosted converted checkpoint yet, so omitting
            this raises an asset error instead of downloading.
        output_dir: Destination folder for ``mask.nii.gz``. Created only after
            validation.
        chunk_size: Positive number of axial slices processed per independent
            preprocessing/inference pass, or ``None`` for the conservative
            eight-slice default. A value at least the volume depth runs one pass.
        overlap_percent: Gaussian slice-window overlap in ``[0, 100)``; default 90,
            matching the publisher's inference configuration.
        overwrite: Replace an existing ``mask.nii.gz`` when true. Input aliases are
            always refused.

    Returns:
        ``{"outputs": {"mask": absolute path}, "details": {...}}``. Details include
        the original labels, MLX device, peak MLX memory, chunking facts, duration
        and ``requires_human_review``.

    Raises:
        InvalidInputError: The image, output folder, booleans, chunk size, overlap
            or output collision is invalid.
        AssetNotReadyError: Converted weights are missing or differ from the pinned
            tensor inventory.
        ModelExecutionError: Preprocessing, network execution or output validation
            fails.
        MissingDependencyError: MLX or its Metal device is unavailable.
    """
    import numpy as np
    from nibabel.filebasedimages import ImageFileError

    from musclemap_mlx.geometry import axis_codes, publish_mask, read_nifti
    from musclemap_mlx.weights import resolve_weights_file

    started = time.perf_counter()
    source = _require_nifti(image_path)
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be true or false")
    if isinstance(cast(object, chunk_size), bool) or (
        chunk_size is not None
        and (not isinstance(cast(object, chunk_size), int) or chunk_size < 1)
    ):
        raise InvalidInputError("chunk_size must be a positive integer or omitted")
    if isinstance(cast(object, overlap_percent), bool) or not isinstance(
        cast(object, overlap_percent), int
    ):
        raise InvalidInputError("overlap_percent must be an integer in [0, 99]")
    if not 0 <= overlap_percent < 100:
        raise InvalidInputError("overlap_percent must be in [0, 99]")
    directory = Path(output_dir).expanduser().absolute()
    if directory.exists() and not directory.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {directory}")
    mask_path = directory / OUTPUT_NAME
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"{mask_path} already exists; pass overwrite=true to replace it")
    if mask_path.exists() and mask_path.samefile(source):
        raise InvalidInputError("MuscleMap output must differ from the input image")
    if mask_path.resolve() == source.resolve():
        raise InvalidInputError("MuscleMap output must differ from the input image")
    if weights_path is None:
        raise AssetNotReadyError(
            "MuscleMap converted weights are not configured",
            reason="no automatic converted weights",
            hint=_WEIGHTS_HINT,
        )
    weights = resolve_weights_file(weights_path)
    try:
        volume = read_nifti(source)
    except (OSError, ValueError, ImageFileError) as error:
        raise InvalidInputError(f"Could not read MuscleMap input: {error}") from error
    spec = _load_spec(weights)
    mx = import_mlx()
    mx.reset_peak_memory()
    from musclemap_mlx.pipeline import infer_volume
    from musclemap_mlx.weights import load_network

    try:
        network = load_network(weights, spec)
        labels, chunk_used, chunked = infer_volume(
            volume.data,
            volume.affine,
            spec,
            network,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
        )
    except (ValueError, RuntimeError) as error:
        raise ModelExecutionError(f"MuscleMap inference failed: {error}") from error
    if not set(np.unique(labels).tolist()) <= set(spec.class_labels):
        raise ModelExecutionError("MuscleMap produced labels outside the published ID table")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        publish_mask(mask_path, labels, volume.output_affine, volume.output_header)
    except OSError as error:
        raise ModelExecutionError(f"Could not write MuscleMap mask: {error}") from error
    counts = {
        str(int(value)): int(count)
        for value, count in zip(*np.unique(labels, return_counts=True), strict=True)
        if int(value) != 0
    }
    return {
        "outputs": {"mask": mask_path.resolve().as_posix()},
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "labels": [dict(item) for item in spec.label_details],
            "voxel_counts": counts,
            "shape": [int(value) for value in labels.shape],
            "dtype": "int16",
            "input_orientation": axis_codes(volume.affine),
            "chunk_size": chunk_used,
            "chunked": chunked,
            "overlap_percent": overlap_percent,
            "mlx_device": str(mx.default_device()),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "requires_human_review": True,
            "warnings": [
                "Synthetic and released-weight checks only; no patient-data validation."
            ],
        },
    }


def _require_nifti(path: str | os.PathLike[str]) -> Path:
    source = Path(path).expanduser().absolute()
    if not source.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a .nii or .nii.gz file: {source}")
    if not source.is_file():
        raise InvalidInputError(f"image_path does not exist: {source}")
    return source


def _load_spec(weights: Path) -> ModelSpec:
    for candidate in (weights.parent / "config.json", weights.parent / CONFIG_NAME):
        if candidate.is_file():
            try:
                return load_model_config(candidate)
            except (OSError, ValueError) as error:
                raise AssetNotReadyError(
                    f"MuscleMap configuration is invalid: {candidate}",
                    reason=str(error),
                    hint=_WEIGHTS_HINT,
                ) from error
    return default_spec()
