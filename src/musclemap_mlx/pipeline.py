"""End-to-end MuscleMap v1.4 preprocessing, slice inference and label geometry."""

from __future__ import annotations

from typing import cast

import numpy as np
from medmlx_core.sliding_window import sliding_window_inference

from musclemap_mlx.config import ModelSpec
from musclemap_mlx.geometry import (
    Float32Array,
    Float64Array,
    Int16Array,
    PreparedVolume,
    ReadVolume,
    discretize,
    invert_logits,
    keep_largest_components,
    preprocess,
    read_chunk,
)
from musclemap_mlx.network import MuscleMapUNet

AUTO_CHUNK_SLICES = 8
GAUSSIAN_SIGMA_SCALE = 0.125
SLICE_ROI = 256


def infer_volume(
    data: Float32Array,
    affine: Float64Array,
    spec: ModelSpec,
    network: MuscleMapUNet,
    *,
    chunk_size: int | None = None,
    overlap_percent: int = 90,
    source_volume: ReadVolume | None = None,
) -> tuple[Int16Array, int, bool]:
    """Segment one reader-grid volume and return labels plus chunking facts.

    ``chunk_size=None`` uses a conservative eight-slice chunk; a value at least
    the volume depth runs the whole-volume path. Chunking follows the upstream
    CLI: each chunk is independently preprocessed, inferred, inverted and
    discretized, then the assembled label map gets the largest-component filter.
    It therefore changes the input to the network at chunk boundaries, exactly as
    the publisher's ``temp_chunks`` path does.

    Args:
        data: Float32 volume with the reader's storage axes ``(H, W, D)``.
        affine: Reader-grid 4x4 voxel-to-world affine in millimetres.
        spec: Validated publisher model settings.
        network: Native MLX graph loaded from the converted checkpoint.
        chunk_size: Positive number of axial slices per independent pipeline
            call, or ``None`` for the eight-slice default.
        overlap_percent: Slice-window overlap percentage in ``[0, 100)``;
            default 90, the publisher's inference setting.
        source_volume: NIfTI reader metadata for the upstream source-header chunk
            round trip. Array-only callers omit this and retain float32 pixels.

    Returns:
        ``(labels, chunk_size_used, chunked)``. Labels are int16 on the reader
        grid with the original 113 anatomical IDs plus background.

    Raises:
        ValueError: The volume, chunk size, overlap or preprocessing geometry is invalid.
        MissingDependencyError: MLX or its Metal device is unavailable.
        RuntimeError: The inverse geometry or label map fails its shape checks.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    if not 0 <= overlap_percent < 100:
        raise ValueError("overlap_percent must be in [0, 100)")
    depth = int(data.shape[2])
    if chunk_size is not None and (
        isinstance(cast(object, chunk_size), bool)
        or not isinstance(cast(object, chunk_size), int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer or None for automatic sizing")
    effective = min(depth, AUTO_CHUNK_SLICES) if chunk_size is None else min(chunk_size, depth)
    overlap = overlap_percent / 100.0
    if effective >= depth:
        prepared = preprocess(data, affine, spec)
        labels = _infer_prepared(prepared, spec, network, overlap)
        return keep_largest_components(labels), effective, False

    full = np.zeros(data.shape, dtype=np.int16)
    for start in range(0, depth, effective):
        end = min(start + effective, depth)
        if source_volume is None:
            chunk_data, chunk_affine = np.ascontiguousarray(data[..., start:end]), affine
        else:
            chunk = read_chunk(source_volume, start, end)
            chunk_data, chunk_affine = chunk.data, chunk.affine
        prepared = preprocess(chunk_data, chunk_affine, spec)
        labels = _infer_prepared(prepared, spec, network, overlap)
        if labels.shape != (data.shape[0], data.shape[1], end - start):
            raise RuntimeError("MuscleMap chunk labels do not match the requested slice range")
        full[..., start:end] = labels
    return keep_largest_components(full), effective, True


def _infer_prepared(
    prepared: PreparedVolume,
    spec: ModelSpec,
    network: MuscleMapUNet,
    overlap: float,
) -> Int16Array:
    # Each slice is an independent 2D image. This makes host accumulation
    # contiguous within a slice while preserving window order within every
    # slice. The singleton-depth Gaussian factor in the 3D inferer is exactly 1.
    inputs = np.moveaxis(prepared.data, -1, 0)
    result = sliding_window_inference(
        inputs,
        (SLICE_ROI, SLICE_ROI),
        spec.spatial_window_batch_size,
        network,
        overlap=overlap,
        mode="gaussian",
        sigma_scale=GAUSSIAN_SIGMA_SCALE,
        padding_mode="constant",
        cval=0.0,
        sw_device="gpu",
        device="cpu",
        predictor_input="mlx",
        overlap_prediction=True,
    )
    logits = np.asarray(result, dtype=np.float32)
    if (
        logits.ndim != 4
        or logits.shape[0] != inputs.shape[0]
        or int(logits.shape[1]) != spec.out_channels
    ):
        raise RuntimeError(f"MuscleMap network returned an unexpected logit shape {logits.shape}")
    inverted = invert_logits(np.moveaxis(logits, 0, -1), prepared.geometry)
    labels = discretize(inverted, spec)
    expected = prepared.geometry.original_shape
    if tuple(labels.shape) != expected:
        raise RuntimeError(
            f"MuscleMap labels {labels.shape} do not match the reader grid {expected}"
        )
    return labels
