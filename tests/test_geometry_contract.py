"""CPU geometry contracts for the MONAI-compatible preprocessing and inversion."""

from __future__ import annotations

import numpy as np
import pytest

from musclemap_mlx.config import default_spec
from musclemap_mlx.geometry import (
    discretize,
    invert_logits,
    keep_largest_components,
    preprocess,
)


def _phantom() -> tuple[np.ndarray, np.ndarray]:
    data = np.zeros((32, 32, 2), dtype=np.float32)
    data[10:22, 11:23, :] = 10.0
    data[15:17, 16:18, :] = -5.0
    return data, np.eye(4, dtype=np.float64)


def test_preprocess_pads_and_tracks_geometry() -> None:
    data, affine = _phantom()
    prepared = preprocess(data, affine, default_spec())
    geometry = prepared.geometry
    assert prepared.data.shape == (1, 256, 256, 2)
    assert geometry.original_shape == (32, 32, 2)
    assert geometry.padded_shape == (256, 256, 2)
    assert geometry.crop_shape[2] == 2
    assert prepared.data.dtype == np.float32
    assert np.isfinite(prepared.data).all()


def test_empty_foreground_is_rejected() -> None:
    with pytest.raises(ValueError, match="no positive foreground"):
        preprocess(np.zeros((16, 16, 2), dtype=np.float32), np.eye(4), default_spec())


def test_inverse_logits_restores_the_reader_grid() -> None:
    data, affine = _phantom()
    prepared = preprocess(data, affine, default_spec())
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, *prepared.geometry.padded_shape)).astype(np.float32)
    inverted = invert_logits(logits, prepared.geometry)
    assert inverted.shape == (4, *prepared.geometry.original_shape)
    assert inverted.dtype == np.float32


def test_discretize_uses_original_anatomical_ids() -> None:
    spec = default_spec()
    logits = np.zeros((114, 4, 5, 1), dtype=np.float32)
    logits[0] = 1.0
    logits[1, 1, 2, 0] = 5.0
    logits[7, 3, 4, 0] = 5.0
    labels = discretize(logits, spec)
    assert labels.dtype == np.int16
    assert labels[1, 2, 0] == spec.class_labels[1]
    assert labels[3, 4, 0] == spec.class_labels[7]
    assert labels[0, 0, 0] == 0


def test_largest_component_filter_keeps_one_region_per_label() -> None:
    labels = np.zeros((8, 8, 1), dtype=np.int16)
    labels[1:3, 1:3, 0] = 1101
    labels[6:8, 6:8, 0] = 1101
    labels[4:5, 4:5, 0] = 1102
    filtered = keep_largest_components(labels)
    assert int(np.count_nonzero(filtered == 1101)) == 4
    assert int(np.count_nonzero(filtered == 1102)) == 1
    assert filtered.dtype == np.int16
