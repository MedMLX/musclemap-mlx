"""Recorded MONAI 1.3.2 geometry parity, runnable without the GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musclemap_mlx.config import default_spec
from musclemap_mlx.geometry import discretize, invert_logits, preprocess

pytestmark = pytest.mark.oracle

# MONAI resamples in float64 and then reduces nonzero values in float32 with
# ATen's vectorized sum. The raw spacing array is bitwise exact; normalization
# can differ by one float32 reduction ulp on the rotated case.
OBLIQUE_FLOAT32_TOL = 2e-7


def test_preprocess_matches_recorded_monai_132(reference_dir: Path) -> None:
    with np.load(reference_dir / "preprocess.npz", allow_pickle=False) as fixture:
        data = fixture["input"]
        affine = fixture["loaded_affine"]
        expected = fixture["final"]
        expected_affine = fixture["spacing_affine"]
    prepared = preprocess(data, affine, default_spec())
    assert np.array_equal(prepared.data, expected)
    assert np.allclose(prepared.geometry.spaced_affine, expected_affine, rtol=0.0, atol=1e-12)


def test_inverse_logits_match_recorded_monai_132(reference_dir: Path) -> None:
    with np.load(reference_dir / "preprocess.npz", allow_pickle=False) as fixture:
        data = fixture["input"]
        affine = fixture["loaded_affine"]
    with np.load(reference_dir / "inverse.npz", allow_pickle=False) as fixture:
        padded = fixture["padded_logits"]
        expected = fixture["inverted_logits"]
    prepared = preprocess(data, affine, default_spec())
    inverted = invert_logits(padded, prepared.geometry)
    assert np.array_equal(inverted, expected)


def test_oblique_preprocess_matches_recorded_monai_132(reference_dir: Path) -> None:
    with np.load(reference_dir / "preprocess_oblique.npz", allow_pickle=False) as fixture:
        data = fixture["input"]
        affine = fixture["loaded_affine"]
        expected = fixture["final"]
        expected_spacing = fixture["spacing_data"]
        expected_affine = fixture["spacing_affine"]
    prepared = preprocess(data, affine, default_spec())
    assert np.allclose(prepared.geometry.spaced_affine, expected_affine, rtol=0.0, atol=1e-12)
    # The resampled grid and normalization mask must be exact; the final
    # float32 values admit ATen's reduction-order rounding.
    assert prepared.geometry.spaced_shape == expected_spacing.shape[1:]
    assert np.allclose(prepared.data, expected, rtol=0.0, atol=OBLIQUE_FLOAT32_TOL)


def test_oblique_inverse_matches_recorded_monai_132(reference_dir: Path) -> None:
    with np.load(reference_dir / "preprocess_oblique.npz", allow_pickle=False) as fixture:
        data = fixture["input"]
        affine = fixture["loaded_affine"]
    with np.load(reference_dir / "inverse_oblique.npz", allow_pickle=False) as fixture:
        padded = fixture["padded_logits"]
        expected = fixture["inverted_logits"]
    prepared = preprocess(data, affine, default_spec())
    inverted = invert_logits(padded, prepared.geometry)
    assert np.array_equal(inverted, expected)


def test_argmax_matches_recorded_pipeline_labels(reference_dir: Path) -> None:
    with np.load(reference_dir / "pipeline.npz", allow_pickle=False) as fixture:
        inverted = fixture["inverted_logits"]
        expected = fixture["labels_unfiltered"]
    labels = discretize(inverted, default_spec())
    assert np.array_equal(labels, expected)
