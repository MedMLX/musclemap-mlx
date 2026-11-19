"""CPU geometry contracts for the MONAI-compatible preprocessing and inversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musclemap_mlx.config import default_spec
from musclemap_mlx.geometry import (
    discretize,
    invert_logits,
    keep_largest_components,
    preprocess,
    read_nifti,
)
from musclemap_mlx.geometry import nib as nifti_io


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


def test_reader_rectifies_uncoded_qform_like_monai(tmp_path: Path) -> None:
    """MONAI 1.3.2 replaces an sform that disagrees with pixdim by the stored qform,
    even with qform_code 0 (NiBabel's default), which changes the resampled grid."""
    data, affine = _phantom()
    affine[:3, :3] = np.diag((3.0, 3.0, 3.0))
    image = nifti_io.Nifti1Image(data, affine)
    image.set_qform(affine, 0)
    image.set_sform(affine, 2)
    image.header.set_zooms((2.5, 2.5, 3.0))
    path = tmp_path / "edited_pixdim.nii"
    nifti_io.save(image, str(path))
    assert np.array_equal(read_nifti(path).affine, nifti_io.load(str(path)).get_qform())


def test_empty_foreground_is_rejected() -> None:
    with pytest.raises(ValueError, match="no positive foreground"):
        preprocess(np.zeros((16, 16, 2), dtype=np.float32), np.eye(4), default_spec())


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (-1.5, 0.75, 2.5)])
def test_inverse_logits_restores_the_reader_grid(spacing: tuple[float, float, float]) -> None:
    data, affine = _phantom()
    affine[:3, :3] = np.diag(spacing)
    prepared = preprocess(data, affine, default_spec())
    geometry = prepared.geometry
    assert geometry.crop_start == (0, 0, 0)
    assert geometry.crop_shape == geometry.spaced_shape
    h, w, d = np.indices(geometry.padded_shape, dtype=np.float32)
    # Distinct channel ramps expose swapped channels and incomplete worker writes.
    logits = np.stack([h + 2 * w + 4 * d + 16 * c for c in range(12)])
    inverted = invert_logits(logits, geometry)
    source_h, source_w, source_d = np.indices(data.shape, dtype=np.float32)
    if spacing[0] < 0:
        source_h = data.shape[0] - 1 - source_h
    source_h = np.minimum(source_h * abs(spacing[0]), geometry.spaced_shape[0] - 1)
    source_w = np.minimum(source_w * spacing[1], geometry.spaced_shape[1] - 1)
    expected = np.stack([
        source_h + 2 * source_w + 4 * source_d + 16 * c for c in range(12)
    ])
    assert np.array_equal(inverted, expected)
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
