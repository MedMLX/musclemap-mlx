"""NIfTI output geometry contracts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from medmlx_core.volumes import read_volume

from musclemap_mlx.geometry import read_nifti, write_mask

if TYPE_CHECKING:
    from conftest import SaveNifti

AFFINE = np.array(
    [
        [-0.9, 0.0, 0.0, 12.0],
        [0.0, -1.3, 0.0, -7.0],
        [0.0, 0.0, 2.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def test_write_mask_keeps_affine_forms_and_dtype(tmp_path: Path, save_nifti: SaveNifti) -> None:
    image = save_nifti(
        tmp_path / "image.nii.gz",
        np.zeros((6, 7, 2), dtype=np.float32),
        sform=AFFINE,
        sform_code=1,
        qform=AFFINE,
        qform_code=2,
        units=("mm", "sec"),
    )
    labels = np.zeros((6, 7, 2), dtype=np.int16)
    labels[2:4, 3:5, :] = 1101
    target = tmp_path / "mask.nii.gz"
    write_mask(target, labels, read_nifti(image).source)
    written = read_volume(target, dtype=None)
    assert np.array_equal(written.data, labels)
    assert int(written.header["datatype"]) == 4
    # NIfTI xyzt_units: millimeter (2) plus second (8).
    assert int(written.header["xyzt_units"]) == 10
    assert np.allclose(written.affine, AFFINE, rtol=0.0, atol=1e-6)
    assert (written.qform_code, written.sform_code) == (2, 1)
    assert np.allclose(written.zooms, (0.9, 1.3, 2.5))


def test_write_mask_preserves_distinct_source_forms(
    tmp_path: Path, save_nifti: SaveNifti
) -> None:
    qform = AFFINE.copy()
    qform[0, 3] += 10
    image = save_nifti(
        tmp_path / "image.nii.gz",
        np.zeros((6, 7, 2), dtype=np.float32),
        sform=AFFINE,
        sform_code=1,
        qform=qform,
        qform_code=2,
    )
    target = tmp_path / "mask.nii.gz"
    labels = np.full((6, 7, 2), 8222, dtype=np.int16)
    write_mask(target, labels, read_nifti(image).source)
    written = read_volume(target, dtype=None)
    np.testing.assert_allclose(written.qform, qform, atol=1e-6, rtol=0)
    np.testing.assert_allclose(written.sform, AFFINE, atol=1e-6, rtol=0)
