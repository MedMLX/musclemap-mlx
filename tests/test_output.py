"""NIfTI output geometry and atomic publication contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from musclemap_mlx.geometry import nib as nifti_io
from musclemap_mlx.geometry import publish_mask

AFFINE = np.array(
    [
        [-0.9, 0.0, 0.0, 12.0],
        [0.0, -1.3, 0.0, -7.0],
        [0.0, 0.0, 2.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def test_publish_mask_keeps_affine_forms_and_dtype(tmp_path: Path) -> None:
    source = nifti_io.Nifti1Image(np.zeros((6, 7, 2), dtype=np.float32), AFFINE)
    source.set_qform(AFFINE, 2)
    source.set_sform(AFFINE, 1)
    labels = np.zeros((6, 7, 2), dtype=np.int16)
    labels[2:4, 3:5, :] = 1101
    target = tmp_path / "mask.nii.gz"
    publish_mask(target, labels, AFFINE, source.header)
    written = nifti_io.load(str(target))
    assert np.array_equal(np.asarray(written.dataobj), labels)
    assert np.allclose(written.affine, AFFINE, rtol=0.0, atol=1e-6)
    _, qcode = written.get_qform(coded=True)
    _, scode = written.get_sform(coded=True)
    assert (qcode, scode) == (2, 1)
    assert np.allclose(np.asarray(written.header.get_zooms()[:3]), (0.9, 1.3, 2.5))
    assert list(tmp_path.glob(".*.mask.nii.gz")) == []
