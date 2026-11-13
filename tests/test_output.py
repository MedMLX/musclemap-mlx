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
    source.header.set_xyzt_units("mm", "sec")
    labels = np.zeros((6, 7, 2), dtype=np.int16)
    labels[2:4, 3:5, :] = 1101
    target = tmp_path / "mask.nii.gz"
    publish_mask(target, labels, AFFINE, source.header)
    written = nifti_io.load(str(target))
    assert np.array_equal(np.asarray(written.dataobj), labels)
    assert int(np.asarray(written.header.get("datatype"))) == 4
    assert written.header.get_xyzt_units() == ("mm", "sec")
    assert np.allclose(written.affine, AFFINE, rtol=0.0, atol=1e-6)
    _, qcode = written.get_qform(coded=True)
    _, scode = written.get_sform(coded=True)
    assert (qcode, scode) == (2, 1)
    assert np.allclose(np.asarray(written.header.get_zooms()[:3]), (0.9, 1.3, 2.5))
    assert list(tmp_path.glob(".*.mask.nii.gz")) == []


def test_publish_mask_preserves_distinct_source_forms(tmp_path: Path) -> None:
    source = nifti_io.Nifti1Image(np.zeros((6, 7, 2), dtype=np.float32), AFFINE)
    qform = AFFINE.copy()
    qform[0, 3] += 10
    source.set_qform(qform, 2)
    source.set_sform(AFFINE, 1)
    target = tmp_path / "mask.nii.gz"
    publish_mask(target, np.full((6, 7, 2), 8222, dtype=np.int16), AFFINE, source.header)
    written = nifti_io.load(str(target))
    restored_qform, _ = written.get_qform(coded=True)
    restored_sform, _ = written.get_sform(coded=True)
    assert restored_qform is not None and restored_sform is not None
    np.testing.assert_allclose(restored_qform, qform, atol=1e-6, rtol=0)
    np.testing.assert_allclose(restored_sform, AFFINE, atol=1e-6, rtol=0)
