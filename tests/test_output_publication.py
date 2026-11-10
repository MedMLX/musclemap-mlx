"""Mask publication cannot write through pre-existing temporary links."""

import os
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np

from musclemap_mlx.geometry import NiftiHeader, publish_mask, read_nifti


def test_mask_staging_preserves_external_files(tmp_path: Path) -> None:
    victim = tmp_path / "kept.txt"
    victim.write_bytes(b"keep this file")
    target = tmp_path / "mask.nii.gz"
    (tmp_path / f".{os.getpid()}.{target.name}").symlink_to(victim)
    labels = np.ones((2, 3, 4), np.int16)
    affine = np.eye(4)
    image = nib.Nifti1Image(labels, affine)
    publish_mask(target, labels, affine, cast(NiftiHeader, image.header))
    assert victim.read_bytes() == b"keep this file"
    np.testing.assert_array_equal(read_nifti(target).data, labels)
