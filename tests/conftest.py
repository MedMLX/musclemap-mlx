"""Shared native runtime and opt-in reference paths."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest
from medmlx_core.typing import MlxRuntime

type Float64Array = npt.NDArray[np.float64]


class _Header(Protocol):
    def set_zooms(self, zooms: tuple[float, ...]) -> None: ...

    def set_xyzt_units(self, xyz: str, t: str) -> None: ...


class _Image(Protocol):
    header: _Header

    def set_qform(self, affine: Float64Array, code: int) -> None: ...

    def set_sform(self, affine: Float64Array, code: int) -> None: ...


class _Nibabel(Protocol):
    def Nifti1Image(self, dataobj: npt.ArrayLike, affine: Float64Array) -> _Image: ...

    def save(self, img: _Image, filename: str) -> None: ...


class SaveNifti(Protocol):
    def __call__(
        self,
        path: Path,
        data: npt.ArrayLike,
        *,
        sform: Float64Array,
        sform_code: int,
        qform: Float64Array,
        qform_code: int,
        zooms: tuple[float, float, float] | None = None,
        units: tuple[str, str] = ("mm", "unknown"),
    ) -> Path: ...


@pytest.fixture(scope="session")
def mx() -> MlxRuntime:
    """Provide the admitted MLX Metal runtime, skipping non-Apple hosts."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MuscleMap native tests require macOS arm64 with MLX Metal")
    from medmlx_core.runtime import import_mlx

    return cast(MlxRuntime, import_mlx())


@pytest.fixture
def save_nifti() -> SaveNifti:
    """Write a NIfTI-1 input with explicit forms, codes, pixdim and units."""
    import nibabel

    nib = cast(_Nibabel, nibabel)

    def save(
        path: Path,
        data: npt.ArrayLike,
        *,
        sform: Float64Array,
        sform_code: int,
        qform: Float64Array,
        qform_code: int,
        zooms: tuple[float, float, float] | None = None,
        units: tuple[str, str] = ("mm", "unknown"),
    ) -> Path:
        image = nib.Nifti1Image(data, sform)
        image.set_qform(qform, qform_code)
        image.set_sform(sform, sform_code)
        if zooms is not None:
            image.header.set_zooms(zooms)
        image.header.set_xyzt_units(*units)
        nib.save(image, str(path))
        return path

    return save


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """Return the opt-in recorded MONAI reference directory."""
    value = _env_path("MEDMLX_TEST_REFERENCE")
    if value is None:
        pytest.skip("Set MEDMLX_TEST_REFERENCE to the recorded MONAI fixture folder")
    return value


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    """Return the opt-in converted safetensors folder."""
    value = _env_path("MEDMLX_TEST_WEIGHTS")
    if value is None:
        pytest.skip("Set MEDMLX_TEST_WEIGHTS to a folder containing model.safetensors")
    return value


def _env_path(name: str) -> Path | None:
    import os

    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser().absolute()
    return path
