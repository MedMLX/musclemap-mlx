"""NumPy port of the MuscleMap v1.4 MONAI 1.3.2 image geometry.

The publisher's inference transforms are ``LoadImaged`` -> ``EnsureChannelFirstd``
-> ``Orientationd("RAS")`` -> ``Spacingd([1, 1, -1], bilinear)`` ->
``NormalizeIntensityd(nonzero=True)`` -> ``CropForegroundd(margin=20)`` ->
``SpatialPadd((256, 256, 1), method="end")``. ``Invertd`` reverses the spatial
transforms on the logits with the same bilinear mode before ``argmax``.

Only host geometry runs here. The network is native MLX in ``network.py``. NIfTI
reading and admission, header correction, orientation and the spacing grid come
from ``medmlx_core``. The forward spacing resampler is a local port of MONAI
1.3.2's ATen ``grid_sample`` arithmetic; the inverse uses
``scipy.ndimage.affine_transform`` in float64 with border clamping, which matches
MONAI's ``grid_sample(align_corners=False, padding_mode="border",
mode="bilinear")`` pull mapping.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import nibabel as nib_runtime
import numpy as np
import numpy.typing as npt
from medmlx_core.geometry import _linspace_from_neg_one, _mm
from medmlx_core.transforms import (
    axis_codes,
    correct_nifti_header,
    orient,
    spacing_grid,
)
from medmlx_core.volumes import NiftiHeader, Volume, read_volume
from scipy import ndimage

from musclemap_mlx.config import (
    CROP_MARGIN,
    PAD_SIZE,
    ModelSpec,
)

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]
type Int16Array = npt.NDArray[np.int16]
type Shape3 = tuple[int, int, int]


class _NiftiImage(Protocol):
    def set_qform(self, affine: Float64Array, code: int) -> None: ...

    def set_sform(self, affine: Float64Array, code: int) -> None: ...


class _Nibabel(Protocol):
    """The nibabel writer calls whose header handling ``write_like`` does not share."""

    def Nifti1Image(
        self, dataobj: npt.ArrayLike, affine: Float64Array, header: NiftiHeader
    ) -> _NiftiImage: ...

    def save(self, img: _NiftiImage, filename: str) -> None: ...


nib = cast(_Nibabel, nib_runtime)

SPATIAL_AXCODES = "RAS"


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """Every grid needed to invert the preprocessing back to the reader grid."""

    original_shape: Shape3
    original_affine: Float64Array
    oriented_shape: Shape3
    oriented_affine: Float64Array
    spaced_shape: Shape3
    spaced_affine: Float64Array
    crop_start: Shape3
    crop_shape: Shape3
    padded_shape: Shape3


@dataclass(frozen=True, slots=True)
class ReadVolume:
    """An admitted source volume and the MONAI-corrected affine preprocessing uses.

    ``source`` keeps the file's header for the chunk round trip and the output
    mask; ``affine`` is the reader affine after MONAI's header correction.
    """

    source: Volume[np.float32]
    affine: Float64Array

    @property
    def data(self) -> Float32Array:
        """Float32 voxels in file axis order with slope and intercept applied."""
        return self.source.data


@dataclass(frozen=True, slots=True)
class PreparedVolume:
    """The padded network input and the geometry that produced it."""

    data: Float32Array
    geometry: VolumeGeometry


def read_nifti(path: Path) -> ReadVolume:
    """Read a 3D NIfTI as float32 and apply MONAI 1.3.2's reader header correction.

    ``read_volume`` admits the file and converts meter or micron headers to
    millimeters; the correction then runs on those millimeter forms. When the
    header zooms disagree with the affine, the compatible sform/qform is kept or
    the zooms are reset to the affine spacing, as MONAI's nibabel reader does.

    Raises:
        InvalidInputError: ``read_volume`` refuses the file, or its sform cannot be
            stored as a qform quaternion.
    """
    source = read_volume(path)
    # NIfTI-2 headers (540 bytes) store forms in float64, NIfTI-1 in float32.
    header_float = np.float64 if int(source.header["sizeof_hdr"]) == 540 else np.float32
    affine, _ = correct_nifti_header(
        source.affine,
        source.zooms,
        source.sform,
        source.sform_code,
        source.qform,
        source.qform_code,
        header_float=header_float,
    )
    return ReadVolume(source=source, affine=affine)


def preprocess(data: Float32Array, affine: Float64Array, spec: ModelSpec) -> PreparedVolume:
    """Apply the publisher's orientation, spacing, normalization, crop and pad."""
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    channel_first = np.ascontiguousarray(data[None], dtype=np.float32)
    original_shape = _shape3(channel_first.shape[1:])

    oriented, oriented_affine = orient(channel_first, affine, SPATIAL_AXCODES)
    oriented_shape = _shape3(oriented.shape[1:])

    spaced_affine, spaced_shape = spacing_grid(oriented_affine, oriented_shape, spec.pix_dim)
    spaced = _resample_monai(
        oriented, np.linalg.solve(oriented_affine, spaced_affine), spaced_shape
    )
    normalized = _normalize_nonzero(spaced)
    crop_start, crop_end = _foreground_box(normalized[0], CROP_MARGIN)
    if any(end <= start for start, end in zip(crop_start, crop_end, strict=True)):
        raise ValueError("MuscleMap found no positive foreground after normalization")
    crop_shape = _shape3(
        tuple(end - start for start, end in zip(crop_start, crop_end, strict=True))
    )
    cropped = np.ascontiguousarray(
        normalized[
            :,
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ]
    )
    pad_width = [(0, 0)] + [
        (0, max(PAD_SIZE[index] - cropped.shape[index + 1], 0)) for index in range(3)
    ]
    padded = np.pad(cropped, pad_width, mode="constant").astype(np.float32, copy=False)
    geometry = VolumeGeometry(
        original_shape=original_shape,
        original_affine=affine,
        oriented_shape=oriented_shape,
        oriented_affine=oriented_affine,
        spaced_shape=spaced_shape,
        spaced_affine=spaced_affine,
        crop_start=_shape3(crop_start),
        crop_shape=crop_shape,
        padded_shape=_shape3(padded.shape[1:]),
    )
    return PreparedVolume(data=np.ascontiguousarray(padded), geometry=geometry)


def read_chunk(volume: ReadVolume, start: int, end: int) -> ReadVolume:
    """Read a chunk after upstream's source-header NIfTI serialization.

    ``mm_util._write_temp_chunk`` writes float32 pixels with the original storage
    dtype. NiBabel consequently rescales integer images, including their zeros.
    This round trip must precede nonzero normalization to match that contract,
    so it uses nibabel directly: ``write_like`` casts to the stored dtype without
    rescaling and keeps NIfTI-2, where upstream writes NIfTI-1.
    """
    header = volume.source.header
    with TemporaryDirectory(prefix="musclemap-chunk-") as temporary:
        path = Path(temporary) / "chunk.nii"
        image = nib.Nifti1Image(
            volume.data[..., start:end], header.get_best_affine(), header.copy()
        )
        nib.save(image, str(path))
        return read_nifti(path)


def invert_logits(logits: Float32Array, geometry: VolumeGeometry) -> Float32Array:
    """Reverse pad, crop, spacing and orientation on channel-first logits."""
    if logits.ndim != 4:
        raise ValueError(f"Expected channel-first logits, got shape {logits.shape}")
    pad_shape = geometry.padded_shape
    if tuple(logits.shape[1:]) != pad_shape:
        raise ValueError(
            f"Logits spatial shape {logits.shape[1:]} does not match the padded grid {pad_shape}"
        )
    cropped = logits[
        :,
        : geometry.crop_shape[0],
        : geometry.crop_shape[1],
        : geometry.crop_shape[2],
    ]
    spaced = np.zeros((logits.shape[0], *geometry.spaced_shape), dtype=np.float32)
    start_h, start_w, start_d = geometry.crop_start
    size_h, size_w, size_d = geometry.crop_shape
    spaced[
        :,
        start_h : start_h + size_h,
        start_w : start_w + size_w,
        start_d : start_d + size_d,
    ] = cropped
    inverse_spacing = np.linalg.solve(geometry.spaced_affine, geometry.oriented_affine)
    oriented = _resample(spaced, inverse_spacing, geometry.oriented_shape)
    original, _ = orient(
        oriented, geometry.oriented_affine, axis_codes(geometry.original_affine)
    )
    if tuple(original.shape[1:]) != geometry.original_shape:
        raise RuntimeError("MuscleMap inverse geometry did not restore the reader grid")
    return np.ascontiguousarray(original)


def discretize(logits: Float32Array, spec: ModelSpec) -> Int16Array:
    """Take MONAI's channel argmax and remap class indices to original label IDs."""
    class_indices = np.argmax(logits, axis=0).astype(np.int16, copy=False)
    mapping = np.asarray(spec.class_labels, dtype=np.int16)
    return np.ascontiguousarray(mapping[class_indices])


def keep_largest_components(labels: Int16Array) -> Int16Array:
    """Keep the largest 6-connected component for every nonzero label.

    This mirrors ``mm_util.connected_chunks(..., connectivity=1)``. It is host
    numerical postprocessing, not network math.
    """
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got shape {labels.shape}")
    result = np.array(labels, dtype=np.int16, copy=True)
    structure = ndimage.generate_binary_structure(3, 1)
    values = np.unique(result)
    # Every voxel of a label, hence every component, lies in the label's bounding
    # box, and labelling the box visits voxels in the same raster order, so the
    # components, their numbering and the kept one match a full-volume pass. The
    # uint16 view gives each nonzero int16 ID its own find_objects index.
    boxes = ndimage.find_objects(result.view(np.uint16))
    for label, index in zip(values.tolist(), values.view(np.uint16).tolist(), strict=True):
        if label == 0:
            continue
        # find_objects returns None only for absent IDs; ``values`` are present.
        region = result[boxes[index - 1]]
        mask = region == label
        components, count = ndimage.label(mask, structure=structure)
        if int(count) <= 1:
            continue
        sizes = np.bincount(components.ravel())
        keep = int(np.argmax(sizes[1:])) + 1
        region[mask & (components != keep)] = 0
    return np.ascontiguousarray(result)


def write_mask(path: Path, labels: Int16Array, source: Volume[np.float32]) -> None:
    """Write int16 anatomical IDs on the source grid with its header and units.

    Both forms are stored again from the source header: a coded form keeps its
    matrix and code, an uncoded one is written from the source affine with code 0.
    ``write_like`` keeps the stored forms and pixdim untouched instead, which
    changes the header for files with uncoded forms or pixdim that disagrees with
    the sform.
    """
    header = source.header.copy()
    # NIfTI constructors retain the header dtype even when the array is int16.
    header.set_data_dtype(np.int16)
    affine = source.header.get_best_affine()
    image = nib.Nifti1Image(labels.astype(np.int16, copy=False), affine, header)
    qform = source.header.get_qform() if source.qform_code else affine
    sform = source.header.get_sform() if source.sform_code else affine
    image.set_qform(qform, source.qform_code)
    image.set_sform(sform, source.sform_code)
    nib.save(image, str(path))


def _shape3(shape: tuple[int, ...]) -> Shape3:
    if len(shape) != 3:
        raise ValueError(f"Expected three spatial dimensions, got {shape}")
    return (int(shape[0]), int(shape[1]), int(shape[2]))


_FMA_SPLIT = 134217729.0


def _two_sum(left: Float64Array, right: Float64Array) -> tuple[Float64Array, Float64Array]:
    total = left + right
    shifted = total - left
    return total, (left - (total - shifted)) + (right - shifted)


def _two_product(left: Float64Array, right: Float64Array) -> tuple[Float64Array, Float64Array]:
    product = left * right
    scaled_left = _FMA_SPLIT * left
    high_left = scaled_left - (scaled_left - left)
    low_left = left - high_left
    scaled_right = _FMA_SPLIT * right
    high_right = scaled_right - (scaled_right - right)
    low_right = right - high_right
    error = (
        ((high_left * high_right - product) + high_left * low_right)
        + low_left * high_right
    ) + low_left * low_right
    return product, error


def _fma(
    left: Float64Array | np.float64,
    right: Float64Array | np.float64,
    addend: Float64Array | np.float64,
) -> Float64Array:
    """Correctly rounded float64 ``left * right + addend`` (Veltkamp/Dekker)."""
    left_array, right_array, addend_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(addend, dtype=np.float64),
    )
    product, product_error = _two_product(left_array, right_array)
    total, total_error = _two_sum(addend_array, product)
    tail, tail_error = _two_sum(total_error, product_error)
    even = (tail.view(np.int64) & 1) == 0
    toward = np.where(tail_error > 0, np.inf, -np.inf)
    tail = np.where((tail_error != 0) & even, np.nextafter(tail, toward), tail)
    return total + tail


def _normalize_transform(shape: Shape3) -> Float64Array:
    values = np.asarray(shape, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(2.0 / values)
    transform[:3, 3] = 1.0 / values - 1.0
    return transform


def _grid_theta(
    transform: Float64Array, source_shape: Shape3, output_shape: Shape3
) -> Float64Array:
    source = _normalize_transform(source_shape)
    destination = _normalize_transform(output_shape)
    theta = _mm(_mm(source, transform), np.linalg.inv(destination))
    reverse = [2, 1, 0]
    theta = theta.copy()
    theta[:3, :3] = theta[:3, :3][np.ix_(reverse, reverse)]
    theta[:3, 3] = theta[:3, 3][reverse]
    return theta


def _resample_monai(
    data: Float32Array, transform: Float64Array, output_shape: Shape3
) -> Float32Array:
    """MONAI 1.3.2 ``SpatialResample`` bilinear border sampling.

    This reproduces ATen's 3D CPU ``grid_sample`` arithmetic, including the
    fused multiply-add in source-coordinate unnormalization. The exact zero
    pattern matters because ``NormalizeIntensityd(nonzero=True)`` uses
    ``input != 0``: a different resampler can silently change normalization
    statistics on oblique geometry.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected channel-first data, got shape {data.shape}")
    spatial = data[0]
    source_shape = _shape3(tuple(spatial.shape))
    depth, height, width = source_shape
    out_depth, out_height, out_width = output_shape
    theta = _grid_theta(transform, source_shape, output_shape)
    x_coords = _linspace_from_neg_one(out_width)
    y_coords = _linspace_from_neg_one(out_height)
    z_coords = _linspace_from_neg_one(out_depth)
    base = np.empty((out_depth, out_height, out_width, 4), dtype=np.float64)
    base[..., 0] = x_coords
    base[..., 1] = y_coords[None, :, None]
    base[..., 2] = z_coords[:, None, None]
    base[..., 3] = 1.0
    grid = (base.reshape(-1, 4) @ theta[:3, :].T).reshape(
        out_depth, out_height, out_width, 3
    )
    ix = _fma(grid[..., 0] + 1.0, np.float64(width), np.float64(-1.0)) / 2.0
    iy = _fma(grid[..., 1] + 1.0, np.float64(height), np.float64(-1.0)) / 2.0
    iz = _fma(grid[..., 2] + 1.0, np.float64(depth), np.float64(-1.0)) / 2.0
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    iz = np.clip(iz, 0, depth - 1)
    x0 = np.floor(ix).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    z0 = np.floor(iz).astype(np.int64)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    tnw = (x1 - ix) * (y1 - iy) * (z1 - iz)
    tne = (ix - x0) * (y1 - iy) * (z1 - iz)
    tsw = (x1 - ix) * (iy - y0) * (z1 - iz)
    tse = (ix - x0) * (iy - y0) * (z1 - iz)
    bnw = (x1 - ix) * (y1 - iy) * (iz - z0)
    bne = (ix - x0) * (y1 - iy) * (iz - z0)
    bsw = (x1 - ix) * (iy - y0) * (iz - z0)
    bse = (ix - x0) * (iy - y0) * (iz - z0)
    corners = (
        (z0, y0, x0, tnw),
        (z0, y0, x1, tne),
        (z0, y1, x0, tsw),
        (z0, y1, x1, tse),
        (z1, y0, x0, bnw),
        (z1, y0, x1, bne),
        (z1, y1, x0, bsw),
        (z1, y1, x1, bse),
    )
    output = np.empty((data.shape[0], *output_shape), dtype=np.float32)
    for channel in range(data.shape[0]):
        channel_data = np.asarray(data[channel], dtype=np.float64)
        flattened = channel_data.reshape(-1)
        result = np.zeros((out_depth, out_height, out_width), dtype=np.float64)
        for zz, yy, xx, weight in corners:
            valid = (
                (zz >= 0)
                & (zz < depth)
                & (yy >= 0)
                & (yy < height)
                & (xx >= 0)
                & (xx < width)
            )
            index = (
                zz.clip(0, depth - 1) * height * width
                + yy.clip(0, height - 1) * width
                + xx.clip(0, width - 1)
            )
            values = flattened[index] * valid
            result = _fma(values, weight, result)
        output[channel] = result.astype(np.float32, copy=False)
    return output


def _resample(
    data: Float32Array, transform: Float64Array, output_shape: Shape3
) -> Float32Array:
    """Resample independent channels with scipy's bilinear border sampler.

    SciPy accumulates interpolation in double precision even for float32 input.
    Writing directly into the float32 destination avoids two full-channel copies.
    Its sampler releases the GIL; at most eight channels run concurrently.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected channel-first data, got shape {data.shape}")
    matrix = np.asarray(transform[:3, :3], dtype=np.float64)
    offset = np.asarray(transform[:3, 3], dtype=np.float64)
    output = np.empty((data.shape[0], *output_shape), dtype=np.float32)
    def resample_channel(channel: int) -> None:
        ndimage.affine_transform(
            data[channel],
            matrix=matrix,
            offset=offset,
            output_shape=output_shape,
            output=output[channel],
            order=1,
            mode="nearest",
            prefilter=False,
        )
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Consume the iterator so worker failures propagate before returning.
        for _ in executor.map(resample_channel, range(data.shape[0])):
            pass
    return output

def _normalize_nonzero(data: Float32Array) -> Float32Array:
    output = np.array(data, dtype=np.float32, copy=True)
    mask = output != 0
    if not bool(mask.any()):
        return output
    values = output[mask]
    mean = np.float32(values.mean(dtype=np.float32))
    deviation = np.float32(values.std(dtype=np.float32, ddof=0))
    if deviation == 0:
        deviation = np.float32(1.0)
    output[mask] = (values - mean) / deviation
    return output


def _foreground_box(volume: Float32Array, margin: int) -> tuple[Shape3, Shape3]:
    mask = volume > 0
    if not bool(mask.any()):
        return (0, 0, 0), (0, 0, 0)
    starts: list[int] = []
    ends: list[int] = []
    for axis in range(3):
        other = tuple(index for index in range(3) if index != axis)
        present = np.flatnonzero(np.any(mask, axis=other))
        if present.size == 0:
            return (0, 0, 0), (0, 0, 0)
        starts.append(max(int(present[0]) - margin, 0))
        ends.append(min(int(present[-1]) + margin + 1, mask.shape[axis]))
    return _shape3(tuple(starts)), _shape3(tuple(ends))
