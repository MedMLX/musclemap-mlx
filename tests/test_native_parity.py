"""Native Metal execution against recorded MONAI 1.3.2 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from medmlx_core.outputs import validate_outputs

from musclemap_mlx import medmlx
from musclemap_mlx.config import default_spec
from musclemap_mlx.geometry import read_nifti
from musclemap_mlx.network import MuscleMapRuntime
from musclemap_mlx.pipeline import infer_volume
from musclemap_mlx.weights import load_network

pytestmark = [pytest.mark.metal, pytest.mark.oracle]


def test_network_64_shapes_and_finiteness(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    with np.load(reference_dir / "network64.npz", allow_pickle=False) as fixture:
        network_input = fixture["input"]
    network = load_network(weights_dir, default_spec())
    result = network(mx.array(network_input))
    mx.eval(result)
    logits = cast(npt.NDArray[np.float32], np.asarray(result))
    assert logits.shape == (1, 114, 64, 64)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()
    output = Path("build/msk-porting/parity")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "network64_candidate.npz", logits=logits)


def test_small_pipeline_labels_match_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "small_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data, volume.affine, default_spec(), network, chunk_size=None, overlap_percent=90
    )
    assert (chunk_size, chunked) == (4, False)
    with np.load(reference_dir / "pipeline.npz", allow_pickle=False) as fixture:
        expected = fixture["labels"]
    assert np.array_equal(labels, expected)


def test_public_runner_writes_the_recorded_mask(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path, tmp_path: Path
) -> None:
    result = medmlx.segment(
        image_path=reference_dir / "small_input.nii.gz",
        weights_path=weights_dir,
        output_dir=tmp_path,
        chunk_size=4,
        overlap_percent=90,
    )
    outputs = cast(dict[str, str], result["outputs"])
    _, artifacts = validate_outputs(result)
    assert artifacts["mask"]["kind"] == "label_map"
    assert artifacts["mask"].get("grid_of") == "image_path"
    assert artifacts["mask"].get("labels") == [
        {"value": int(item["value"]), "name": str(item["name"])}
        for item in default_spec().label_details
    ]
    mask = np.asanyarray(read_nifti(Path(outputs["mask"])).image.dataobj)
    with np.load(reference_dir / "pipeline.npz", allow_pickle=False) as fixture:
        expected = fixture["labels"]
    assert np.array_equal(mask, expected)
    details = cast(dict[str, object], result["details"])
    assert cast(bool, details["chunked"]) is False
    assert cast(int, details["chunk_size"]) == 4
    assert len(cast(list[dict[str, str | int]], details["labels"])) == 114
    assert cast(bool, details["requires_human_review"]) is True
    assert cast(int, details["peak_memory_bytes"]) > 0
    assert cast(float, details["duration_seconds"]) > 0


def test_large_whole_pipeline_matches_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "large_oblique_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data,
        volume.affine,
        default_spec(),
        network,
        chunk_size=int(volume.data.shape[2]),
        overlap_percent=90,
    )
    assert (chunk_size, chunked) == (20, False)
    with np.load(reference_dir / "large_oblique.npz", allow_pickle=False) as fixture:
        expected = fixture["labels_whole"]
    assert np.array_equal(labels, expected)


def test_large_default_chunking_matches_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "large_oblique_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data,
        volume.affine,
        default_spec(),
        network,
        chunk_size=None,
        overlap_percent=90,
    )
    assert (chunk_size, chunked) == (8, True)
    with np.load(reference_dir / "large_oblique.npz", allow_pickle=False) as fixture:
        expected = fixture["labels_chunked"]
    assert np.array_equal(labels, expected)


def test_large_chunk_size_two_matches_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "large_oblique_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data,
        volume.affine,
        default_spec(),
        network,
        chunk_size=2,
        overlap_percent=90,
    )
    assert (chunk_size, chunked) == (2, True)
    with np.load(reference_dir / "large_oblique.npz", allow_pickle=False) as fixture:
        expected = fixture["labels_chunked2"]
    assert np.array_equal(labels, expected)
    assert np.count_nonzero(labels) > 0


def test_auto_chunking_is_eight_slices(mx: MuscleMapRuntime, weights_dir: Path) -> None:
    data = np.zeros((64, 64, 10), dtype=np.float32)
    ys, xs = np.mgrid[0:64, 0:64]
    data[20:44, 20:44, :] = (
        10.0 + 0.2 * ys[20:44, 20:44, None] + 0.1 * xs[20:44, 20:44, None]
    ).astype(np.float32)
    data[28:36, 28:36, :] = -5.0
    network = load_network(weights_dir, default_spec())
    labels, chunk_size, chunked = infer_volume(
        data,
        np.eye(4, dtype=np.float64),
        default_spec(),
        network,
        chunk_size=None,
        overlap_percent=90,
    )
    assert (chunk_size, chunked) == (8, True)
    assert labels.shape == data.shape
    assert labels.dtype == np.int16
    assert set(np.unique(labels).tolist()) <= set(default_spec().class_labels)


def test_chunked_pipeline_labels_match_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "small_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data, volume.affine, default_spec(), network, chunk_size=2, overlap_percent=90
    )
    assert (chunk_size, chunked) == (2, True)
    with np.load(reference_dir / "pipeline_chunked.npz", allow_pickle=False) as fixture:
        expected = fixture["labels"]
    assert np.array_equal(labels, expected)


def test_window_pipeline_labels_match_monai(
    mx: MuscleMapRuntime, weights_dir: Path, reference_dir: Path
) -> None:
    network = load_network(weights_dir, default_spec())
    volume = read_nifti(reference_dir / "window_input.nii.gz")
    labels, chunk_size, chunked = infer_volume(
        volume.data, volume.affine, default_spec(), network, chunk_size=None, overlap_percent=90
    )
    assert (chunk_size, chunked) == (2, False)
    with np.load(reference_dir / "windows.npz", allow_pickle=False) as fixture:
        expected = fixture["labels"]
    assert np.array_equal(labels, expected)
