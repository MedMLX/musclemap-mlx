"""Record independent MONAI 1.3.2 reference outputs for the MuscleMap port.

Run in the pinned oracle environment (Python 3.11, torch 2.4.1, MONAI 1.3.2):

    PYTHONPATH=src environment/oracle/.venv/bin/python \
        scripts/make_reference_fixtures.py --assets weights --output build/msk-porting/fixtures

The script uses only synthetic arrays and the publisher checkpoint. It records
the exact upstream preprocessing stages, 64x64 network logits, the full small
pipeline labels and inverted logits, and a larger sliding-window case. The
resulting .npz files are development artifacts and stay ignored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import scipy
import torch
from monai.inferers import SliceInferer
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    SpatialPadd,
)
from scipy import ndimage as ndi

from musclemap_mlx.assets import verify_assets
from musclemap_mlx.config import CONFIG_NAME, load_model_config

SCHEMA_VERSION = 1
SMALL_SHAPE = (100, 90, 4)
WINDOW_SHAPE = (320, 220, 2)
OBLIQUE_SHAPE = (44, 40, 3)
LARGE_SHAPE = (280, 260, 20)
AFFINE = np.array(
    [
        [-0.9, 0.0, 0.0, 12.0],
        [0.0, -1.3, 0.0, -7.0],
        [0.0, 0.0, 2.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _rotated_affine(
    angle_deg: float, spacing: tuple[float, float, float], translation: tuple[float, float, float]
) -> np.ndarray:
    """An in-plane rotated, anisotropic RAS-like affine for geometry coverage."""
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = np.diag(np.asarray(spacing, dtype=np.float64))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ scale
    affine[:3, 3] = np.asarray(translation, dtype=np.float64)
    return affine


OBLIQUE_AFFINE = _rotated_affine(18.0, (0.9, 1.25, 2.5), (10.0, -6.0, 20.0))
LARGE_AFFINE = _rotated_affine(8.0, (1.0, 1.2, 2.5), (15.0, -9.0, 25.0))


class SqueezeTransform(MapTransform):
    """Remove a leading singleton channel after argparse, as upstream does."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        super().__init__(keys)

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        output = dict(data)
        for key in self.keys:
            value = output[key]
            if isinstance(value, torch.Tensor) and value.dim() > 3 and value.shape[0] == 1:
                output[key] = value.squeeze(0)
        return output


class RemapLabels(MapTransform):
    """Restore the publisher's anatomical IDs from network channel indices."""

    def __init__(self, keys: tuple[str, ...], id_map: dict[int, int]) -> None:
        super().__init__(keys)
        self.id_map = id_map

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        output = dict(data)
        for key in self.keys:
            value = output[key]
            if not isinstance(value, torch.Tensor):
                continue
            result = value.clone()
            for source, target in self.id_map.items():
                result[value == source] = target
            output[key] = result
        return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="Record a real NIfTI instead of synthetic cases")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args(argv)
    assets = args.assets.expanduser().absolute()
    output = args.output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    verify_assets(assets)
    config_path = assets / CONFIG_NAME
    load_model_config(config_path)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    state = torch.load(
        assets / "contrast_agnostic_wholebody_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    network = _network(config_path)
    network.load_state_dict(state)
    network.eval()

    if args.image is not None:
        _record_real(args.image, network, config_path, output, args.chunk_size)
        return 0

    small_image = _phantom(SMALL_SHAPE, (50, 45, 2), (13, 10, 2), seed=7)
    small_path = output / "small_input.nii.gz"
    _write_nifti(small_path, small_image, AFFINE)
    preprocess, stages, final = _run_stages(small_path, config_path)
    _save_preprocess(output / "preprocess.npz", small_image, stages, final)

    inverted, raw_labels = _run_pipeline(
        preprocess, final, network, config_path, filter_connected=False
    )
    labels = connected_chunks(raw_labels)
    reader_affine = np.asarray(stages["loaded_affine"], dtype=np.float64)
    np.savez_compressed(
        output / "pipeline.npz",
        input=small_image,
        input_affine=reader_affine,
        inverted_logits=np.ascontiguousarray(inverted),
        labels_unfiltered=np.ascontiguousarray(raw_labels),
        labels=np.ascontiguousarray(labels),
    )
    chunked_labels = _run_chunked_pipeline(small_path, network, config_path, chunk_size=2)
    np.savez_compressed(
        output / "pipeline_chunked.npz",
        input=small_image,
        input_affine=reader_affine,
        labels=np.ascontiguousarray(chunked_labels),
    )
    inverse_rng = np.random.default_rng(99)
    padded_logits = inverse_rng.normal(size=(4, 256, 256, 4)).astype(np.float32)
    inverted_logits = _invert_logits(preprocess, final, padded_logits)
    np.savez_compressed(
        output / "inverse.npz",
        padded_logits=padded_logits,
        inverted_logits=np.ascontiguousarray(inverted_logits),
    )

    rng = np.random.default_rng(20261102)
    network_input = rng.normal(size=(1, 1, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        network_logits = network(torch.from_numpy(network_input)).numpy()
    np.savez_compressed(
        output / "network64.npz",
        input=network_input,
        logits=np.ascontiguousarray(network_logits),
    )

    window_image = _slab(WINDOW_SHAPE)
    window_path = output / "window_input.nii.gz"
    _write_nifti(window_path, window_image, AFFINE)
    window_preprocess, window_stages, window_final = _run_stages(window_path, config_path)
    _, window_labels = _run_pipeline(window_preprocess, window_final, network, config_path)
    window_affine = np.asarray(window_stages["loaded_affine"], dtype=np.float64)
    np.savez_compressed(
        output / "windows.npz",
        input=window_image,
        input_affine=window_affine,
        labels=np.ascontiguousarray(window_labels),
    )

    oblique_image = _phantom(OBLIQUE_SHAPE, (22, 20, 1), (9, 8, 1), seed=11)
    oblique_path = output / "oblique_input.nii.gz"
    _write_nifti(oblique_path, oblique_image, OBLIQUE_AFFINE)
    oblique_preprocess, oblique_stages, oblique_final = _run_stages(oblique_path, config_path)
    _save_preprocess(
        output / "preprocess_oblique.npz", oblique_image, oblique_stages, oblique_final
    )
    oblique_pad_shape = np.asarray(oblique_stages["pad_data"]).shape[1:]
    oblique_rng = np.random.default_rng(101)
    oblique_padded = oblique_rng.normal(size=(4, *oblique_pad_shape)).astype(np.float32)
    oblique_inverted = _invert_logits(oblique_preprocess, oblique_final, oblique_padded)
    np.savez_compressed(
        output / "inverse_oblique.npz",
        padded_logits=oblique_padded,
        inverted_logits=np.ascontiguousarray(oblique_inverted),
    )

    large_image = _large_slab(LARGE_SHAPE)
    large_path = output / "large_oblique_input.nii.gz"
    _write_nifti(large_path, large_image, LARGE_AFFINE)
    large_preprocess, large_stages, large_final = _run_stages(large_path, config_path)
    _save_preprocess(output / "preprocess_large.npz", large_image, large_stages, large_final)
    _, raw_whole = _run_pipeline(
        large_preprocess, large_final, network, config_path, filter_connected=False
    )
    large_whole = connected_chunks(raw_whole)
    large_chunked = _run_chunked_pipeline(large_path, network, config_path, chunk_size=8)
    large_chunked2 = _run_chunked_pipeline(large_path, network, config_path, chunk_size=2)
    large_reader_affine = np.asarray(large_stages["loaded_affine"], dtype=np.float64)
    np.savez_compressed(
        output / "large_oblique.npz",
        input=large_image,
        input_affine=large_reader_affine,
        labels_whole=np.ascontiguousarray(large_whole),
        labels_chunked=np.ascontiguousarray(large_chunked),
        labels_chunked2=np.ascontiguousarray(large_chunked2),
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "upstream": {
            "repository": "https://github.com/MuscleMap/MuscleMap",
            "revision": "35a2138a4a3a63be713bae5584632ffc6433b960",
            "zenodo_record": "https://zenodo.org/records/21929873",
        },
        "versions": {
            "torch": torch.__version__,
            "monai": _module_version("monai"),
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "scipy": scipy.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "cases": {
            "preprocess": {"shape": list(SMALL_SHAPE), "affine": AFFINE.tolist()},
            "network64": {"shape": [1, 1, 64, 64], "seed": 20261102},
            "pipeline": {"shape": list(SMALL_SHAPE), "affine": AFFINE.tolist()},
            "windows": {"shape": list(WINDOW_SHAPE), "affine": AFFINE.tolist()},
            "oblique": {"shape": list(OBLIQUE_SHAPE), "affine": OBLIQUE_AFFINE.tolist()},
            "large": {"shape": list(LARGE_SHAPE), "affine": LARGE_AFFINE.tolist()},
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _record_real(
    path: Path, network: UNet, config: Path, output: Path, chunk_size: int
) -> None:
    """Record upstream float32 chunk logits and the assembled anatomical mask.

    Chunk NIfTI construction follows upstream ``_write_temp_chunk``. Component
    filtering runs once after assembly, never independently on each chunk.
    """
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    started = time.perf_counter()
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("A real reference input must be three-dimensional")
    full = np.zeros(data.shape, dtype=np.int16)
    with TemporaryDirectory(prefix="chunks-", dir=output) as temporary:
        for start in range(0, data.shape[2], chunk_size):
            end = min(start + chunk_size, data.shape[2])
            chunk = data[..., start:end]
            chunk_path = Path(temporary) / "input.nii"
            nib.save(nib.Nifti1Image(chunk, image.affine, image.header.copy()), str(chunk_path))
            transforms, stages, final = _run_stages(chunk_path, config)
            inverted, labels = _run_pipeline(
                transforms, final, network, config, filter_connected=False
            )
            full[..., start:end] = labels
            prepared = final["image"]
            window = prepared[None, :, :256, :256, :1].squeeze(-1)
            with torch.inference_mode():
                network_logits = network(window).detach().cpu().numpy()
            np.savez_compressed(
                output / f"chunk-{start}.npz", input=np.asarray(stages["loaded_data"]),
                affine=np.asarray(stages["loaded_affine"]),
                prepared=np.asarray(prepared), network_input=np.asarray(window),
                network_logits=network_logits, inverted_logits=inverted,
            )
            print(f"Recorded slices {start}:{end}", flush=True)
    np.save(output / "labels.npy", connected_chunks(full))
    metadata = {
        "upstream_revision": "35a2138a4a3a63be713bae5584632ffc6433b960",
        "checkpoint_record": "https://zenodo.org/records/21929873",
        "input_name": path.name, "shape": list(data.shape),
        "affine": image.affine.tolist(), "units": image.header.get_xyzt_units(),
        "chunk_size": chunk_size, "overlap": 0.9, "dtype": "float32",
        "torch": torch.__version__, "monai": _module_version("monai"),
        "numpy": np.__version__, "nibabel": nib.__version__, "scipy": scipy.__version__,
        "duration_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _network(config_path: Path) -> UNet:
    spec = load_model_config(config_path)
    return UNet(
        spatial_dims=spec.spatial_dims,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        channels=spec.channels,
        act=spec.act,
        strides=spec.strides,
        num_res_units=spec.num_res_units,
        norm=Norm.INSTANCE,
    )


def _transforms(config_path: Path) -> Compose:
    spec = load_model_config(config_path)
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spec.pix_dim, mode="bilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            CropForegroundd(keys=["image"], source_key="image", margin=20),
            SpatialPadd(
                keys=["image"],
                spatial_size=(256, 256, 1),
                method="end",
                mode="constant",
            ),
            EnsureTyped(keys=["image"]),
        ]
    )


def _run_stages(
    path: Path, config_path: Path
) -> tuple[Compose, dict[str, object], dict[str, object]]:
    transforms = _transforms(config_path)
    data: dict[str, object] = {"image": str(path)}
    stages: dict[str, object] = {}
    names = ["loaded", "channel", "orientation", "spacing", "normalize", "crop", "pad", "typed"]
    for name, transform in zip(names, transforms.transforms, strict=True):
        data = transform(data)
        image = data["image"]
        if isinstance(image, torch.Tensor):
            stages[f"{name}_data"] = np.array(image.detach().cpu(), copy=True)
            affine = getattr(image, "affine", None)
            if affine is not None:
                stages[f"{name}_affine"] = np.array(affine.detach().cpu(), copy=True)
    if "foreground_start_coord" in data:
        stages["crop_start"] = np.asarray(data["foreground_start_coord"], dtype=np.int64)
        stages["crop_end"] = np.asarray(data["foreground_end_coord"], dtype=np.int64)
    return transforms, stages, data


def _invert_logits(
    transforms: Compose, data: dict[str, object], padded_logits: np.ndarray
) -> np.ndarray:
    image = data["image"]
    if not isinstance(image, torch.Tensor):
        raise TypeError("upstream transform did not produce a tensor")
    invert = Invertd(
        keys="pred",
        transform=transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device=torch.device("cpu"),
    )
    post = {
        "pred": torch.from_numpy(padded_logits),
        "image": image,
        "image_meta_dict": data["image_meta_dict"],
        "image_transforms": image.applied_operations,
    }
    inverted = invert(post)["pred"]
    if not isinstance(inverted, torch.Tensor):
        raise TypeError("Invertd did not produce a tensor")
    return np.asarray(inverted.detach().cpu(), dtype=np.float32)


def _run_pipeline(
    transforms: Compose,
    data: dict[str, object],
    network: UNet,
    config_path: Path,
    *,
    filter_connected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    spec = load_model_config(config_path)
    labels = sorted({label.value for label in spec.labels})
    id_map = {0: 0}
    for new_id, original in enumerate(labels, start=1):
        id_map[original] = new_id
    inverse_map = {new_id: original for original, new_id in id_map.items()}
    image = data["image"]
    if not isinstance(image, torch.Tensor):
        raise TypeError("upstream transform did not produce a tensor")
    tensor = image
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    inferer = SliceInferer(
        roi_size=spec.roi_size,
        sw_batch_size=spec.spatial_window_batch_size,
        spatial_dim=2,
        mode="gaussian",
        overlap=0.9,
    )
    with torch.inference_mode():
        prediction = inferer(tensor, network)
    single = prediction.squeeze(0).squeeze(0)
    invert = Invertd(
        keys="pred",
        transform=transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device=torch.device("cpu"),
    )
    post = {"pred": single, "image": data["image"], "image_meta_dict": data["image_meta_dict"]}
    inverted = invert(post)
    inverted_tensor = inverted["pred"]
    if not isinstance(inverted_tensor, torch.Tensor):
        raise TypeError("Invertd did not produce a tensor")
    inverted_logits = np.asarray(inverted_tensor.detach().cpu(), dtype=np.float32)
    discrete = AsDiscreted(keys="pred", argmax=True)(inverted)
    squeezed = SqueezeTransform(keys=("pred",))(discrete)
    remapped = RemapLabels(keys=("pred",), id_map=inverse_map)(squeezed)
    result = remapped["pred"]
    if not isinstance(result, torch.Tensor):
        raise TypeError("upstream postprocessing did not produce a tensor")
    labels = np.asarray(result.detach().cpu(), dtype=np.int16)
    return inverted_logits, connected_chunks(labels) if filter_connected else labels


def _run_chunked_pipeline(
    path: Path, network: UNet, config_path: Path, *, chunk_size: int
) -> np.ndarray:
    image = nib.load(str(path))
    affine = np.asarray(image.affine, dtype=np.float64)
    data = np.asanyarray(image.dataobj)
    header = image.header.copy()
    full = np.zeros(data.shape, dtype=np.int16)
    depth = int(data.shape[-1])
    for start in range(0, depth, chunk_size):
        end = min(start + chunk_size, depth)
        chunk_path = path.parent / f"chunk_{start}_{end}.nii"
        chunk = np.asarray(data[..., start:end], dtype=np.float32)
        chunk_image = nib.Nifti1Image(chunk, affine, header.copy())
        chunk_image.set_qform(affine, 1)
        chunk_image.set_sform(affine, 1)
        nib.save(chunk_image, str(chunk_path))
        transforms, _stages, final = _run_stages(chunk_path, config_path)
        _, labels = _run_pipeline(
            transforms, final, network, config_path, filter_connected=False
        )
        full[..., start:end] = labels
        chunk_path.unlink()
    return connected_chunks(full)


def connected_chunks(seg: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Keep the largest 3D connected component of each nonzero label.

    Copied from ``mm_util.connected_chunks`` at the pinned upstream revision.
    """
    labels = np.unique(seg)
    labels = labels[labels != 0]
    result = np.array(seg, dtype=np.int16, copy=True)
    if labels.size == 0:
        return result
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    mask3d = np.empty(result.shape, dtype=bool)
    lab3d = np.empty(result.shape, dtype=np.int32)
    for label in labels:
        np.equal(result, label, out=mask3d)
        if not mask3d.any():
            continue
        ndi.label(mask3d, structure=structure, output=lab3d)
        max_lab = int(lab3d.max())
        if max_lab <= 1:
            continue
        counts = np.bincount(lab3d.ravel())
        keep = counts[1:].argmax() + 1
        np.logical_and(mask3d, lab3d != keep, out=mask3d)
        result[mask3d] = 0
    return result


def _save_preprocess(
    path: Path, image: np.ndarray, stages: dict[str, object], final: dict[str, object]
) -> None:
    final_image = final["image"]
    if not isinstance(final_image, torch.Tensor):
        raise TypeError("final transform did not produce a tensor")
    stages["input"] = image
    stages["final"] = np.asarray(final_image.detach().cpu())
    np.savez_compressed(path, **stages)


def _phantom(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    radii: tuple[int, int, int],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = [np.arange(size, dtype=np.float32) for size in shape]
    grid = np.meshgrid(*axes, indexing="ij")
    distance = sum(
        ((coordinate - center[index]) / radii[index]) ** 2
        for index, coordinate in enumerate(grid)
    )
    data = np.zeros(shape, dtype=np.float32)
    inside = distance <= 1.0
    phase = distance[inside] * np.float32(np.pi)
    data[inside] = (
        np.float32(40.0)
        + np.float32(30.0) * np.cos(phase)
        + np.float32(3.0) * rng.normal(size=int(inside.sum())).astype(np.float32)
    )
    data[distance <= 0.12] = np.float32(-25.0)
    return np.ascontiguousarray(data)


def _slab(shape: tuple[int, int, int]) -> np.ndarray:
    data = np.zeros(shape, dtype=np.float32)
    data[20:300, 10:210, :] = np.float32(100.0)
    data[20:300, 10:210, :] += np.linspace(-5.0, 5.0, 280, dtype=np.float32)[:, None, None]
    data[120:180, 80:130, :] = np.float32(-20.0)
    return np.ascontiguousarray(data)


def _large_slab(shape: tuple[int, int, int]) -> np.ndarray:
    """A deterministic large volume made by tiling the labelled synthetic phantom.

    A uniform slab made the released model predict background everywhere; tiling the
    smaller phantom keeps the multi-window and multi-chunk cases numerically active.
    """
    small = _phantom((100, 90, 4), (50, 45, 2), (13, 10, 2), seed=7)
    tiled = np.tile(small, (3, 3, 5))
    return np.ascontiguousarray(tiled[: shape[0], : shape[1], : shape[2]])


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


def _module_version(name: str) -> str:
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
