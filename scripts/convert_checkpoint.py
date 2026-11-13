"""Convert the MuscleMap v1.4 checkpoint with the standard safetensors writer.

Run with the project environment:

    .venv/bin/python scripts/convert_checkpoint.py \
        --checkpoint weights/contrast_agnostic_wholebody_model.pth \
        --config weights/contrast_agnostic_wholebody_model.json \
        --output build/msk-porting/weights/model.safetensors

Torch is used only to deserialize the publisher state dictionary with
``weights_only=True`` and to copy tensors to NumPy. No Torch graph or model is
constructed here, and the runtime never imports Torch. The script checks asset
completeness, the exact 46-tensor inventory, shapes,
dtypes and finiteness, then reloads safetensors and compares every tensor exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from musclemap_mlx.assets import verify_assets
from musclemap_mlx.config import (
    CONFIG_NAME,
    UPSTREAM_REVISION,
    expected_parameters,
    load_model_config,
)

CONVERTER = f"musclemap-mlx {version("musclemap-mlx")}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Publisher asset folder (default: checkpoint parent)",
    )
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().absolute()
    config = args.config.expanduser().absolute()
    output = args.output.expanduser().absolute()
    assets_dir = (args.assets_dir or checkpoint.parent).expanduser().absolute()
    verify_assets(assets_dir)
    spec = load_model_config(config)
    inventory = expected_parameters(spec)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint.name} is not a state-dictionary mapping")
    arrays = _extract(payload, inventory.shapes)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "source": checkpoint.name,
        "upstream_revision": UPSTREAM_REVISION,
        "converter": CONVERTER,
    }
    save_file(arrays, output, metadata=metadata)
    reloaded = load_file(output)
    if set(reloaded) != set(arrays):
        raise ValueError("written safetensors inventory differs from the source")
    for name, value in arrays.items():
        if not np.array_equal(reloaded[name], value):
            raise ValueError(f"written tensor differs from the source: {name}")
    _copy_metadata(config, output.parent, assets_dir)
    record = {
        "format": "musclemap-mlx-safetensors-v1",
        "upstream_revision": UPSTREAM_REVISION,
        "source": {
            "name": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
        },
        "config": {
            "name": config.name,
        },
        "converter": CONVERTER,
        "tensors": [
            {"name": name, "shape": list(shape), "dtype": "float32"}
            for name, shape in sorted(inventory.shapes.items())
        ],
        "output": {
            "name": output.name,
            "bytes": output.stat().st_size,
        },
    }
    (output.parent / "conversion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _extract(
    payload: dict[object, object], expected: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    names = {str(name) for name in payload}
    if names != set(expected):
        missing = sorted(set(expected) - names)
        extra = sorted(names - set(expected))
        raise ValueError(f"checkpoint inventory differs: missing={missing[:4]} extra={extra[:4]}")
    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {}
    for name, shape in expected.items():
        value = cast(object, payload[name])
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint entry {name} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"checkpoint tensor {name} has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.dtype != torch.float32:
            raise ValueError(
                f"checkpoint tensor {name} has dtype {value.dtype}, expected float32"
            )
        array = value.detach().cpu().contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"checkpoint tensor {name} contains non-finite values")
        arrays[name] = np.ascontiguousarray(array, dtype=np.float32)
    return arrays


def _copy_metadata(config: Path, output_dir: Path, assets_dir: Path) -> None:
    target_config = output_dir / "config.json"
    if config != target_config:
        shutil.copyfile(config, target_config)
    license_path = assets_dir / "LICENSE"
    if license_path.is_file():
        target_license = output_dir / "LICENSE"
        if license_path != target_license:
            shutil.copyfile(license_path, target_license)
    source_config = assets_dir / CONFIG_NAME
    if source_config.is_file() and source_config != target_config:
        shutil.copyfile(source_config, output_dir / CONFIG_NAME)



if __name__ == "__main__":
    raise SystemExit(main())
