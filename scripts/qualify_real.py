"""Compare native Metal inference with recorded real-image MONAI 1.3.2 outputs.

Run after recording with make_reference_fixtures.py:

    .venv/bin/python scripts/qualify_real.py \
        --reference build/reference --weights build/converted --output build/parity.json

The existing float32 contract is atol=5e-4, rtol=1e-5 for network and restored
logits. Anatomical masks must match exactly, including component filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from medmlx_core.runtime import import_mlx
from medmlx_core.sliding_window import sliding_window_inference

from musclemap_mlx.config import CONVERTED_NAME, load_model_config
from musclemap_mlx.geometry import (
    discretize,
    invert_logits,
    keep_largest_components,
    preprocess,
)
from musclemap_mlx.weights import load_network


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    difference = np.abs(actual - expected)
    return {
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "allclose": bool(np.allclose(actual, expected, atol=5e-4, rtol=1e-5)),
        "finite": bool(np.isfinite(actual).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.reference / "metadata.json").read_text())
    spec = load_model_config(args.weights / "config.json")
    mx = import_mlx()
    network = load_network(args.weights / CONVERTED_NAME, spec)
    expected = np.load(args.reference / "labels.npy", allow_pickle=False)
    full = np.zeros(expected.shape, dtype=np.int16)
    chunks: list[dict[str, object]] = []
    for path in sorted(args.reference.glob("chunk-*.npz"), key=lambda p: int(p.stem[6:])):
        start = int(path.stem[6:])
        with np.load(path, allow_pickle=False) as fixture:
            prepared = preprocess(fixture["input"], fixture["affine"], spec)
            preprocessing_error = float(np.max(np.abs(prepared.data - fixture["prepared"])))
            prediction = network(mx.array(fixture["network_input"]))
            mx.eval(prediction)
            network_error = _error(np.asarray(prediction), fixture["network_logits"])
            padded = sliding_window_inference(
                prepared.data[None], (256, 256, 1), spec.spatial_window_batch_size,
                lambda window: network(window[..., 0])[..., None], overlap=0.9,
                mode="gaussian", sigma_scale=0.125, padding_mode="constant", cval=0.0,
                sw_device="gpu", device="cpu", predictor_input="mlx",
            )
            inverted = invert_logits(np.asarray(padded)[0], prepared.geometry)
            restored_error = _error(inverted, fixture["inverted_logits"])
            labels = discretize(inverted, spec)
            end = start + labels.shape[2]
            full[..., start:end] = labels
            chunks.append({
                "slices": [start, end], "preprocessing_max_absolute_error": preprocessing_error,
                "network": network_error, "restored_logits": restored_error,
            })
            print(json.dumps(chunks[-1]), flush=True)
    actual = keep_largest_components(full)
    mismatch = int(np.count_nonzero(actual != expected))
    dice: dict[str, float] = {}
    for label in np.union1d(np.unique(actual), np.unique(expected)):
        if label != 0:
            left, right = actual == label, expected == label
            dice[str(int(label))] = float(
                2 * np.count_nonzero(left & right) / (left.sum() + right.sum())
            )
    report = {
        "input_name": metadata["input_name"], "shape": list(actual.shape),
        "chunks": chunks, "mismatched_voxels": mismatch,
        "voxel_agreement": float(np.mean(actual == expected)), "dice": dice,
        "foreground_labels": len(dice), "mlx_device": str(mx.default_device()),
        "criteria": {"logits_atol": 5e-4, "logits_rtol": 1e-5, "mask_exact": True},
        "passed": mismatch == 0 and all(
            c["network"]["allclose"] and c["restored_logits"]["allclose"]
            for c in chunks
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "chunks"}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
