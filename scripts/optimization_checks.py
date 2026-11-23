"""Candidate functions for the MuscleMap optimization loop."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from medmlx_core.sliding_window import sliding_window_inference

from musclemap_mlx.config import CONVERTED_NAME, default_spec
from musclemap_mlx.geometry import discretize, invert_logits, keep_largest_components, preprocess
from musclemap_mlx.pipeline import infer_volume
from musclemap_mlx.weights import load_network

if TYPE_CHECKING:
    from mlx.core import array

_NETWORK = None
_GEOMETRY = None


def _reference() -> Path:
    return Path(os.environ.get("MEDMLX_TEST_REFERENCE", "build/msk-porting/fixtures"))


def _weights() -> Path:
    return Path(os.environ.get("MEDMLX_TEST_WEIGHTS", "build/msk-porting/weights"))


def network() -> object:
    global _NETWORK
    if _NETWORK is None:
        _NETWORK = load_network(_weights() / CONVERTED_NAME, default_spec())
    return _NETWORK


def geometry() -> object:
    global _GEOMETRY
    if _GEOMETRY is None:
        ys, xs = np.mgrid[0:256, 0:256]
        phantom = np.zeros((256, 256, 8), dtype=np.float32)
        phantom[48:208, 48:208, :] = (
            20.0 + 0.2 * ys[48:208, 48:208, None] + 0.1 * xs[48:208, 48:208, None]
        )
        _GEOMETRY = preprocess(phantom, np.eye(4, dtype=np.float64), default_spec()).geometry
    return _GEOMETRY


def network_64(input_array: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    result = network()(mx.array(input_array))
    mx.eval(result)
    return np.asarray(result)


def network_256(input_array: array) -> array:
    return network()(input_array)


def preprocess_large(input_array: np.ndarray) -> np.ndarray:
    with np.load(_reference() / "large_oblique.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    return preprocess(input_array, affine, default_spec()).data


def pipeline_small(input_array: np.ndarray) -> np.ndarray:
    with np.load(_reference() / "pipeline.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    labels, _, _ = infer_volume(input_array, affine, default_spec(), network(), chunk_size=None)
    return labels


def pipeline_large(input_array: np.ndarray) -> np.ndarray:
    with np.load(_reference() / "large_oblique.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    labels, _, _ = infer_volume(input_array, affine, default_spec(), network(), chunk_size=None)
    return labels


def inverse_stage(padded_logits: np.ndarray) -> np.ndarray:
    return invert_logits(padded_logits, geometry())


def inverse_labels(padded_logits: np.ndarray) -> np.ndarray:
    return discretize(invert_logits(padded_logits, geometry()), default_spec())


def component_stage(labels: np.ndarray) -> np.ndarray:
    return keep_largest_components(labels)


def blend_stage(input_array: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    def predictor(window: array) -> array:
        batch = int(window.shape[0])
        return mx.zeros((batch, 114, 256, 256, 1), dtype=mx.float32)

    result = sliding_window_inference(
        input_array[None, None],
        (256, 256, 1),
        1,
        predictor,
        overlap=0.9,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="constant",
        cval=0.0,
        sw_device="gpu",
        device="cpu",
        predictor_input="mlx",
    )
    return result[0]
