"""Perfkit parity and benchmark entry points for the MuscleMap port.

These functions are maintainer tools, not package API. Point Perfkit at this
module with ``PYTHONPATH=scripts`` and the project interpreter. Recorded MONAI
1.3.2 outputs come from ``scripts/make_reference_fixtures.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from musclemap_mlx.config import default_spec
from musclemap_mlx.geometry import invert_logits, preprocess

if TYPE_CHECKING:
    from mlx.core import array

_NETWORK = None


def reference_dir() -> Path:
    value = os.environ.get("MEDMLX_TEST_REFERENCE")
    if not value:
        raise RuntimeError("Set MEDMLX_TEST_REFERENCE to the recorded MONAI fixture folder")
    return Path(value).expanduser().absolute()


def weights_dir() -> Path:
    value = os.environ.get("MEDMLX_TEST_WEIGHTS")
    if not value:
        raise RuntimeError("Set MEDMLX_TEST_WEIGHTS to a folder containing model.safetensors")
    return Path(value).expanduser().absolute()


def _network():
    global _NETWORK
    if _NETWORK is None:
        from musclemap_mlx.config import CONVERTED_NAME
        from musclemap_mlx.weights import load_network

        _NETWORK = load_network(weights_dir() / CONVERTED_NAME, default_spec())
    return _NETWORK


def preprocess_candidate(input_array: np.ndarray) -> np.ndarray:
    with np.load(reference_dir() / "preprocess.npz", allow_pickle=False) as fixture:
        affine = fixture["loaded_affine"]
    return preprocess(input_array, affine, default_spec()).data


def preprocess_reference(input_array: np.ndarray) -> np.ndarray:
    del input_array
    with np.load(reference_dir() / "preprocess.npz", allow_pickle=False) as fixture:
        return fixture["final"]


def inverse_candidate(padded_logits: np.ndarray) -> np.ndarray:
    with np.load(reference_dir() / "preprocess.npz", allow_pickle=False) as fixture:
        data = fixture["input"]
        affine = fixture["loaded_affine"]
    geometry = preprocess(data, affine, default_spec()).geometry
    return invert_logits(padded_logits, geometry)


def inverse_reference(padded_logits: np.ndarray) -> np.ndarray:
    del padded_logits
    with np.load(reference_dir() / "inverse.npz", allow_pickle=False) as fixture:
        return fixture["inverted_logits"]


def network_candidate(input_array: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    network = _network()
    result = network(mx.array(input_array))
    mx.eval(result)
    return np.asarray(result)


def network_reference(input_array: np.ndarray) -> np.ndarray:
    del input_array
    with np.load(reference_dir() / "network64.npz", allow_pickle=False) as fixture:
        return fixture["logits"]


def pipeline_candidate(input_array: np.ndarray) -> np.ndarray:
    from musclemap_mlx.pipeline import infer_volume

    with np.load(reference_dir() / "pipeline.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    labels, _, _ = infer_volume(
        input_array, affine, default_spec(), _network(), chunk_size=None, overlap_percent=90
    )
    return labels


def pipeline_reference(input_array: np.ndarray) -> np.ndarray:
    del input_array
    with np.load(reference_dir() / "pipeline.npz", allow_pickle=False) as fixture:
        return fixture["labels"]


def chunked_pipeline_candidate(input_array: np.ndarray) -> np.ndarray:
    from musclemap_mlx.pipeline import infer_volume

    with np.load(reference_dir() / "pipeline_chunked.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    labels, _, _ = infer_volume(
        input_array, affine, default_spec(), _network(), chunk_size=2, overlap_percent=90
    )
    return labels


def chunked_pipeline_reference(input_array: np.ndarray) -> np.ndarray:
    del input_array
    with np.load(reference_dir() / "pipeline_chunked.npz", allow_pickle=False) as fixture:
        return fixture["labels"]


def window_pipeline_candidate(input_array: np.ndarray) -> np.ndarray:
    from musclemap_mlx.pipeline import infer_volume

    with np.load(reference_dir() / "windows.npz", allow_pickle=False) as fixture:
        affine = fixture["input_affine"]
    labels, _, _ = infer_volume(
        input_array, affine, default_spec(), _network(), chunk_size=None, overlap_percent=90
    )
    return labels


def window_pipeline_reference(input_array: np.ndarray) -> np.ndarray:
    del input_array
    with np.load(reference_dir() / "windows.npz", allow_pickle=False) as fixture:
        return fixture["labels"]


def benchmark_network_256(input_array: array) -> array:
    return _network()(input_array)


def benchmark_pipeline_small(input_array: np.ndarray) -> np.ndarray:
    return pipeline_candidate(input_array)
