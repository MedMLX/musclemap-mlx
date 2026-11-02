"""Publisher configuration, label order and checkpoint inventory contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musclemap_mlx.config import (
    EXPECTED_CHANNELS,
    EXPECTED_OUTPUTS,
    EXPECTED_PIXDIM,
    default_spec,
    expected_parameters,
    load_model_config,
)


def test_default_spec_preserves_published_ids() -> None:
    spec = default_spec()
    assert spec.out_channels == EXPECTED_OUTPUTS
    assert spec.channels == EXPECTED_CHANNELS
    assert spec.pix_dim == EXPECTED_PIXDIM
    assert len(spec.labels) == 113
    assert len(spec.class_labels) == 114
    assert spec.class_labels[0] == 0
    assert spec.class_labels[1] == 1101
    assert spec.class_labels[-1] == 8222
    assert len(set(spec.class_labels)) == 114


def test_class_labels_are_sorted_and_unique() -> None:
    labels = default_spec().class_labels[1:]
    assert labels == tuple(sorted(labels))
    assert len(labels) == len(set(labels))


def test_inventory_matches_the_published_unet() -> None:
    spec = default_spec()
    inventory = expected_parameters(spec)
    assert len(inventory.shapes) == 46
    assert inventory.shapes["model.0.conv.unit0.conv.weight"] == (64, 1, 3, 3)
    assert inventory.shapes["model.0.residual.weight"] == (64, 1, 3, 3)
    bottom = "model.1.submodule.1.submodule.1.submodule.1.submodule.residual.weight"
    assert inventory.shapes[bottom] == (1024, 512, 1, 1)
    assert inventory.shapes["model.2.0.conv.weight"] == (128, 114, 3, 3)
    assert inventory.deconv_weights == frozenset(
        {
            "model.1.submodule.1.submodule.1.submodule.2.0.conv.weight",
            "model.1.submodule.1.submodule.2.0.conv.weight",
            "model.1.submodule.2.0.conv.weight",
            "model.2.0.conv.weight",
        }
    )


def test_load_model_config_round_trips_default(tmp_path: Path) -> None:
    spec = default_spec()
    payload = {
        "parameters": {
            "roi_size": [256, 256],
            "spatial_window_batch_size": 1,
            "pix_dim": [1, 1, -1],
        },
        "model": {
            "version": 1.4,
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 114,
            "channels": [64, 128, 256, 512, 1024],
            "act": "LeakyReLU",
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "instance",
        },
        "labels": [
            {
                "value": label.value,
                "region": label.region,
                "anatomy": label.anatomy,
                "side": label.side,
            }
            for label in spec.labels
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_model_config(path)
    assert loaded.class_labels == spec.class_labels
    assert loaded.channels == spec.channels


def test_load_model_config_rejects_renumbered_labels(tmp_path: Path) -> None:
    spec = default_spec()
    payload = {
        "parameters": {
            "roi_size": [256, 256],
            "spatial_window_batch_size": 1,
            "pix_dim": [1, 1, -1],
        },
        "model": {
            "version": 1.4,
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 114,
            "channels": [64, 128, 256, 512, 1024],
            "act": "LeakyReLU",
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "instance",
        },
        "labels": [
            {
                "value": label.value + 1,
                "region": label.region,
                "anatomy": label.anatomy,
                "side": label.side,
            }
            for label in spec.labels
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"anatomical table|ascending|unique"):
        load_model_config(path)


def test_class_to_label_mapping_is_int16() -> None:
    mapping = np.asarray(default_spec().class_labels, dtype=np.int16)
    assert mapping.dtype == np.int16
    assert mapping[0] == 0
    assert mapping[113] == 8222
