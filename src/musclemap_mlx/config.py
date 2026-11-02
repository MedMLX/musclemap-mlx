"""Publisher configuration, label order and the pinned MONAI UNet inventory."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from musclemap_mlx.labels import LABELS

CONFIG_NAME = "contrast_agnostic_wholebody_model.json"
CHECKPOINT_NAME = "contrast_agnostic_wholebody_model.pth"
CONVERTED_NAME = "model.safetensors"
MODEL_ID = "musclemap"
UPSTREAM_REVISION = "35a2138a4a3a63be713bae5584632ffc6433b960"
ZENODO_RECORD = "https://zenodo.org/records/21929873"

# The publisher's inference configuration and checkpoint were trained with these
# fixed choices. The converter and network reject a config that drifts from them.
EXPECTED_CHANNELS = (64, 128, 256, 512, 1024)
EXPECTED_STRIDES = (2, 2, 2, 2)
EXPECTED_PIXDIM = (1.0, 1.0, -1.0)
EXPECTED_ROI = (256, 256)
EXPECTED_OUTPUTS = 114
EXPECTED_LABELS = 113
INSTANCE_NORM_EPS = 1e-5
LEAKY_SLOPE = 0.01
CROP_MARGIN = 20
PAD_SIZE = (256, 256, 1)


@dataclass(frozen=True, slots=True)
class Label:
    """One publisher anatomical label, keeping its original integer ID."""

    value: int
    region: str
    anatomy: str
    side: str

    @property
    def name(self) -> str:
        """Return a readable name without replacing the anatomical ID."""
        return self.anatomy if self.side == "no side" else f"{self.side} {self.anatomy}"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Validated publisher model and inference settings."""

    version: str
    spatial_dims: Literal[2]
    in_channels: Literal[1]
    out_channels: Literal[114]
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    num_res_units: int
    act: Literal["LeakyReLU"]
    norm: Literal["instance"]
    roi_size: tuple[int, int]
    pix_dim: tuple[float, float, float]
    spatial_window_batch_size: int
    labels: tuple[Label, ...]

    @property
    def class_labels(self) -> tuple[int, ...]:
        """Return background followed by the sorted original anatomical IDs."""
        return (0, *(label.value for label in self.labels))

    @property
    def label_details(self) -> tuple[dict[str, str | int], ...]:
        """Return the MedMLX ``details.labels`` entries in network channel order."""
        return (
            {"value": 0, "name": "background"},
            *({"value": label.value, "name": label.name} for label in self.labels),
        )


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    """Expected checkpoint names, shapes and the transposed-convolution subset."""

    shapes: dict[str, tuple[int, ...]]
    deconv_weights: frozenset[str]


def default_labels() -> tuple[Label, ...]:
    """Return the publisher's 113 anatomical labels in ascending ID order."""
    return tuple(
        Label(value=value, region=region, anatomy=anatomy, side=side)
        for value, region, anatomy, side in LABELS
    )


def default_spec() -> ModelSpec:
    """Return the pinned v1.4 model contract without reading local assets."""
    return ModelSpec(
        version="1.4",
        spatial_dims=2,
        in_channels=1,
        out_channels=114,
        channels=EXPECTED_CHANNELS,
        strides=EXPECTED_STRIDES,
        num_res_units=2,
        act="LeakyReLU",
        norm="instance",
        roi_size=EXPECTED_ROI,
        pix_dim=EXPECTED_PIXDIM,
        spatial_window_batch_size=1,
        labels=default_labels(),
    )


def load_model_config(path: Path) -> ModelSpec:
    """Load the publisher JSON and reject anything outside the v1.4 contract.

    Raises:
        OSError: The file cannot be read.
        ValueError: The JSON is malformed, a label ID is duplicated, or an
            architecture or inference setting differs from the published v1.4
            configuration this native port implements.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(payload, "configuration")
    parameters = _mapping(root.get("parameters"), "parameters")
    model = _mapping(root.get("model"), "model")
    raw_labels = _sequence(root.get("labels"), "labels")

    labels = tuple(_parse_label(item, index) for index, item in enumerate(raw_labels))
    if len(labels) != EXPECTED_LABELS:
        raise ValueError(f"expected {EXPECTED_LABELS} anatomical labels, got {len(labels)}")
    values = {label.value for label in labels}
    if len(values) != len(labels) or 0 in values or min(values) < 1:
        raise ValueError("anatomical label IDs must be unique positive integers")
    if tuple(sorted(values)) != tuple(label.value for label in labels):
        raise ValueError("publisher labels are not in ascending anatomical ID order")
    if tuple((label.value, label.region, label.anatomy, label.side) for label in labels) != LABELS:
        raise ValueError("publisher labels differ from the pinned v1.4 anatomical table")

    spatial_dims = _integer(model.get("spatial_dims"), "spatial_dims")
    in_channels = _integer(model.get("in_channels"), "in_channels")
    out_channels = _integer(model.get("out_channels"), "out_channels")
    num_res_units = _integer(model.get("num_res_units"), "num_res_units")
    channels = _integers(model.get("channels"), "channels")
    strides = _integers(model.get("strides"), "strides")
    roi_size = _integers(parameters.get("roi_size"), "roi_size")
    pix_dim = _floats(parameters.get("pix_dim"), "pix_dim")
    window_batch = _integer(
        parameters.get("spatial_window_batch_size"), "spatial_window_batch_size"
    )
    version = f"{_number(model.get('version'), 'version'):g}"
    act = _string(model.get("act"), "act")
    norm = _string(model.get("norm"), "norm")

    if spatial_dims != 2:
        raise ValueError("MuscleMap v1.4 native MLX port requires spatial_dims=2")
    if in_channels != 1 or out_channels != EXPECTED_OUTPUTS:
        raise ValueError("MuscleMap v1.4 requires one input and 114 output channels")
    if channels != EXPECTED_CHANNELS or strides != EXPECTED_STRIDES:
        raise ValueError("MuscleMap v1.4 channels/strides differ from the pinned UNet")
    if num_res_units != 2:
        raise ValueError("MuscleMap v1.4 requires two residual units per encoder stage")
    if act != "LeakyReLU" or norm != "instance":
        raise ValueError("MuscleMap v1.4 requires LeakyReLU and instance normalization")
    if roi_size != EXPECTED_ROI:
        raise ValueError(f"MuscleMap v1.4 requires roi_size={EXPECTED_ROI}")
    if pix_dim != EXPECTED_PIXDIM:
        raise ValueError(f"MuscleMap v1.4 requires pix_dim={EXPECTED_PIXDIM}")
    if window_batch < 1:
        raise ValueError("spatial_window_batch_size must be positive")
    if tuple(_version_parts(version)) != (1, 4):
        raise ValueError("MuscleMap native MLX port implements configuration version 1.4")

    return ModelSpec(
        version=version,
        spatial_dims=2,
        in_channels=1,
        out_channels=114,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act="LeakyReLU",
        norm="instance",
        roi_size=(roi_size[0], roi_size[1]),
        pix_dim=(pix_dim[0], pix_dim[1], pix_dim[2]),
        spatial_window_batch_size=window_batch,
        labels=labels,
    )


def expected_parameters(spec: ModelSpec) -> ParameterInventory:
    """Return every checkpoint tensor and shape for the published UNet.

    The names and nesting follow ``monai.networks.nets.UNet``: top-level
    ``model.0``/``model.1``/``model.2``, nested ``submodule`` sequences, two
    residual convolutions per encoder stage and one transposed convolution per
    decoder stage.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    deconv_weights: set[str] = set()

    def add(name: str, shape: tuple[int, ...]) -> None:
        if name in shapes:
            raise ValueError(f"duplicate expected parameter: {name}")
        shapes[name] = shape

    def residual_unit(
        prefix: str, in_channels: int, out_channels: int, stride: int, subunits: int
    ) -> None:
        for unit in range(subunits):
            input_channels = in_channels if unit == 0 else out_channels
            add(f"{prefix}.conv.unit{unit}.conv.weight", (out_channels, input_channels, 3, 3))
            add(f"{prefix}.conv.unit{unit}.conv.bias", (out_channels,))
        if stride != 1 or in_channels != out_channels:
            kernel = 1 if stride == 1 else 3
            add(f"{prefix}.residual.weight", (out_channels, in_channels, kernel, kernel))
            add(f"{prefix}.residual.bias", (out_channels,))

    def up_layer(prefix: str, in_channels: int, out_channels: int, is_top: bool) -> None:
        deconv = f"{prefix}.0.conv.weight"
        deconv_weights.add(deconv)
        add(deconv, (in_channels, out_channels, 3, 3))
        add(f"{prefix}.0.conv.bias", (out_channels,))
        residual_unit(f"{prefix}.1", out_channels, out_channels, 1, 1)
        # The final top residual unit is conv-only; instance norm has no parameters.
        _ = is_top

    def block(
        down_prefix: str,
        up_prefix: str,
        sub_prefix: str,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        is_top: bool,
    ) -> None:
        channel_values = list(channels)
        stride_values = list(strides)
        channels_here = channel_values[0]
        stride = stride_values[0]
        residual_unit(down_prefix, in_channels, channels_here, stride, spec.num_res_units)
        if len(channel_values) > 2:
            block(
                f"{sub_prefix}.submodule.0",
                f"{sub_prefix}.submodule.2",
                f"{sub_prefix}.submodule.1",
                channels_here,
                channels_here,
                tuple(channel_values[1:]),
                tuple(stride_values[1:]),
                False,
            )
            up_channels = channels_here * 2
        else:
            next_channels = channel_values[1]
            residual_unit(
                f"{sub_prefix}.submodule",
                channels_here,
                next_channels,
                1,
                spec.num_res_units,
            )
            up_channels = channels_here + next_channels
        up_layer(up_prefix, up_channels, out_channels, is_top)

    block(
        "model.0",
        "model.2",
        "model.1",
        spec.in_channels,
        spec.out_channels,
        spec.channels,
        spec.strides,
        True,
    )
    return ParameterInventory(shapes=shapes, deconv_weights=frozenset(deconv_weights))


def _parse_label(value: object, index: int) -> Label:
    entry = _mapping(value, f"labels[{index}]")
    return Label(
        value=_integer(entry.get("value"), f"labels[{index}].value"),
        region=_string(entry.get("region"), f"labels[{index}].region"),
        anatomy=_string(entry.get("anatomy"), f"labels[{index}].anatomy"),
        side=_string(entry.get("side"), f"labels[{index}].side"),
    )


def _mapping(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object")
    return cast(dict[str, object], value)


def _sequence(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a JSON array")
    return cast(list[object], value)


def _integer(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    return value


def _string(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number")
    return float(value)


def _integers(value: object, what: str) -> tuple[int, ...]:
    values = _sequence(value, what)
    return tuple(_integer(item, f"{what}[]") for item in values)


def _floats(value: object, what: str) -> tuple[float, ...]:
    values = _sequence(value, what)
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{what} must contain numbers")
        result.append(float(item))
    return tuple(result)


def _version_parts(version: str) -> Sequence[int]:
    parts: list[int] = []
    for component in version.split("."):
        if not component.isdigit():
            raise ValueError(f"unsupported model version: {version}")
        parts.append(int(component))
    return parts
