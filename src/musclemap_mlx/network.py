"""Native MLX implementation of the MuscleMap v1.4 2D MONAI UNet."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, cast

from medmlx_core.errors import MissingDependencyError
from medmlx_core.typing import MlxRuntime

from musclemap_mlx.config import (
    INSTANCE_NORM_EPS,
    LEAKY_SLOPE,
    ModelSpec,
    ParameterInventory,
)

if TYPE_CHECKING:
    from mlx.core import array


class _StreamContext(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class MuscleMapRuntime(MlxRuntime, Protocol):
    """The MLX operations this graph needs beyond the shared core protocol."""

    def contiguous(self, a: array, /, *, stream: object = None) -> array: ...

    def compile(self, fun: Callable[[array], array]) -> Callable[[array], array]: ...

    def conv_transpose2d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        *,
        stream: object = None,
    ) -> array: ...

    def load(
        self, file: str, /, format: str | None = None
    ) -> dict[str, array] | array: ...

    def save_safetensors(
        self,
        file: str,
        arrays: dict[str, array],
        metadata: dict[str, str] | None = None,
    ) -> None: ...

    def stream(self, s: object) -> _StreamContext: ...


class MuscleMapUNet:
    """MONAI ``UNet`` with instance norm, LeakyReLU, residuals and 114 outputs.

    Activations are channel-last inside ``_forward``; the public boundary accepts
    and returns channel-first ``(N, 1, H, W)`` / ``(N, 114, H, W)`` float32 arrays,
    matching ``monai.networks.nets.UNet``. Weights use the publisher's Torch names
    and ``(out, in, kH, kW)`` convolution layout. They are transposed once in
    :meth:`_prepare`; the published checkpoint has no instance-norm parameters.

    The graph runs on the MLX Metal device selected by ``mx.stream(mx.gpu)``.
    It refuses a non-Metal default device rather than falling back to CPU.
    """

    def __init__(
        self,
        weights: Mapping[str, array],
        spec: ModelSpec,
        inventory: ParameterInventory,
        *,
        mx: MuscleMapRuntime,
    ) -> None:
        self.w = dict(weights)
        self.spec = spec
        self.inventory = inventory
        self.mx = mx
        self._filters: dict[str, array] = {}
        self._prepared = False
        self._compiled_forward = self.mx.compile(self._forward)

    def _prepare(self) -> None:
        if self._prepared:
            return
        filters: dict[str, array] = {}
        for name, value in self.w.items():
            if not name.endswith(".weight") or value.ndim != 4:
                continue
            axes = (1, 2, 3, 0) if name in self.inventory.deconv_weights else (0, 2, 3, 1)
            filters[name] = self.mx.contiguous(self.mx.transpose(value, axes))
        self._filters = filters
        self.mx.eval(list(filters.values()))
        self._prepared = True

    def _require_metal(self) -> None:
        if not self.mx.metal.is_available() or self.mx.default_device().type != self.mx.gpu:
            raise MissingDependencyError(
                "MuscleMap inference requires an available Metal GPU as the MLX default device",
                hint="Run on an Apple Silicon Mac through medmlx_core.import_mlx().",
            )

    def _conv(self, x: array, name: str, *, stride: int = 1, padding: int = 1) -> array:
        y = self.mx.conv2d(x, self._filters[f"{name}.weight"], stride=stride, padding=padding)
        return y + self.w[f"{name}.bias"]

    def _deconv(self, x: array, name: str) -> array:
        y = self.mx.conv_transpose2d(
            x,
            self._filters[f"{name}.weight"],
            stride=2,
            padding=1,
            output_padding=1,
        )
        return y + self.w[f"{name}.bias"]

    def _norm(self, x: array) -> array:
        axes = (1, 2)
        mean = self.mx.mean(x, axis=axes, keepdims=True)
        variance = self.mx.var(x, axis=axes, keepdims=True, ddof=0)
        return (x - mean) * self.mx.rsqrt(variance + INSTANCE_NORM_EPS)

    def _activate(self, x: array) -> array:
        return self.mx.maximum(x, x * LEAKY_SLOPE)

    def _convolution(
        self,
        x: array,
        name: str,
        *,
        stride: int,
        transposed: bool = False,
        conv_only: bool = False,
    ) -> array:
        y = (
            self._deconv(x, f"{name}.conv")
            if transposed
            else self._conv(x, f"{name}.conv", stride=stride)
        )
        return y if conv_only else self._activate(self._norm(y))

    def _residual_unit(
        self,
        x: array,
        name: str,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        subunits: int,
        last_conv_only: bool = False,
    ) -> array:
        y = x
        for unit in range(subunits):
            y = self._convolution(
                y,
                f"{name}.conv.unit{unit}",
                stride=stride if unit == 0 else 1,
                conv_only=last_conv_only and unit == subunits - 1,
            )
        if stride == 1 and in_channels == out_channels:
            residual = x
        else:
            kernel = 1 if stride == 1 else 3
            residual = self._conv(
                x,
                f"{name}.residual",
                stride=stride,
                padding=1 if kernel == 3 else 0,
            )
        return y + residual

    def _up_layer(
        self, x: array, name: str, in_channels: int, out_channels: int, *, is_top: bool
    ) -> array:
        y = self._convolution(x, f"{name}.0", stride=2, transposed=True)
        return self._residual_unit(
            y,
            f"{name}.1",
            out_channels,
            out_channels,
            stride=1,
            subunits=1,
            last_conv_only=is_top,
        )

    def _block(
        self,
        x: array,
        down_name: str,
        up_name: str,
        sub_name: str,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        *,
        is_top: bool,
    ) -> array:
        channel_values = list(channels)
        stride_values = list(strides)
        here = channel_values[0]
        stride = stride_values[0]
        down = self._residual_unit(
            x,
            down_name,
            in_channels,
            here,
            stride=stride,
            subunits=self.spec.num_res_units,
        )
        if len(channel_values) > 2:
            sub = self._block(
                down,
                f"{sub_name}.submodule.0",
                f"{sub_name}.submodule.2",
                f"{sub_name}.submodule.1",
                here,
                here,
                tuple(channel_values[1:]),
                tuple(stride_values[1:]),
                is_top=False,
            )
            up_channels = here * 2
        else:
            next_channels = channel_values[1]
            sub = self._residual_unit(
                down,
                f"{sub_name}.submodule",
                here,
                next_channels,
                stride=1,
                subunits=self.spec.num_res_units,
            )
            up_channels = here + next_channels
        joined = self.mx.concatenate([down, sub], axis=-1)
        return self._up_layer(joined, up_name, up_channels, out_channels, is_top=is_top)

    def __call__(self, x: array) -> array:
        """Return ``(N, 114, H, W)`` logits for a ``(N, 1, H, W)`` float32 batch.

        Raises:
            MissingDependencyError: Metal is unavailable or no longer the default device.
            ValueError: The input rank, channels, dtype or spatial divisibility is wrong.
        """
        self._require_metal()
        if (
            x.ndim != 4
            or int(x.shape[1]) != 1
            or int(x.shape[2]) < 16
            or int(x.shape[3]) < 16
            or int(x.shape[2]) % 16 != 0
            or int(x.shape[3]) % 16 != 0
        ):
            raise ValueError(
                f"Expected Nx1xHxW float32 input with H,W multiples of 16; got {x.shape}"
            )
        if x.dtype != self.mx.float32:
            raise ValueError("MuscleMap native UNet requires float32 input")
        self._prepare()
        with self.mx.stream(self.mx.gpu):
            return self._compiled_forward(x)

    def _forward(self, x: array) -> array:
        """Build the reusable graph after weight preparation and input admission."""
        channels_last = self.mx.transpose(x, (0, 2, 3, 1))
        output = self._block(
            channels_last,
            "model.0",
            "model.2",
            "model.1",
            1,
            self.spec.out_channels,
            self.spec.channels,
            self.spec.strides,
            is_top=True,
        )
        # Host Gaussian blending traverses each channel spatially. Materialize
        # that layout on Metal instead of exporting a channel-last strided view.
        return self.mx.contiguous(self.mx.transpose(output, (0, 3, 1, 2)))


def as_musclemap_runtime(mx: MlxRuntime) -> MuscleMapRuntime:
    """Narrow the shared MLX runtime to the operations this graph uses."""
    return cast(MuscleMapRuntime, mx)
