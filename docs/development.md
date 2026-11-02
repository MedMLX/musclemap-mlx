# musclemap-mlx development

Maintainer notes for the `dev` branch. Users need only the [README](../README.md).

## Checks

`make env` installs the locked Python 3.13 environment. `make qa` runs Ruff and
strict Pyright over `src` and `tests`, plus the fleet-wide `typing.Any` ban over
`src`, `tests` and `scripts`. The oracle-only scripts import Torch, MONAI and
partial nibabel stubs, so they are checked by Ruff and run by path rather than
included in the strict Pyright target.

`make test` runs the default offline suite: configuration and 113-label
inventory, asset verification, CLI and runner preflight rejection, and
CPU geometry self-consistency. It loads no model weights and runs no network.

`make build` builds the sdist and then the wheel from it, as a user would
resolve them.

The GPU suite is opt-in and needs the shared Perfkit GPU lock; the CPU
reference checks are opt-in but need no GPU:

```sh
make convert
make fixtures
MEDMLX_TEST_WEIGHTS=build/msk-porting/weights \
MEDMLX_TEST_REFERENCE=build/msk-porting/fixtures \
perfkit gpu lock run -- make metal
MEDMLX_TEST_REFERENCE=build/msk-porting/fixtures make oracle
```

`make metal` runs the native network and the full pipeline against recorded
MONAI 1.3.2 outputs. The recorded cases are a 64x64 network forward, a small
whole-volume pipeline, the same volume with two-slice chunking, and a larger
320x220x2 volume that exercises multiple sliding windows. Pipeline labels are
compared exactly; the 64x64 test checks output shape, dtype and finiteness. The
separate Perfkit parity receipt compares the 64x64 logits with the
source-derived float32 contract `atol=5e-4, rtol=1e-5`. No tolerance is relaxed
to obtain a pass.

`make oracle` runs the CPU-only recorded checks: the final preprocessed array and
affine, the inverse geometry on synthetic padded logits, and the argmax/remap
step. They run without Metal and without the network.

`scripts/parity_checks.py` provides the recorded-reference and native functions
used by the Perfkit commands. Perfkit parity receipts are written under
`build/msk-porting/parity/` with `perfkit parity run`; timings and peak memory
come from `perfkit benchmark call`. Observed Metal dispatch comes from an MLX
`mx.metal.start_capture` `.gputrace` analyzed with
`perfkit metal kernels`. These artifacts stay ignored and are not release
evidence by themselves.

## Reference fixtures

`make oracle-env` creates `environment/oracle/.venv` with the upstream pinned
versions: Python 3.11, Torch 2.4.1, MONAI 1.3.2, NumPy 1.24.4, nibabel 5.2.1 and
SciPy 1.11.4. Torch 2.4.1 has no CPython 3.13 wheel, which is why this oracle
environment is separate from the package environment. The package runtime and
the default tests never import Torch or MONAI.

`make fixtures` runs `scripts/make_reference_fixtures.py` in that environment
with `PYTHONPATH=src`. It writes synthetic-only recordings to
`build/msk-porting/fixtures/`:

| File | Content |
| :--- | :--- |
| `metadata.json` | Upstream revision, checkpoint/config SHA-256, framework versions, synthetic case definitions |
| `preprocess.npz` | Input, reader affine and every MONAI preprocessing stage |
| `inverse.npz` | Seeded 4-channel padded logits and MONAI `Invertd` output |
| `pipeline.npz` | Recorded network logits, pre-filter labels and filtered labels |
| `pipeline_chunked.npz` | Labels from the upstream two-slice chunking path |
| `windows.npz` | Labels from a 320x220x2 volume with multiple 256x256 windows |
| `network64.npz` | Seeded 64x64 network input and native-reference logits |

The fixtures are not committed: the repository policy keeps numerical fixtures,
weights, synthetic images and run artifacts ignored. Point the opt-in variables
at the generated folder. The synthetic phantom and slab are deterministic and
contain no patient data.

`make convert` runs `scripts/convert_checkpoint.py` against the verified
publisher checkpoint and writes
`build/msk-porting/weights/model.safetensors`, `config.json`, `LICENSE` and
`conversion.json`. The record contains the source checkpoint and configuration
hashes, the converter version and the output SHA-256.

## Upstream

- Code: [MuscleMap/MuscleMap](https://github.com/MuscleMap/MuscleMap) at
  `35a2138a4a3a63be713bae5584632ffc6433b960`, read-only in `reference/`.
  Upstream license: MIT, retained in [UPSTREAM_LICENSE](../UPSTREAM_LICENSE).
- Assets: Zenodo record 21929873, contrast-agnostic wholebody v1.4. The
  checkpoint is 104,884,148 bytes with MD5
  `910b722aeb641c380404c99ec6d1af97` and SHA-256
  `45dfa2843d2e0b1fd842152d6a79bfd4bcb90899c076ceb6346c59da9a79a16c`.
  The configuration is 10,555 bytes with SHA-256
  `82c74f854ab74d8770e6e2b9a240bd23ccc646a7b71b13e6cfd8f9919d2acb26`.
  Both are MIT. The publisher license is copied beside the converted weights.
- Reference framework: MONAI 1.3.2 with Torch 2.4.1 in the oracle environment.
  Conversion deserializes the same state dictionary with `torch.load(...,
  weights_only=True)` in the package environment, but never builds a Torch
  graph and never runs Torch at inference time.

Architecture and inference contract:

- 2D MONAI `UNet`, one input channel, 114 output channels, channels
  `(64, 128, 256, 512, 1024)`, strides `(2, 2, 2, 2)`, two residual units per
  encoder stage, instance normalization with `eps=1e-5` and no affine
  parameters, LeakyReLU slope 0.01, 3x3 convolutions with bias, 3x3 stride-2
  transposed convolutions with `output_padding=1`, and a conv-only final top
  residual unit.
- Preprocessing: `LoadImaged` (float32, slope/intercept applied),
  `EnsureChannelFirstd`, `Orientationd("RAS")`, `Spacingd([1, 1, -1],
  bilinear)`, `NormalizeIntensityd(nonzero=True)`, `CropForegroundd(margin=20,
  positive foreground)`, `SpatialPadd((256, 256, 1), method="end")`.
- Inference: `SliceInferer(roi_size=(256, 256), sw_batch_size=1, spatial_dim=2,
  mode="gaussian", overlap=0.9)`. The native port uses the shared
  `medmlx_core.sliding_window_inference` with a `(256, 256, 1)` ROI and the
  same Gaussian weights.
- Postprocessing: `Invertd(nearest_interp=False)` reverses pad, crop, spacing
  and orientation with bilinear resampling; `AsDiscreted(argmax=True,
  keepdim=True)` then squeeze, label remap to the original IDs, and
  `connected_chunks(connectivity=1)` (largest 6-connected component per label).
- Chunking follows `mm_util.run_inference`: the original array is sliced along
  its last axis, each chunk is independently preprocessed and inferred, labels
  are written back, and the largest-component filter runs after assembly.
  `chunk_size=None` uses eight slices in this port; an explicit value at least
  the volume depth runs one pass.

Kept on purpose:

- `pix_dim` component `-1` means "keep the source slice spacing" through
  MONAI's `fall_back_tuple`, not a negative-direction flip.
- Crop inversion uses MONAI's asymmetric border padding, restored from the
  recorded crop start and end, not a symmetric placement.
- The padding added to reach 256x256 is zero and is cropped away by the inverse
  before argmax.
- The final logits are not softmaxed; argmax runs on the inverted float32
  logits.
- Chunk boundaries can change the network input because each chunk is
  independently normalized and cropped. This is upstream behavior, not a port
  shortcut.
- A volume with no positive voxels after nonzero normalization is rejected with
  a model execution error instead of building an empty crop. NIfTI spatial units
  other than millimetres or unknown are rejected; the publisher pipeline assumes
  millimetres.

Dispatch and execution:

- Every network stage runs in the MLX Metal graph. Preprocessing, NIfTI IO,
  Gaussian window blending, inverse resampling and largest-component filtering
  are CPU host work. The network has no CPU or Torch fallback.
- The model transposes trainable convolution weights once, dispatches MLX
  `conv2d` and `conv_transpose2d` kernels on the GPU stream, and evaluates on
  Metal. Instance normalization and LeakyReLU are compositions of MLX GPU
  kernels.
- An `mx.metal.start_capture` capture of one 256x256 forward is analyzed with
  `perfkit metal kernels`. The capture and its JSON stay under
  `build/msk-porting/metal/`. Standard MLX Metal convolution, GEMM,
  elementwise and copy pipelines are observed; there is no custom Metal source
  in this port and no kernel-time claim from the capture.
