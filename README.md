# musclemap-mlx

Segment 113 muscle and bone structures in MRI or CT NIfTI images with
MuscleMap v1.4, running natively through MLX and Metal on Apple Silicon.
Weights download automatically on first use.

## Install and segment

Requires an Apple Silicon Mac, Metal and Python 3.13. With
[uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install musclemap-mlx

# Real MRI example from TotalSegmentator's official test data.
curl -L https://raw.githubusercontent.com/wasserth/TotalSegmentator/d89e47f73d24f5fba732a715aeb6fdc0ae34526b/tests/reference_files/example_mr_sm.nii.gz -o mri.nii.gz
musclemap-mlx segment --input mri.nii.gz --output-dir out
```

The command writes `out/mask.nii.gz` and prints JSON containing the output path,
label table, device, memory use and chunking settings. To use the corresponding
CT example, replace `example_mr_sm.nii.gz` with `example_ct_sm.nii.gz`.

### Python

```python
from musclemap_mlx.medmlx import run

result = run(image_path="mri.nii.gz", output_dir="out-python")
print(result["outputs"]["mask"])
```

### MedMLX

```sh
uv pip install 'medmlx[segment]>=0.9.1,<0.10'
medmlx describe musclemap
medmlx setup musclemap
medmlx run musclemap --arg image_path=mri.nii.gz output_dir=out-medmlx
```

Or use its Python interface:

```python
import medmlx

medmlx.setup_model("musclemap")
job = medmlx.run("musclemap", arguments={
    "image_path": "mri.nii.gz",
    "output_dir": "out-medmlx-python",
})
print(job["result"])
```

The MCP server exposes the same model through `list_models`, `describe_model`,
`model_readiness`, `prepare_model` and `run_model`; poll `get_job` for setup and
segmentation results. Start it with `medmlx serve --root .`.

## Inputs and outputs

Input is one finite 3D `.nii` or `.nii.gz` MRI or CT volume. Millimetre spatial
units are supported; an unspecified unit is interpreted as millimetres, matching
upstream. Other declared spatial units and 4D images are rejected.

The output is an `int16` mask with background 0 and the original 113 anatomical
IDs, including 1101 for left levator scapulae and 8222 for right plantaris. Its
shape, affine, qform, sform and spatial-unit metadata follow the input. The
largest 6-connected component of each label is retained after chunk assembly.
The JSON result lists every anatomical ID and name.

| Option | Meaning | Default |
| --- | --- | --- |
| `--input` | 3D NIfTI image | Required |
| `--output-dir` | Destination for `mask.nii.gz` | Required |
| `--model-dir` | Local folder containing `model.safetensors` | Automatic download/cache |
| `--chunk-size` | Source storage-axis slices per independent pass | Eight |
| `--overlap` | Gaussian window overlap, integer 0–99 percent | 90 |
| `--overwrite` | Replace an existing output | Off |

Each chunk is independently normalized. For integer source images, chunking
also reproduces upstream's source-header NIfTI round trip before normalization.
Changing chunk size can change the mask. A chunk size at least the source depth
uses one whole-volume pass. Inference uses MLX on Metal; there is no CPU or
PyTorch network fallback.

## Weights and offline use

The approximately 105 MB float32 bundle is hosted at
[MedMLX/musclemap-mlx](https://huggingface.co/MedMLX/musclemap-mlx/tree/a798b3073e3b2c449987c13aa193626aaacc3092),
pinned to revision `a798b3073e3b2c449987c13aa193626aaacc3092`. It includes the
publisher configuration, original label metadata and MIT notices.

```sh
# Pre-download to the shared cache; subsequent runs can work offline.
musclemap-mlx download

# Export a local bundle, then supply it explicitly.
musclemap-mlx download --model-dir musclemap-weights
musclemap-mlx segment --input mri.nii.gz --model-dir musclemap-weights --output-dir out-local
```

MedMLX and this package share the `MEDMLX_ASSET_CACHE` cache setting. Python
callers supply a local bundle with `weights_path="musclemap-weights"`.

## Qualification and license

The native implementation was compared with the pinned MuscleMap v1.4 reference
on real CT and MRI examples. Reproduction commands, acceptance criteria and
measurements are in
[docs/technical.md](https://github.com/MedMLX/musclemap-mlx/blob/dev/docs/technical.md).
These two compact examples establish implementation parity; anatomical accuracy
against manual segmentations and coverage of all 113 structures were not measured.

Code and publisher weights are MIT licensed. See
[UPSTREAM_LICENSE](https://github.com/MedMLX/musclemap-mlx/blob/dev/UPSTREAM_LICENSE)
and [THIRD_PARTY_NOTICES.md](https://github.com/MedMLX/musclemap-mlx/blob/dev/THIRD_PARTY_NOTICES.md).
The original checkpoint and configuration are published in
[Zenodo record 21929873](https://zenodo.org/records/21929873).
