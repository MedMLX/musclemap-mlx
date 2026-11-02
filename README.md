# musclemap-mlx

Segment 113 whole-body muscle and bone structures from MRI or CT with the
MuscleMap v1.4 contrast-agnostic 2D MONAI UNet, running natively on Apple
Silicon through MLX and Metal. Converted weights are not hosted yet; this
checkout can convert the publisher checkpoint locally.

[Publisher weights and configuration](https://zenodo.org/records/21929873) ·
[Upstream source](https://github.com/MuscleMap/MuscleMap) ·
[MedMLX](https://medmlx.org)

| | |
| :--- | :--- |
| Upstream model | MuscleMap v1.4 contrast-agnostic wholebody, 2D MONAI UNet |
| Model ID | `musclemap` |
| Input | One 3D NIfTI volume in millimetres, MRI or CT |
| Output | `mask.nii.gz`, int16, 113 original anatomical IDs plus background |
| Label configuration | 114 output channels; original IDs preserved, not renumbered |
| Native runtime | MLX 0.32.4 on Metal, float32 |
| Hardware | Apple Silicon Mac with Metal; no CPU network fallback |
| Weights | 104.9 MB publisher PyTorch checkpoint, MIT; local conversion required |
| Hosted converted weights | None yet |
| Python | 3.13 |
| Intended use | Research and education. Not a medical device; outputs require human review. |

## Use with MedMLX

The package declares model ID `musclemap` for segmentation. No converted
weights are hosted, so pass a local `weights_path`; `medmlx setup` is not
available for this model until a converted repository is published.

```json
{
  "image_path": "wholebody.nii.gz",
  "weights_path": "converted-musclemap",
  "output_dir": "out",
  "chunk_size": null,
  "overlap_percent": 90,
  "overwrite": false
}
```

`chunk_size: null` uses the conservative eight-slice default. The folder named
by `weights_path` contains `model.safetensors` and, optionally, `config.json`
and the publisher license.

## Install

This development package is not published on PyPI. Install [uv](https://docs.astral.sh/uv/)
first, then install from a checkout:

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install .
```

## Run

```sh
musclemap-mlx segment \
  --input wholebody.nii.gz \
  --model-dir converted-musclemap \
  --output-dir out
```

The command writes `out/mask.nii.gz` and prints one JSON object with the output
path, label table, MLX device, peak MLX memory, chunking facts and duration.

### Convert the publisher checkpoint

The development tree contains the maintainer conversion tool. It verifies the
publisher MD5 and SHA-256 values, loads the state dictionary with
`weights_only=True`, checks all 46 tensor names, shapes, dtypes and finite
values, writes deterministic safetensors, reloads it, and records the source
and output hashes. The runtime never imports PyTorch.

The converted folder is not redistributed from this repository. A wheel install
does not include the conversion tool, so use a development checkout until
converted weights are hosted. See the publisher license and terms before sharing
the converted folder.

### Command line

| Option | Meaning | Default |
| :--- | :--- | :--- |
| `--input` | 3D NIfTI image (`.nii` or `.nii.gz`) | Required |
| `--model-dir` | Folder containing `model.safetensors` | Local folder required |
| `--output-dir` | Destination folder | Required |
| `--chunk-size` | Axial slices per independent preprocessing pass | Automatic, eight slices |
| `--overlap` | Gaussian slice-window overlap percentage, 0-99 | 90 |
| `--overwrite` | Replace an existing `mask.nii.gz` | Off |
| `--version` | Print installed version | Off |

The `download` subcommand accepts `--repo`, a full 40-character `--revision`
and `--model-dir`, but no converted MuscleMap repository is published yet.

### Weights

| File | Size | Terms |
| :--- | :--- | :--- |
| `contrast_agnostic_wholebody_model.pth` | 104,884,148 bytes | MIT, publisher |
| `contrast_agnostic_wholebody_model.json` | 10,555 bytes | MIT, publisher |
| `model.safetensors` after conversion | 104,873,688 bytes | MIT, publisher weights |

The publisher checkpoint SHA-256 is
`45dfa2843d2e0b1fd842152d6a79bfd4bcb90899c076ceb6346c59da9a79a16c`
and the configuration SHA-256 is
`82c74f854ab74d8770e6e2b9a240bd23ccc646a7b71b13e6cfd8f9919d2acb26`.
The converter writes a `conversion.json` beside the safetensors with the exact
source and output hashes. Model assets remain under the publisher's MIT terms;
no weights are committed to this repository.

### Inputs

The model is contrast-agnostic and accepts MRI or CT. The pipeline follows the
publisher's transforms: load the NIfTI in float32, apply the header slope and
intercept, move channels first, reorient to RAS, resample the first two axes to
1 mm bilinearly while keeping the source slice spacing (`pix_dim [1, 1, -1]`),
normalize nonzero voxels, crop the positive foreground with a 20-voxel margin,
pad at the end to at least 256x256x1, and run 2D slices with 90% Gaussian window
overlap. Logits are resampled back to the reader grid before argmax, so the mask
keeps the input shape, affine and qform/sform codes.

### Outputs

`mask.nii.gz` is int16 with background 0 and the original anatomical IDs from
the publisher configuration, for example 1101 for the left levator scapulae
and 8222 for the right plantaris. The largest 6-connected component of each
label is kept, matching the upstream postprocessing. The runner never
interprets the segmentation; review it before any scientific or clinical use.

## License

MIT. Upstream source terms are preserved in
[UPSTREAM_LICENSE](https://github.com/MedMLX/musclemap-mlx/blob/dev/UPSTREAM_LICENSE),
with attribution and publisher asset terms in
[THIRD_PARTY_NOTICES.md](https://github.com/MedMLX/musclemap-mlx/blob/dev/THIRD_PARTY_NOTICES.md).
Model assets remain under their publisher's MIT terms. Research software; no
clinical performance is established.
