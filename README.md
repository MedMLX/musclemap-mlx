# musclemap-mlx

Development intake for a future native MLX port of MuscleMap v1.4 whole-body
muscle segmentation. This package currently verifies local assets only. Native
inference, checkpoint conversion and numerical comparisons are not implemented.

[Upstream](https://github.com/MuscleMap/MuscleMap) ·
[Weights and original configuration](https://zenodo.org/records/21929873) ·
[MedMLX](https://medmlx.org)

| | |
| :--- | :--- |
| Upstream model | Contrast-agnostic wholebody v1.4, 2D MONAI UNet |
| Label configuration | 113 original anatomical labels plus background, 114 output channels |
| Assets | 104,884,148-byte PyTorch checkpoint and original JSON, MIT |
| Current output | JSON verification/status only; no segmentation output |

## Use with MedMLX

This development package is not registered with the MedMLX model catalog and
has no inference runner. Installing it does not enable `medmlx run`.

## Install

From a local checkout, with uv and Python 3.13:

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

## Run

```sh
musclemap-mlx status
musclemap-mlx inspect-assets --model-dir weights
```

The asset directory must contain `contrast_agnostic_wholebody_model.pth`,
`contrast_agnostic_wholebody_model.json` and the publisher `LICENSE` from
Zenodo record 21929873. Inspection verifies publisher file sizes and MD5 values
and prints the original configuration, including label values, without loading
the checkpoint. It makes no network calls and runs no model.

The configuration uses 2D input, channels 64/128/256/512/1024, instance
normalization, two residual units and four stride-2 stages. Original anatomical
IDs must be preserved rather than replaced by channel indices.

## License

MIT. Upstream source terms are preserved in [UPSTREAM_LICENSE](UPSTREAM_LICENSE),
with attribution and publisher asset terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model assets remain under their
publisher's MIT terms. Research software; no clinical performance is established.
