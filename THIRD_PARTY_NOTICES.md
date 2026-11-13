# Third-party notices

## MuscleMap

- Source: https://github.com/MuscleMap/MuscleMap
- Revision: `35a2138a4a3a63be713bae5584632ffc6433b960`
- Copyright: Copyright (c) 2023 MuscleMap
- License: MIT; original text retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
  The package [LICENSE](LICENSE) retains the same MIT terms.
- Adapted: native MLX implementation of the published 2D MONAI UNet and its
  inference preprocessing/postprocessing. The original anatomical label IDs,
  configuration semantics and largest-connected-component helper are preserved.
  Upstream source remains unchanged and read-only in the development-only
  `reference/` submodule.

## Dependencies, weights and data

Dependencies declared in [pyproject.toml](pyproject.toml) are installed
separately and keep their own licenses. MLX is the runtime backend. MONAI 1.3.2
and PyTorch 2.4.1 are used only in the development reference environment; the
converter uses PyTorch's restricted `weights_only=True` deserializer and the
runtime never imports Torch.

Zenodo record [21929873](https://zenodo.org/records/21929873) supplies the
contrast-agnostic wholebody v1.4 checkpoint, original JSON configuration and
publisher `LICENSE` under MIT. The license grant matches
[UPSTREAM_LICENSE](UPSTREAM_LICENSE). Converted weights retain all original
float32 parameters and are hosted at
[MedMLX/musclemap-mlx](https://huggingface.co/MedMLX/musclemap-mlx/tree/a798b3073e3b2c449987c13aa193626aaacc3092),
with configuration, labels, license and attribution. No weights or medical images
are included in package distributions.

## Showcase image

No showcase image or patient data is included.
