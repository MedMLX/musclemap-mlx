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

Zenodo record [21929873](https://zenodo.org/records/21929873) supplies
`contrast_agnostic_wholebody_model.pth` (104,884,148 bytes, MD5
`910b722aeb641c380404c99ec6d1af97`, SHA-256
`45dfa2843d2e0b1fd842152d6a79bfd4bcb90899c076ceb6346c59da9a79a16c`), its
original JSON configuration and the publisher `LICENSE`, under the publisher's
MIT terms. The license text is identical to [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
These assets remain unchanged in ignored local storage and are excluded from
distributions. No weights are retrained. Converted safetensors are local
artifacts and are not hosted by this repository yet.

## Showcase image

No showcase image or patient data is included.
