# Third-party notices

## MuscleMap

- Source: https://github.com/MuscleMap/MuscleMap
- Revision: `35a2138a4a3a63be713bae5584632ffc6433b960`
- Copyright: Copyright (c) 2023 MuscleMap
- License: MIT; original text retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
  The package [LICENSE](LICENSE) retains the same MIT terms.
- Adapted: source remains unchanged in the development-only reference submodule.
  This package adds asset inspection and development packaging; no network code
  has been ported or executed.

## Dependencies, weights and data

Dependencies declared in [pyproject.toml](pyproject.toml) are installed separately
and keep their own licenses. Zenodo record
[21929873](https://zenodo.org/records/21929873) supplies
`contrast_agnostic_wholebody_model.pth`, its original JSON configuration and
`LICENSE`, under the publisher's MIT terms. The license text is identical to
[UPSTREAM_LICENSE](UPSTREAM_LICENSE). These assets remain unchanged in ignored
local storage and are excluded from distributions. No weights are converted or
retrained.

## Showcase image

No showcase image or patient data is included.
