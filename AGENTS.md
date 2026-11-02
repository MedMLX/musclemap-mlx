# musclemap-mlx

Native MLX port of MuscleMap v1.4 contrast-agnostic whole-body segmentation.

## Rules

- Work on dev. Main is the last release; do not release this package.
- Follow sibling standards/runtime-standard.md and templates/runtime.
- Keep the 114 output channels, the original 113 anatomical label IDs, instance
  normalization (no affine), LeakyReLU 0.01 and residual semantics.
- Preserve RAS orientation, `pix_dim [1, 1, -1]` (keep source slice spacing),
  nonzero normalization, crop margin 20, end padding to `[256, 256, 1]`,
  bilinear inverse geometry, argmax order and label remap.
- Network inference is native MLX on Metal. CPU work is limited to NIfTI IO,
  preprocessing, Gaussian window blending, inverse geometry, argmax and the
  largest-component filter. No PyTorch or CPU network fallback.
- Keep interfaces typed; no typing.Any or type-ignore comments.
- Tests never import scripts/. Scripts run by path.
- Keep numerical fixtures, converted weights, synthetic images and run artifacts
  under ignored `build/`; never commit them.
- GPU tests and measurements need `perfkit gpu lock run -- ...`; never bypass or
  delete the shared lock.

## Upstream code, weights and data

- Keep reference/ read-only and preserve upstream licensing and weight terms.
- Upstream code is MuscleMap/MuscleMap at
  `35a2138a4a3a63be713bae5584632ffc6433b960` (MIT).
- Assets are Zenodo record 21929873, contrast-agnostic wholebody v1.4. Verify
  the publisher MD5 and SHA-256 before conversion; load the state dictionary
  with `torch.load(..., weights_only=True)`.
- Never work around executable checkpoint safety or invent trained parameters.
- Review overlapping changes and stage only your own requested work.

## Docs and releases

- `main` is the last release and only release work changes it.
- README.md is the only user documentation; developer verification belongs in
  docs/development.md.
- This file, Makefile, docs/development.md, environment/ and reference/ are
  dev-only.
- Do not advertise hosted converted weights or automatic downloads before they
  exist. `setup_supported` remains false until a converted repository is
  published.

## Checks

- make env creates the locked Python 3.13 environment.
- make qa runs Ruff and strict Pyright; requires host Node and runs no models.
- make test runs the offline package and CPU geometry contracts; no weights.
- make metal runs the opt-in native parity tests under the shared GPU lock.
- make oracle runs the opt-in CPU MONAI 1.3.2 geometry checks.
- make fixtures records synthetic MONAI reference outputs with the Python 3.11
  oracle environment.
- make convert writes deterministic safetensors and a conversion record.
- make build builds the sdist and wheel as users would resolve them.
- .venv/bin/python -I ../standards/check_standard.py . checks committed dev only.
- No numerical parity has been established on patient data. Do not claim it from
  synthetic fixtures or package checks.
- There is no hosted CI; local checks are authoritative.
