# musclemap-mlx

Development intake for MuscleMap v1.4; no native inference exists yet.

## Rules

- Work on dev. Main is the last release; do not release this intake package.
- Follow sibling standards/runtime-standard.md and templates/runtime.
- Keep reference/ read-only and preserve upstream licensing and weight terms.
- Never commit weights, medical images, patient data or generated outputs.
- Keep original anatomical label IDs and upstream preprocessing semantics.
- Network inference, when implemented, must run natively in MLX on Metal.
  CPU I/O and preprocessing are allowed; no PyTorch or CPU network fallback.
- Do not advertise inference, conversion or MedMLX registration before they exist.
- Keep interfaces typed; no typing.Any or type-ignore comments.
- Tests never import scripts/. Scripts run by path.

## Checks

- make env creates the locked development environment.
- make qa runs Ruff and strict Pyright; requires host Node and runs no models.
- make test exercises the local package contract; no weight downloads.
- make build builds the sdist and wheel as users would resolve them.
- .venv/bin/python -I ../standards/check_standard.py . checks committed dev only.
- No numerical parity has been established. Do not claim it from package checks.
- There is no hosted CI; local checks are authoritative.
