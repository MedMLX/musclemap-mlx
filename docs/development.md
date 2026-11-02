# musclemap-mlx development

## Checks

```sh
make env
make qa
make test
make build
uv run --locked musclemap-mlx status
uv run --locked musclemap-mlx inspect-assets --model-dir weights
```

The CLI verifies publisher bytes and parses JSON only. It never deserializes a
checkpoint. Tests exercise missing and incomplete assets, same-size corruption, original
configuration labels and the CLI's honest implementation status; they neither download weights nor run inference.

Run the structural checker after committing authorized changes to `dev`:

```sh
.venv/bin/python -I ../standards/check_standard.py .
```

It reads committed `dev`, not working files. Keep changes scoped to the requested
work; a successful structural check does not establish native inference.

## Reference fixtures

No numerical fixtures exist yet. A native port needs an independent upstream
recorder on synthetic inputs, conversion checks, preprocessing and geometry
comparisons, and native Metal parity before it can advertise inference. Do not
substitute zeros, placeholder outputs or a PyTorch inference fallback.

## Upstream

- Code: MuscleMap/MuscleMap at
  `35a2138a4a3a63be713bae5584632ffc6433b960`, in read-only `reference` submodule.
- Assets: Zenodo record 21929873, contrast-agnostic wholebody v1.4; preserve the
  downloaded original JSON and its 113 anatomical entries unchanged.
- Architecture: 2D MONAI UNet, 114 outputs, LeakyReLU, instance normalization.
- Pipeline to preserve: RAS orientation, configured spacing `[1, 1, -1]`,
  nonzero intensity normalization, foreground crop margin 20, end padding to
  `[256, 256, 1]`, slice inference, inverse transforms, argmax and label remap.
- Checkpoint conversion, native model, runner, parity and central registration
  remain development work. No upstream inference dependencies are installed.

Licensing and publisher terms are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
