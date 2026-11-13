# Development

`make env` provisions the locked Python 3.13 environment. `make qa` runs Ruff
and strict Pyright. `make test` runs the offline package and geometry tests.
`make build` builds the sdist and then the wheel from it.

The runtime imports neither Torch nor MONAI. Reference generation uses the
separate pinned environment from `make oracle-env`; conversion uses the restricted
Torch state-dictionary loader and the standard safetensors writer. Converted
parameters are reloaded and compared exactly.

For real-image qualification commands, the pinned upstream contract and measured
results, see [technical.md](technical.md).

The opt-in synthetic regression checks remain available:

```sh
make convert
make fixtures
MEDMLX_TEST_REFERENCE=build/msk-porting/fixtures make oracle
MEDMLX_TEST_WEIGHTS=build/msk-porting/weights \
MEDMLX_TEST_REFERENCE=build/msk-porting/fixtures \
perfkit gpu lock run -- make metal
```

Weights, medical images, fixtures and generated results belong in ignored
folders. `scripts/` is development tooling and is excluded from distributions.
