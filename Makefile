.DEFAULT_GOAL := qa
.PHONY: env check-env lint typecheck test metal oracle fixtures convert oracle-env qa build loc clean


TESTS ?= tests
PYTEST_ARGS ?=
SYNC_ARGS ?=
PYTHON ?= .venv/bin/python
ORACLE_PYTHON ?= environment/oracle/.venv/bin/python
SOURCE_DIRS ?= src tests scripts
FIXTURE_DIR ?= build/msk-porting/fixtures
CONVERTED_DIR ?= build/msk-porting/weights
export PYTHONNOUSERSITE := 1
export MLX_ENABLE_TF32 ?= 0

env:
	uv sync --locked --all-extras $(SYNC_ARGS)

check-env:
	@command -v "$(PYTHON)" >/dev/null || { echo "Missing project Python: run make env explicitly, or set PYTHON to its existing interpreter" >&2; exit 1; }

lint: check-env
	$(PYTHON) -m ruff check .

typecheck: check-env
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(PYTHON) -m pyright --project . --pythonpath "$(PYTHON)" --warnings
	bash -o pipefail -c 'find $(wildcard $(SOURCE_DIRS)) -type f \( -name "*.py" -o -name "*.pyi" \) -print0 | xargs -0 $(PYTHON) -m ruff check --config pyproject.toml --select TID251,ANN401 --ignore-noqa --no-force-exclude --'

test: check-env
	$(PYTHON) -m pytest $(TESTS) -m "not oracle and not metal" $(PYTEST_ARGS)

metal: check-env
	$(PYTHON) -m pytest $(TESTS) -m "metal" $(PYTEST_ARGS)

oracle: check-env
	$(PYTHON) -m pytest $(TESTS) -m "oracle and not metal" $(PYTEST_ARGS)

oracle-env:
	uv venv --python 3.11 --allow-existing environment/oracle/.venv
	uv pip install --python $(ORACLE_PYTHON) -r environment/oracle/requirements.txt

fixtures: oracle-env
	PYTHONPATH=src $(ORACLE_PYTHON) scripts/make_reference_fixtures.py --assets weights --output $(FIXTURE_DIR)

convert: check-env
	$(PYTHON) scripts/convert_checkpoint.py --checkpoint weights/contrast_agnostic_wholebody_model.pth --config weights/contrast_agnostic_wholebody_model.json --output $(CONVERTED_DIR)/model.safetensors

qa: lint typecheck

build:
	rm -rf dist
	uv build --no-sources

loc:
	@find $(wildcard $(SOURCE_DIRS)) -type f \( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 | xargs -0 wc -l

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov .coverage build dist
	find . -name .venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
