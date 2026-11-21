# Quarkon development tasks.
#
#   make install     install the package in editable mode with dev extras
#   make test        run the test suite (no GPU or network required)
#   make typecheck   run mypy in strict mode
#   make lint        run ruff
#   make check       typecheck + lint + test
#   make clean       remove caches and build artefacts

PYTHON ?= python
CONFIG ?= examples/distill.yaml

.PHONY: install test typecheck typecheck-linux lint check clean validate dry-run ui doctor dist release-check help

help:
	@echo "targets: install test typecheck typecheck-linux lint check clean validate dry-run dist release-check"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

typecheck:
	$(PYTHON) -m mypy quarkon quark

# CI runs mypy on Linux; platform-specific branches (ctypes.windll, /proc) must
# type-check there too, which a local Windows run cannot prove on its own.
typecheck-linux:
	$(PYTHON) -m mypy --platform linux quarkon quark

lint:
	$(PYTHON) -m ruff check quarkon quark tests

check: typecheck lint test

validate:
	$(PYTHON) -m quarkon --config $(CONFIG) validate

ui:
	$(PYTHON) -m quarkon ui

doctor:
	$(PYTHON) -m quarkon doctor

dry-run:
	$(PYTHON) -m quarkon --config $(CONFIG) synthesize --dry-run

dist:
	$(PYTHON) -m build --wheel --sdist --outdir dist

# Everything a tag must pass: lint, types, tests, then a clean artefact build.
release-check: check dist
	@echo "release artefacts:"
	@ls -l dist

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .workspace
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
