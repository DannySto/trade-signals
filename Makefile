# This Makefile requires the following commands to be available:
# * poetry (installation guide: https://python-poetry.org/docs/#installing-with-the-official-installer)

# for local development and testing
ENVFILES_SECRETS?=$(CURDIR)/.env
export ENVFILES_SECRETS

## Test reports
UNITTESTS_REPORT_DIR=.

.PHONY: run
run:
	PYTHONPATH="src" poetry run python src/main.py

.PHONY: install
install: REVISION
	poetry install

.PHONY: update
update:
	poetry update --no-ansi

# Updates the lock file
.PHONY: lock
lock:
	poetry check --lock

.PHONY: clean
clean:
	rm -rf .venv
	find . -name '*.py[cod]' -delete
	rm -rf *.egg-info build dist
	rm -rf coverage.xml .coverage junit.xml

REVISION:
	echo "$$(git branch | grep '*' | cut -d ' ' -f 2) $$(git describe --tags --always)" > REVISION

.PHONY: checks
checks:
	poetry check

.PHONY: unittests
unittests:
	PYTHONPATH="src" \
	poetry run coverage run -m pytest -sv . --junitxml=$(UNITTESTS_REPORT_DIR)/junit.xml
	poetry run coverage xml -i -o $(UNITTESTS_REPORT_DIR)/coverage.xml

.PHONY: lint
lint: lint/ruff lint/mypy

.PHONY: lint/ruff
lint/ruff:
	NO_COLOR=1 poetry run ruff check src tests

.PHONY: lint/mypy
lint/mypy:
	poetry run mypy src tests

.PHONY: format
format: format/ruff

.PHONY: format/ruff
format/ruff:
	NO_COLOR=1 poetry run ruff format src tests


## Local Development
.PHONY: test
test: lint checks unittests

# Trivy
trivy/src:
	trivy fs ./src --severity HIGH,CRITICAL  --ignore-unfixed
