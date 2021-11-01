# Makefile
SHELL := /bin/bash
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates development environment."
	@echo "style   : runs style formatting."
	@echo "clean   : cleans all unnecessary files."

# Environment
.ONESHELL:
venv: $(INSTALL_STAMP)
$(INSTALL_STAMP): pyproject.toml poetry.lock
		@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
		$(POETRY) install
		$(POETRY) shell
		touch $(INSTALL_STAMP)


# Styling
.PHONY: style
style:
	autopep8 --in-place --aggressive --aggressive --recursive *.py
	black .
	isort .
	flake8

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

.ONESHELL:
run_docker:
	docker run --rm \
	 -v $(PROJECT_DIR)/images:/opt/pysetup/images \
	 -v $(PROJECT_DIR)/models:/opt/pysetup/models \
	 -v $(PROJECT_DIR)/logs:/opt/pysetup/logs \
	 -it customerchurnsolution:latest