PY ?= python

.PHONY: test smoke lint third_party

test:
	$(PY) -m pytest -q

# End-to-end pipeline on synthetic data with a small random-init encoder (CPU, no network).
smoke:
	$(PY) scripts/smoke.py

lint:
	$(PY) -m ruff check mario_dino scripts tests

# Fetch the pinned DF41 clone (submodule) if the repo was cloned without --recursive.
third_party:
	git submodule update --init --recursive
