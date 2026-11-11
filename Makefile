PY ?= python

.PHONY: test benchmark-check smoke lint third_party

test:
	$(PY) -m pytest -q

# Artifact-only validation: skip the repository's image/synthetic conftest.
benchmark-check:
	$(PY) -m pytest --noconftest tests/test_benchmark_artifacts.py -q -p no:cacheprovider

# DINO pipeline on synthetic data with random-init weights (CPU, no network).
# ConvNeXt/VisionFM full fine-tuning is covered separately by test_alternative_encoders.py.
smoke:
	$(PY) scripts/smoke.py

lint:
	$(PY) -m ruff check mario_dino scripts tests

# Fetch the pinned DF41 clone (submodule) if the repo was cloned without --recursive.
third_party:
	git submodule update --init --recursive
