import os

import pytest

from mario_dino.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """One synthetic dataset per test session: {'root', 'images', 'csv': {name: path}}."""
    root = str(tmp_path_factory.mktemp("synth"))
    paths = make_synthetic_dataset(os.path.join(root, "data"), seed=0, n_patients=8, n_heldout_patients=2)
    return {"root": root, "images": os.path.join(root, "data", "images"), "csv": paths}
