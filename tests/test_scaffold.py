import os
import subprocess
import sys

import pytest

from mario_dino.config import ConfigError, load_config
from mario_dino.utils import REPO_ROOT


def test_import_does_not_pull_tensorflow():
    # imports every module in the package, so a module-level TensorFlow import anywhere fails here
    code = (
        "import importlib, pkgutil, sys, mario_dino\n"
        "for m in pkgutil.walk_packages(mario_dino.__path__, 'mario_dino.'): importlib.import_module(m.name)\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras', 'segmentation_models')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_configs_load():
    for name in ("base", "task1", "task2", "smoke"):
        cfg = load_config(os.path.join(REPO_ROOT, "configs", f"{name}.yaml"))
        assert cfg.data.n_folds >= 2


def test_config_override_and_float_coercion(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 1e-3\n")
    cfg = load_config(str(p), ["model.forecaster=identity", "train.folds=[0, 2]"])
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.model.forecaster == "identity"
    assert cfg.train.folds == [0, 2]


def test_config_inheritance(tmp_path):
    (tmp_path / "parent.yaml").write_text("seed: 3\ntrain:\n  patience: 2\n  lr: 0.5\n")
    (tmp_path / "child.yaml").write_text("base: parent.yaml\ntrain:\n  lr: 0.1\n")
    cfg = load_config(str(tmp_path / "child.yaml"))
    assert (cfg.seed, cfg.train.patience, cfg.train.lr) == (3, 2, 0.1)


def test_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rate: 0.1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(str(p))


def test_octip_side_modules_do_not_need_torch():
    # the octip env (TensorFlow 2.10) has no torch: octip_cache and splits must not import it
    code = (
        "import sys, mario_dino.octip_cache, mario_dino.data.splits\n"
        "assert 'torch' not in sys.modules and 'timm' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
