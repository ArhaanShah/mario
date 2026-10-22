"""Challenge scoring, using DF41's ``specificity`` from the pinned clone (not copied).

DF41's ``utils/scoring.py`` imports seaborn, matplotlib and torchvision at module level for plotting helpers
that ``specificity`` never touches. They are stubbed for the duration of the import so the main env does not
need them; ``sys.modules`` is restored afterwards.

``score`` returns the keys of DF41's ``InferenceTask{1,2}.scoring``: micro-F1, MCC ("Rk-correlation"),
specificity, and quadratic-weighted kappa for Task 2.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
import types

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

from .utils import THIRD_PARTY

_STUBBED = ("seaborn", "matplotlib", "matplotlib.pylab", "torchvision", "torchvision.transforms")
SELECT_KEYS = {"mcc": "Rk-correlation", "f1": "F1_score", "specificity": "Specificity",
               "qwk": "Quadratic-weighted_Kappa"}


@functools.lru_cache(maxsize=1)
def _df41_scoring() -> types.ModuleType:
    path = os.path.join(THIRD_PARTY, "utils", "scoring.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found: the pinned DF41 clone is missing (git submodule update --init)"
        )
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    try:
        for name in _STUBBED:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["matplotlib"].pylab = sys.modules["matplotlib.pylab"]
        sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
        spec = importlib.util.spec_from_file_location("_df41_scoring", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return module


def specificity(y_true, y_pred) -> float:
    return float(_df41_scoring().specificity(np.asarray(y_true), np.asarray(y_pred)))


def score(y_true, y_pred, task: int) -> dict[str, float]:
    """The challenge metrics, keyed as in DF41's ``scoring()``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "F1_score": float(f1_score(y_true, y_pred, average="micro")),
        "Rk-correlation": float(matthews_corrcoef(y_true, y_pred)),
        "Specificity": specificity(y_true, y_pred),
    }
    if task == 2:
        out["Quadratic-weighted_Kappa"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return out


def selection_value(y_true, y_pred, task: int, metric: str) -> float:
    """Value of the model-selection metric (higher is better)."""
    if metric == "qwk":
        return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic"))
    return score(y_true, y_pred, task)[SELECT_KEYS[metric]]
