"""Compare train_heads runs (main model vs ablations) from their metrics.json files.

    python scripts/compare_runs.py outputs/heads outputs/abl_identity outputs/abl_ft_only outputs/abl_nopred
"""

import argparse
import json
import os

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    args = p.parse_args()
    rows = []
    for d in args.run_dirs:
        with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        for task in ("task1", "task2"):
            if not m.get(task):
                continue
            fold_mcc = [v["Rk-correlation"] for v in m[task]["per_fold"].values()]
            rows.append(
                {
                    "run": d,
                    "task": task,
                    **m["ablation"],
                    "fold_mcc_mean": sum(fold_mcc) / len(fold_mcc),
                    "fold_mcc_min": min(fold_mcc),
                    **{f"oof_{k}": v for k, v in m[task]["oof"].items()},
                }
            )
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
