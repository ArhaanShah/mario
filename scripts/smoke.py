"""End-to-end smoke run on synthetic data: every CLI, in order, as subprocesses (``make smoke``).

CPU only, no network, random-init ViT-S (2 blocks). Outputs go to ``smoke_out/`` (deleted first unless
``--keep``). Fails if any step fails or an expected output is missing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "configs/smoke.yaml"
OUT = "smoke_out"


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print(f"\n$ python {' '.join(args)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  ok ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", action="store_true", help="do not delete smoke_out/ first")
    p.add_argument("--skip-partial", action="store_true", help="skip the partial fine-tune step")
    args = p.parse_args()
    out = os.path.join(ROOT, OUT)
    if os.path.isdir(out) and not args.keep:
        shutil.rmtree(out)

    c = ["--config", CFG]
    run("scripts/make_synthetic_data.py", "--out", f"{OUT}/data", "--seed", "0")
    run("scripts/fetch_weights.py", *c, "--random-init")
    run("scripts/check_label_alignment.py", *c)
    run("-m", "mario_dino.data.splits", *c)
    run("-m", "mario_dino.octip_cache", *c, "--mode", "raw")
    run("-m", "mario_dino.features", *c)
    run("-m", "mario_dino.train_heads", *c)
    run("-m", "mario_dino.train_heads", *c, "--folds", "0", "--set", "model.forecaster=identity",
        "--set", f"train.out_dir={OUT}/abl_identity")
    run("scripts/compare_runs.py", f"{OUT}/heads", f"{OUT}/abl_identity")
    if not args.skip_partial:
        run("-m", "mario_dino.train_partial", *c, "--folds", "0", "1")
    # challenge CSVs without labels
    run("-m", "mario_dino.infer", *c, "--task", "1")
    run("-m", "mario_dino.infer", *c, "--task", "2")
    # labelled CSV: also scores
    run("-m", "mario_dino.infer", *c, "--task", "1", "--csv", f"{OUT}/data/csv/df_task1_val.csv", "--method", "2")
    if not args.skip_partial:
        run("-m", "mario_dino.infer", *c, "--task", "1", "--run-dir", f"{OUT}/partial", "--method", "3")

    expected = [
        f"{OUT}/infer/results_task1_team_dino_method_1.csv",
        f"{OUT}/infer/results_task2_team_dino_method_1.csv",
        f"{OUT}/infer/results_task1_team_dino_method_2.csv",
        f"{OUT}/infer/scores_task1_method_2.json",
        f"{OUT}/heads/metrics.json",
        f"{OUT}/heads/oof_task1.csv",
        f"{OUT}/heads/oof_task2.csv",
    ] + ([f"{OUT}/infer/results_task1_team_dino_method_3.csv"] if not args.skip_partial else [])
    missing = [e for e in expected if not os.path.isfile(os.path.join(ROOT, e))]
    if missing:
        sys.exit(f"smoke FAILED, missing outputs: {missing}")
    print("\nsmoke OK. Outputs:\n  " + "\n  ".join(expected))


if __name__ == "__main__":
    main()
