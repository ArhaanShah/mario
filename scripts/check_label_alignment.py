"""Print the Task 2 vs Task 1 label crosstab for Task 2 images that start a Task 1 pair.

    python scripts/check_label_alignment.py --config configs/base.yaml          # training CSVs from config
    python scripts/check_label_alignment.py --task1-csv a.csv --task2-csv b.csv  # explicit CSVs

On DF41's val CSVs (third_party/mario_challenge_code/csv/) this prints 2140 / 3822 (56%). Enable
``train.task2_aux_from_task1_pairs`` only if agreement on the real training CSVs is high.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from mario_dino.config import load_config  # noqa: E402
from mario_dino.data import schema  # noqa: E402
from mario_dino.data.alignment import label_alignment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--task1-csv", action="append", default=None)
    p.add_argument("--task2-csv", action="append", default=None)
    args = p.parse_args()
    if args.task1_csv and args.task2_csv:
        t1_paths, t2_paths = args.task1_csv, args.task2_csv
    else:
        cfg = load_config(args.config or "configs/base.yaml")
        t1_paths, t2_paths = cfg.data.task1_train_csvs, cfg.data.task2_train_csvs
    t1 = schema.read_csvs(t1_paths, 1, check_images=False)
    t2 = schema.read_csvs(t2_paths, 2, check_images=False)
    r = label_alignment(t1, t2)
    with pd.option_context("display.width", 120):
        print(r["crosstab"])
    print(
        f"\nTask 2 rows: {r['n_task2']}, starting a Task 1 pair: {r['n_matched']}, "
        f"same label: {r['n_agree']} ({100 * r['agreement']:.1f}% of Task 2 rows)"
    )


if __name__ == "__main__":
    main()
