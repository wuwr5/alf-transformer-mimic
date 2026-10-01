#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 3 - train the Transformers and write out-of-fold predictions.

Example
-------
    python scripts/03_train.py \
        --cohort data/cohort.csv \
        --trajectory data/trajectory_panel.csv \
        --arch ft hybrid \
        --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alf_transformer import DataSpec, load_dataset, run_seeds  # noqa: E402

ARCHS = {"ft": "FT-Transformer", "hybrid": "Hybrid Transformer"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--arch", nargs="+", default=["ft", "hybrid"], choices=list(ARCHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec = DataSpec(outcome=args.outcome)
    ds = load_dataset(args.cohort, args.trajectory, spec)
    print(f"n={len(ds)}  events={int(ds.y.sum())} ({100 * ds.y.mean():.1f}%)  "
          f"numeric={ds.x_num.shape[1]}  binary={ds.x_bin.shape[1]}  "
          f"trajectory={ds.x_traj.shape}")

    summary, matrices = {}, {}
    long_rows = []
    for arch in args.arch:
        name = ARCHS[arch]
        print(f"\n=== {name}: {args.folds}-fold CV x {len(args.seeds)} seeds ===")
        oof, mat = run_seeds(ds, arch=arch, seeds=tuple(args.seeds),
                             n_folds=args.folds)
        matrices[name] = mat
        summary[name] = {
            "auc_mean_seed_ensemble": round(float(roc_auc_score(ds.y, oof)), 4),
            "auc_per_seed": [round(float(roc_auc_score(ds.y, mat[:, i])), 4)
                             for i in range(mat.shape[1])],
        }
        print(f"  ensemble AUC = {summary[name]['auc_mean_seed_ensemble']:.4f}  "
              f"(per seed: {summary[name]['auc_per_seed']})")
        np.save(os.path.join(args.out, f"oof_{arch}.npy"), oof)
        np.save(os.path.join(args.out, f"seed_matrix_{arch}.npy"), mat)

        # Long format: case_index / model / run / prob. Identical shape to the
        # output of an LLM prediction run, so both can be fed to 04_evaluate.py.
        for run in range(mat.shape[1]):
            for i in range(len(ds)):
                long_rows.append({"case_index": i, "model": name, "run": run,
                                  "prob": float(mat[i, run]),
                                  "ok": bool(np.isfinite(mat[i, run]))})

    pd.DataFrame(long_rows).to_csv(
        os.path.join(args.out, "oof_predictions_long.csv"), index=False)
    pd.DataFrame({"y": ds.y, **{k: v.mean(1) for k, v in matrices.items()}}).to_csv(
        os.path.join(args.out, "oof_predictions_wide.csv"), index=False)
    with open(os.path.join(args.out, "training_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote results to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
