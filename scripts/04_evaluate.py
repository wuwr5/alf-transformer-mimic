#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 4 - evaluate one or more sets of predictions with the common framework.

Accepts any number of ``name=path`` pairs, where the path points to a CSV with
columns ``case_index``, ``prob`` (and optionally ``run``). This makes it
straightforward to place Transformer, classical and large-language-model
predictions on the same footing.

Example
-------
    python scripts/04_evaluate.py \
        --cohort data/cohort.csv \
        --model "FT-Transformer=results/oof_ft.csv" \
        --model "Hybrid=results/oof_hybrid.csv" \
        --out results/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alf_transformer import (DataSpec, evaluate_models, load_dataset,  # noqa: E402
                             reproducibility)


def load_prediction(path: str, n: int) -> np.ndarray:
    df = pd.read_csv(path)
    col = "prob" if "prob" in df.columns else df.columns[-1]
    if "run" in df.columns:
        piv = df.pivot_table(index="case_index", columns="run", values=col)
        return piv.mean(axis=1).reindex(range(n)).to_numpy()
    if "case_index" in df.columns:
        return df.set_index("case_index")[col].reindex(range(n)).to_numpy()
    return df[col].to_numpy()


def load_seed_matrix(path: str, n: int):
    df = pd.read_csv(path)
    if "run" not in df.columns:
        return None
    piv = df.pivot_table(index="case_index", columns="run", values="prob")
    return piv.reindex(range(n)).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.cohort, args.trajectory, DataSpec(outcome=args.outcome))

    models, seed_mats = {}, {}
    for item in args.model:
        name, _, path = item.partition("=")
        models[name] = load_prediction(path, len(ds))
        mat = load_seed_matrix(path, len(ds))
        if mat is not None and mat.shape[1] > 1:
            seed_mats[name] = mat
        print(f"loaded {name}: n={np.isfinite(models[name]).sum()}")

    results = evaluate_models(models, ds.y)
    for key, frame in results.items():
        print(f"\n=== {key} ===")
        print(frame.to_string(index=False))

    repro = pd.DataFrame([reproducibility(m, k) for k, m in seed_mats.items()])
    if len(repro):
        print("\n=== reproducibility across repeated fits ===")
        print(repro.to_string(index=False))

    out = os.path.join(args.out, "evaluation.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for key, frame in results.items():
            frame.to_excel(xw, sheet_name=key[:31], index=False)
        if len(repro):
            repro.to_excel(xw, sheet_name="reproducibility", index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
