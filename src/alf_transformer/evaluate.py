# -*- coding: utf-8 -*-
"""Evaluation of one or more risk models with a common statistical framework.

Produces, for every model:

* Spearman rank correlation between the prediction and the observed outcome
* restricted cubic spline logistic regression with a non-linearity test
* ROC AUC with bootstrap CI and all-pairs DeLong tests
* Hosmer-Lemeshow calibration test
* decision curve analysis
* reproducibility across repeated fits (ICC + repeated-measures ANOVA)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.metrics import roc_auc_score

from .stats import (auc_ci, dca, delong_test, hosmer_lemeshow, icc_1_1, icc_2_1,
                    rcs_nonlinearity_test, rm_anova_f)

__all__ = ["to_probability", "evaluate_models", "dca_summary", "reproducibility"]


def to_probability(v) -> np.ndarray:
    """Map an arbitrary risk score onto (0, 1).

    Values already bounded by [0, 1] are clipped; unbounded scores are squashed
    through a logistic using their own mean and SD, which is adequate for
    calibration and decision-curve displays but does not imply the score is a
    genuine probability.
    """
    v = np.asarray(v, dtype=float)
    if v.min() >= 0 and v.max() <= 1:
        p = v.copy()
    else:
        sd = v.std() if v.std() > 1e-9 else 1.0
        p = 1 / (1 + np.exp(-(v - v.mean()) / sd))
    return np.clip(p, 1e-6, 1 - 1e-6)


def evaluate_models(models: dict, y, thresholds=(0.2, 0.3, 0.4)) -> dict:
    """Run the full evaluation. ``models`` maps a display name to predictions."""
    y = np.asarray(y)
    spearman, rcs, aucs, calib, dca_rows = [], [], [], [], []

    for name, raw in models.items():
        rho, p_rho = _st.spearmanr(raw, y)
        spearman.append({"Model": name, "Spearman_rho": round(rho, 3),
                         "P": "<0.001" if p_rho < 0.001 else f"{p_rho:.3f}"})

        try:
            lr, df, p_nl = rcs_nonlinearity_test(y, raw)
            rcs.append({"Model": name, "LR_chi2": round(lr, 2), "df": df,
                        "P_nonlinear": "<0.001" if p_nl < 0.001 else f"{p_nl:.3f}"})
        except Exception as exc:  # pragma: no cover - defensive
            rcs.append({"Model": name, "LR_chi2": None, "df": None,
                        "P_nonlinear": f"ERR {type(exc).__name__}"})

        a = roc_auc_score(y, raw)
        lo, hi = auc_ci(y, raw)
        aucs.append({"Model": name, "AUC": round(a, 4),
                     "95%CI": f"{lo:.3f}-{hi:.3f}"})

        chi2, p_hl, df_hl = hosmer_lemeshow(y, to_probability(raw))
        calib.append({"Model": name,
                      "HL_chi2": round(chi2, 1) if np.isfinite(chi2) else None,
                      "df": df_hl,
                      "P": f"{p_hl:.3f}" if np.isfinite(p_hl) else "—"})

        d = dca(y, to_probability(raw))
        dca_rows.append({"Model": name,
                         **{f"NB@{t}": round(float(d.loc[(d.threshold - t).abs().idxmin(),
                                                        "net_benefit"]), 4)
                            for t in thresholds}})

    names = list(models)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a1, a2, z, p = delong_test(y, models[names[i]], models[names[j]])
            pairs.append({"Model A": names[i], "Model B": names[j],
                          "AUC_A": round(a1, 4), "AUC_B": round(a2, 4),
                          "Z": round(z, 3) if np.isfinite(z) else None,
                          "P": "<0.001" if (np.isfinite(p) and p < 0.001)
                               else (f"{p:.3f}" if np.isfinite(p) else "—")})

    return {
        "spearman": pd.DataFrame(spearman),
        "rcs": pd.DataFrame(rcs),
        "auc": pd.DataFrame(aucs),
        "delong": pd.DataFrame(pairs),
        "calibration": pd.DataFrame(calib),
        "dca": pd.DataFrame(dca_rows),
    }


def dca_summary(y, models: dict) -> pd.DataFrame:
    """Net-benefit curves for every model, stacked into one long DataFrame."""
    frames = []
    for name, raw in models.items():
        d = dca(y, to_probability(raw))
        d["Model"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def reproducibility(seed_matrix: np.ndarray, name: str) -> dict:
    """ICC(2,1), ICC(1,1) and repeated-measures ANOVA over repeated fits.

    ``seed_matrix`` has one row per case and one column per repeated fit
    (random seed, imputed dataset, bootstrap refit, ...).
    """
    mat = np.asarray(seed_matrix, dtype=float)
    f, p, df1, df2 = rm_anova_f(mat)
    return {
        "Model": name,
        "repeats": mat.shape[1],
        "ICC(2,1)": round(icc_2_1(mat), 4),
        "ICC(1,1)": round(icc_1_1(mat), 4),
        "ANOVA_F": round(f, 3) if np.isfinite(f) else None,
        "ANOVA_df": f"{df1},{df2}",
        "ANOVA_P": ("<0.001" if (np.isfinite(p) and p < 0.001)
                    else (f"{p:.3f}" if np.isfinite(p) else "—")),
        "mean_SD_across_repeats": round(float(np.mean(mat.std(1, ddof=1))), 4),
    }
