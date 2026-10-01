# -*- coding: utf-8 -*-
"""Statistical utilities used throughout the pipeline.

All functions are dependency-light (numpy/scipy/statsmodels/pandas) and unit-tested
in ``tests/test_stats.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

__all__ = [
    "rcs_basis", "delong_test", "auc_ci", "hosmer_lemeshow", "dca",
    "icc_2_1", "icc_1_1", "rm_anova_f", "forward_wald",
]


# --------------------------------------------------------------------- RCS
def rcs_basis(x, knots):
    """Harrell restricted cubic spline basis.

    Parameters
    ----------
    x : array-like
        Continuous predictor.
    knots : sequence of float
        Knot locations, ascending. ``k`` knots yield ``k - 1`` basis columns
        (the first column is the linear term).

    Returns
    -------
    ndarray of shape (len(x), k - 1)
    """
    x = np.asarray(x, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)
    if k < 3:
        raise ValueError("restricted cubic splines require at least 3 knots")
    cols = [x]
    tk, tk1 = knots[-1], knots[-2]
    for j in range(1, k - 1):
        tj = knots[j - 1]
        a = np.clip(x - tj, 0, None) ** 3
        b = np.clip(x - tk1, 0, None) ** 3 * (tk - tj) / (tk - tk1)
        c = np.clip(x - tk, 0, None) ** 3 * (tk1 - tj) / (tk - tk1)
        cols.append(a - b + c)
    return np.column_stack(cols)


def rcs_nonlinearity_test(y, x, knots=None):
    """Likelihood-ratio test for non-linearity of ``x`` on binary ``y``.

    Returns ``(chi2, df, p_value)`` comparing a linear-logit model against a
    restricted cubic spline model.
    """
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    if knots is None:
        knots = np.percentile(x, [10, 50, 90])
    B = rcs_basis(x, knots)
    m1 = sm.Logit(y, sm.add_constant(pd.DataFrame(B[:, :1], columns=["x"]),
                                     has_constant="add")).fit(disp=0, maxiter=200)
    m2 = sm.Logit(y, sm.add_constant(pd.DataFrame(B, columns=[f"b{i}" for i in range(B.shape[1])]),
                                     has_constant="add")).fit(disp=0, maxiter=200)
    lr = 2 * (m2.llf - m1.llf)
    df = B.shape[1] - 1
    return float(lr), int(df), float(1 - stats.chi2.cdf(lr, df)) if df > 0 else np.nan


# ------------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    sr = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        sr[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T = np.zeros(N)
    T[J] = sr
    return T


def delong_test(y_true, p1, p2):
    """DeLong test comparing two correlated ROC AUCs (Sun & Xu fast algorithm).

    Returns ``(auc1, auc2, z, p_value)``.
    """
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)
    y = y[order]
    p1 = np.asarray(p1, dtype=float)[order]
    p2 = np.asarray(p2, dtype=float)[order]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan, np.nan

    def comp(p):
        pos, neg = p[:n1], p[n1:]
        tx, ty = _midrank(pos), _midrank(neg)
        tz = _midrank(np.concatenate([pos, neg]))
        auc = (tz[:n1].sum() / n1 - (n1 + 1) / 2) / n0
        return auc, (tz[:n1] - tx) / n0, 1 - (tz[n1:] - ty) / n1

    a1, v01_1, v10_1 = comp(p1)
    a2, v01_2, v10_2 = comp(p2)
    S = np.cov(np.vstack([v01_1, v01_2])) / n0 + np.cov(np.vstack([v10_1, v10_2])) / n1
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        # Degenerate case: the two prediction vectors are identical, so the
        # difference has zero variance and the test is uninformative.
        if abs(a1 - a2) < 1e-12:
            return float(a1), float(a2), 0.0, 1.0
        return float(a1), float(a2), np.nan, np.nan
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def auc_ci(y, p, n_boot=1000, seed=42):
    """Bootstrap 95% CI for a single AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------- Hosmer-Lemeshow
def hosmer_lemeshow(y, p, g=10):
    """Hosmer-Lemeshow goodness-of-fit test. Returns ``(chi2, p_value, df)``.

    Returns ``(nan, nan, 0)`` when the predictions are (near-)constant, because
    the test is undefined without usable risk strata.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if np.unique(p).size < 3:
        return float("nan"), float("nan"), 0
    bins = pd.qcut(p, g, labels=False, duplicates="drop")
    obs = pd.Series(y).groupby(bins).sum()
    cnt = pd.Series(y).groupby(bins).size()
    if len(cnt) < 3:
        return float("nan"), float("nan"), max(len(cnt) - 2, 0)
    exp = np.clip(pd.Series(p).groupby(bins).sum(), 1e-9, cnt - 1e-9)
    stat = float((((obs - exp) ** 2) / (exp * (1 - exp / cnt))).sum())
    df = len(cnt) - 2
    return stat, float(1 - stats.chi2.cdf(stat, df)), df


# --------------------------------------------------------------------- DCA
def dca(y, p, thresholds=None):
    """Decision curve analysis. Returns a DataFrame with net benefit curves."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.61, 0.01)
    prev = y.mean()
    rows = []
    for pt in thresholds:
        pos = p >= pt
        tp = int((pos & (y == 1)).sum())
        fp = int((pos & (y == 0)).sum())
        rows.append({
            "threshold": pt,
            "net_benefit": tp / n - fp / n * (pt / (1 - pt)),
            "treat_all": prev - (1 - prev) * (pt / (1 - pt)),
            "treat_none": 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- ICC
def icc_2_1(mat):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    grand = mat.mean()
    ms_row = k * ((mat.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_col = n * ((mat.mean(0) - grand) ** 2).sum() / (k - 1)
    ss_tot = ((mat - grand) ** 2).sum()
    ms_err = (ss_tot - ms_row * (n - 1) - ms_col * (k - 1)) / ((n - 1) * (k - 1))
    den = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
    return float((ms_row - ms_err) / den) if den > 0 else np.nan


def icc_1_1(mat):
    """ICC(1,1): one-way random effects, single measurement."""
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    grand = mat.mean()
    ms_row = k * ((mat.mean(1) - grand) ** 2).sum() / (n - 1)
    ss_tot = ((mat - grand) ** 2).sum()
    ms_err = (ss_tot - ms_row * (n - 1)) / (n * (k - 1))
    den = ms_row + (k - 1) * ms_err
    return float((ms_row - ms_err) / den) if den > 0 else np.nan


def rm_anova_f(mat):
    """One-way repeated-measures ANOVA, closed form.

    Implemented directly rather than via ``statsmodels.AnovaRM`` because the
    latter builds a dense (n*k) x (n*k) design matrix and exhausts memory for
    cohorts with a few thousand subjects.

    Returns ``(F, p_value, df1, df2)``.
    """
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    grand = mat.mean()
    ss_total = ((mat - grand) ** 2).sum()
    ss_subj = k * ((mat.mean(1) - grand) ** 2).sum()
    ss_rep = n * ((mat.mean(0) - grand) ** 2).sum()
    ss_err = ss_total - ss_subj - ss_rep
    df1, df2 = k - 1, (n - 1) * (k - 1)
    if df1 <= 0 or df2 <= 0 or ss_err <= 0:
        return np.nan, np.nan, df1, df2
    F = (ss_rep / df1) / (ss_err / df2)
    return float(F), float(1 - stats.f.cdf(F, df1, df2)), df1, df2


# --------------------------------------------------- forward Wald selection
def forward_wald(X, y, candidates, p_enter=0.05, max_vars=12, verbose=False):
    """Forward stepwise selection using the Wald p-value of each added term.

    Returns the list of selected column names.
    """
    selected: list[str] = []
    remaining = list(candidates)
    while remaining and len(selected) < max_vars:
        best = None
        for c in remaining:
            try:
                m = sm.Logit(y, sm.add_constant(X[selected + [c]], has_constant="add")
                             ).fit(disp=0, maxiter=200)
            except Exception:
                continue
            pv = m.pvalues.get(c, 1.0)
            if best is None or pv < best[1]:
                best = (c, pv)
        if best is None or best[1] >= p_enter:
            break
        selected.append(best[0])
        remaining.remove(best[0])
        if verbose:
            print(f"  step {len(selected):2d}: {best[0]:24s} p={best[1]:.2e}")
    return selected
