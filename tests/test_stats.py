# -*- coding: utf-8 -*-
"""Unit tests for the statistical routines.

Run with ``pytest -q`` from the repository root.
"""
import numpy as np
import pytest

from alf_transformer.stats import (auc_ci, dca, delong_test, forward_wald,
                                   hosmer_lemeshow, icc_1_1, icc_2_1,
                                   rcs_basis, rcs_nonlinearity_test, rm_anova_f)


# ------------------------------------------------------------------- RCS
def test_rcs_shape_and_knot_count():
    x = np.linspace(0, 10, 200)
    assert rcs_basis(x, [1, 5, 9]).shape == (200, 2)
    assert rcs_basis(x, [1, 4, 7, 9]).shape == (200, 3)
    with pytest.raises(ValueError):
        rcs_basis(x, [1, 5])


def test_rcs_first_column_is_linear_term():
    x = np.linspace(-5, 5, 100)
    assert np.allclose(rcs_basis(x, [-3, 0, 3])[:, 0], x)


def test_rcs_linear_data_is_not_flagged_as_non_linear():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    p = 1 / (1 + np.exp(-(0.8 * x)))
    y = rng.binomial(1, p)
    _, _, p_nl = rcs_nonlinearity_test(y, x)
    assert p_nl > 0.05


def test_rcs_detects_non_linearity():
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, size=4000)
    p = 1 / (1 + np.exp(-(1.5 * x ** 2 - 1.0)))
    y = rng.binomial(1, p)
    _, _, p_nl = rcs_nonlinearity_test(y, x)
    assert p_nl < 0.001


# ---------------------------------------------------------------- DeLong
def test_delong_identical_predictors_gives_zero_z():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.4, size=400)
    p = rng.uniform(size=400)
    a1, a2, z, pval = delong_test(y, p, p)
    assert a1 == pytest.approx(a2, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert pval == pytest.approx(1.0, abs=1e-6)


def test_delong_detects_clearly_better_model():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, size=1500)
    good = y + rng.normal(0, 1.0, size=1500)
    bad = rng.normal(0, 1.0, size=1500)
    a1, a2, z, pval = delong_test(y, good, bad)
    assert a1 > a2
    assert pval < 1e-6


def test_auc_ci_brackets_point_estimate():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.4, size=800)
    # deliberately noisy so that AUC < 1 and the interval has positive width
    p = 0.35 * y + rng.normal(0.2, 0.5, size=800)
    from sklearn.metrics import roc_auc_score
    lo, hi = auc_ci(y, p, n_boot=300)
    point = roc_auc_score(y, p)
    assert 0.5 < point < 1.0
    assert lo < point <= hi


# --------------------------------------------------- Hosmer-Lemeshow
def test_hl_accepts_well_calibrated_predictions():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = rng.binomial(1, p)
    _, pval, _ = hosmer_lemeshow(y, p)
    assert pval > 0.01


def test_hl_rejects_badly_calibrated_predictions():
    rng = np.random.default_rng(6)
    # predictions hover around 0.2 but the event rate is 0.8
    p = np.clip(0.2 + rng.normal(0, 0.01, size=3000), 0.01, 0.99)
    y = rng.binomial(1, 0.8, size=3000)
    _, pval, _ = hosmer_lemeshow(y, p)
    assert pval < 1e-6


def test_hl_returns_nan_for_constant_predictions():
    y = np.random.default_rng(11).binomial(1, 0.5, size=500)
    stat, pval, df = hosmer_lemeshow(y, np.full(500, 0.5))
    assert not np.isfinite(stat)
    assert not np.isfinite(pval)
    assert df == 0


# ------------------------------------------------------------------- DCA
def test_dca_net_benefit_matches_manual_formula():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2, 0.8, 0.3])
    d = dca(y, p, thresholds=[0.5])
    tp = int(((p >= 0.5) & (y == 1)).sum())
    fp = int(((p >= 0.5) & (y == 0)).sum())
    expected = tp / len(y) - fp / len(y) * (0.5 / 0.5)
    assert d.loc[0, "net_benefit"] == pytest.approx(expected)


# ------------------------------------------------------------------- ICC
def test_icc_perfect_agreement_is_one():
    mat = np.tile(np.arange(50, dtype=float)[:, None], (1, 4))
    assert icc_2_1(mat) == pytest.approx(1.0)
    assert icc_1_1(mat) == pytest.approx(1.0)


def test_icc_drops_when_repeats_disagree():
    rng = np.random.default_rng(7)
    agree = np.tile(rng.normal(size=200)[:, None], (1, 3))
    noise = rng.normal(size=(200, 3))
    assert icc_2_1(agree) > icc_2_1(noise) + 0.5


# ------------------------------------------------------------- RM-ANOVA
def test_rm_anova_no_effect_gives_high_p():
    rng = np.random.default_rng(8)
    mat = rng.normal(size=(300, 3))
    f, p, df1, df2 = rm_anova_f(mat)
    assert (df1, df2) == (2, 598)
    assert p > 0.01


def test_rm_anova_detects_shift_between_repeats():
    rng = np.random.default_rng(9)
    mat = rng.normal(scale=0.5, size=(300, 3))
    mat[:, 1] += 1.0          # systematic offset in the second run
    f, p, _, _ = rm_anova_f(mat)
    assert f > 10
    assert p < 1e-6


# --------------------------------------------------- forward Wald selection
def test_forward_wald_recovers_signal_and_ignores_noise():
    import pandas as pd
    rng = np.random.default_rng(10)
    n = 3000
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * signal))))
    X = pd.DataFrame({"signal": signal, "noise": noise})
    selected = forward_wald(X, y, ["signal", "noise"], max_vars=2)
    assert "signal" in selected
    assert "noise" not in selected
