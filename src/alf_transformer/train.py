# -*- coding: utf-8 -*-
"""Cross-validated training with honest early stopping.

Design note
-----------
Early stopping is performed on an **inner** validation split carved out of the
training fold. The outer validation fold is never used for any modelling
decision, so the out-of-fold predictions are free of epoch-selection bias.

Using the outer fold for early stopping inflates the reported AUC by roughly
0.06-0.07 on the cohort this code was developed on. See ``docs/METHODS.md``.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import Dataset, standardize
from .models import build_model

__all__ = ["train_fold", "cross_validate"]


def train_fold(model: nn.Module, fit_idx, val_idx, arrays, y, seed: int,
               epochs: int = 60, batch_size: int = 128, lr: float = 1e-3,
               weight_decay: float = 1e-4, patience: int = 12, verbose: bool = False):
    """Train one fold, early-stopping on an inner validation split.

    ``arrays`` is ``(x_num, x_bin, x_traj, x_mask)`` covering the rows that
    ``fit_idx`` and ``val_idx`` index into.
    """
    x_num, x_bin, x_traj, x_mask = arrays
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()

    def to_t(a):
        return torch.as_tensor(a)

    fit_idx = np.asarray(fit_idx)
    val_idx = np.asarray(val_idx)
    xn_f, xb_f = to_t(x_num[fit_idx]), to_t(x_bin[fit_idx])
    xt_f, xm_f = to_t(x_traj[fit_idx]), to_t(x_mask[fit_idx])
    y_f = to_t(y[fit_idx])
    xn_v, xb_v = to_t(x_num[val_idx]), to_t(x_bin[val_idx])
    xt_v, xm_v = to_t(x_traj[val_idx]), to_t(x_mask[val_idx])

    n = len(fit_idx)
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            out = model(xn_f[idx], xb_f[idx], xt_f[idx], xm_f[idx])
            loss = lossf(out, y_f[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(xn_v, xb_v, xt_v, xm_v)).numpy()
        auc = roc_auc_score(y[val_idx], p_val)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  inner AUC = {auc:.4f}")
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def cross_validate(dataset: Dataset, arch: str, seed: int, n_folds: int = 5,
                   inner_frac: float = 0.15, model_kwargs: dict | None = None,
                   verbose: bool = False):
    """Return out-of-fold predicted probabilities for one architecture and seed."""
    model_kwargs = model_kwargs or {}
    y = dataset.y
    oof = np.full(len(y), np.nan, dtype=np.float64)

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(dataset.x_num, y), 1):
        tr_fit, tr_val = train_test_split(tr, test_size=inner_frac,
                                          stratify=y[tr], random_state=seed)
        (xn_f, xn_v, xn_te), _, _ = standardize(dataset.x_num[tr_fit],
                                                dataset.x_num[tr_val],
                                                dataset.x_num[va])
        arrays = (
            np.vstack([xn_f, xn_v]),
            np.vstack([dataset.x_bin[tr_fit], dataset.x_bin[tr_val]]),
            np.vstack([dataset.x_traj[tr_fit], dataset.x_traj[tr_val]]),
            np.vstack([dataset.x_mask[tr_fit], dataset.x_mask[tr_val]]),
        )
        x_num_te = xn_te
        x_bin_te = dataset.x_bin[va]
        x_traj_te = dataset.x_traj[va]
        x_mask_te = dataset.x_mask[va]

        y_fit = np.concatenate([y[tr_fit], y[tr_val]])
        n_fit = len(tr_fit)
        model = build_model(arch, arrays[0].shape[1], arrays[1].shape[1], **model_kwargs)
        t0 = time.time()
        model, inner_auc = train_fold(model, np.arange(n_fit),
                                      np.arange(n_fit, len(y_fit)),
                                      arrays, y_fit, seed=seed, verbose=verbose)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.as_tensor(x_num_te), torch.as_tensor(x_bin_te),
                                    torch.as_tensor(x_traj_te),
                                    torch.as_tensor(x_mask_te))).numpy()
        oof[va] = p
        if verbose:
            print(f"  fold {fold}/{n_folds}: inner AUC={inner_auc:.4f}  "
                  f"({time.time() - t0:.0f}s)")
    return oof


def run_seeds(dataset: Dataset, arch: str, seeds=(42, 7, 2024), **kwargs):
    """Run ``cross_validate`` for several seeds.

    Returns ``(mean_oof, matrix_of_seed_predictions)``. Averaging across seeds
    acts as a cheap ensemble and is what the reported AUCs are based on; the
    matrix is retained for the reproducibility (ICC) analysis.
    """
    preds = []
    for seed in seeds:
        p = cross_validate(dataset, arch, seed, **kwargs)
        preds.append(p)
    mat = np.column_stack(preds)
    return mat.mean(axis=1), mat
