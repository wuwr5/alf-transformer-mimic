# -*- coding: utf-8 -*-
"""Data loading, cohort assembly and trajectory construction.

The package never ships MIMIC-IV data. See ``docs/DATA.md`` for how to obtain
the source data under the PhysioNet credentialed licence.

Expected inputs (all produced by ``scripts/``):

``cohort.csv``
    One row per ICU stay with the analysis variables already imputed
    (see ``scripts/02_impute.R``). Must contain ``stay_id`` and the outcome.
``trajectory_panel.csv``
    Wide panel with columns ``<var>_d0`` .. ``<var>_d6`` for each trajectory
    variable, keyed by ``stay_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["DataSpec", "Dataset", "load_dataset", "standardize"]

TRAJ_VARS_DEFAULT = ("bilirubin", "inr", "creatinine", "platelet")
NUM_VARS_DEFAULT = ("Age", "MAP", "SpO2", "Hemoglobin", "Platelet", "WBC", "Albumin",
                    "BUN", "Chloride", "Creatinine", "Sodium", "ALT", "AST",
                    "Total_bilirubin", "PT", "APTT", "INR", "Lactate", "MELD", "SOFA")
BIN_VARS_DEFAULT = ("Alcoholic_only", "Alcoholic_plus_viral", "Viral_only", "Other",
                    "Ascites", "Sepsis", "HE", "HRS", "EVB", "SBP", "Shock",
                    "Pneumonia", "Liver_transplantation", "Vasopressin", "rrt")


@dataclass
class DataSpec:
    """Declarative description of the modelling inputs."""

    outcome: str = "outcome"
    n_days: int = 7
    num_vars: Sequence[str] = field(default_factory=lambda: NUM_VARS_DEFAULT)
    bin_vars: Sequence[str] = field(default_factory=lambda: BIN_VARS_DEFAULT)
    traj_vars: Sequence[str] = field(default_factory=lambda: TRAJ_VARS_DEFAULT)


@dataclass
class Dataset:
    """Container passed to the training routines."""

    frame: pd.DataFrame
    y: np.ndarray
    x_num: np.ndarray
    x_bin: np.ndarray
    x_traj: np.ndarray
    x_mask: np.ndarray
    spec: DataSpec

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.y)

    @property
    def n_traj_vars(self) -> int:
        return self.x_traj.shape[-1]


def _trajectory_arrays(frame: pd.DataFrame, spec: DataSpec):
    n = len(frame)
    nv = len(spec.traj_vars)
    traj = np.full((n, spec.n_days, nv), np.nan, dtype=np.float32)
    for i, var in enumerate(spec.traj_vars):
        for d in range(spec.n_days):
            col = f"{var}_d{d}"
            if col in frame.columns:
                traj[:, d, i] = frame[col].to_numpy()
    mask = (~np.isnan(traj)).astype(np.float32)

    # Median-fill per day per variable. The model always sees the mask, so this
    # only provides a numeric placeholder and does not fabricate information.
    for d in range(spec.n_days):
        for i in range(nv):
            col = traj[:, d, i]
            med = np.nanmedian(col)
            if np.isnan(med):
                med = 0.0
            col[np.isnan(col)] = med
    return traj, mask


def load_dataset(cohort_path: str, trajectory_path: str | None = None,
                 spec: DataSpec | None = None) -> Dataset:
    """Load the cohort, optionally merge the trajectory panel, return arrays."""
    spec = spec or DataSpec()
    frame = pd.read_csv(cohort_path)
    if trajectory_path:
        traj = pd.read_csv(trajectory_path)
        tcols = [f"{v}_d{d}" for v in spec.traj_vars for d in range(spec.n_days)
                 if f"{v}_d{d}" in traj.columns]
        frame = frame.merge(traj[["stay_id"] + tcols], on="stay_id", how="left")

    y = frame[spec.outcome].to_numpy().astype(np.float32)
    x_num = frame[list(spec.num_vars)].to_numpy().astype(np.float32)
    x_bin = frame[list(spec.bin_vars)].to_numpy().astype(np.float32)
    x_traj, x_mask = _trajectory_arrays(frame, spec)
    return Dataset(frame=frame, y=y, x_num=x_num, x_bin=x_bin,
                   x_traj=x_traj, x_mask=x_mask, spec=spec)


def standardize(train: np.ndarray, *others: np.ndarray):
    """Z-score using training-set statistics only.

    Returns ``(list_of_transformed_arrays, mean, sd)``.
    """
    mu = train.mean(0).astype(np.float32)
    sd = train.std(0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return [(a - mu) / sd for a in (train,) + others], mu, sd
