# -*- coding: utf-8 -*-
"""Transformer risk models for acute liver failure on MIMIC-IV.

Public API
----------
``load_dataset``      assemble the modelling arrays from prepared CSVs
``run_seeds``         cross-validated training over several random seeds
``evaluate_models``   the common statistical evaluation framework
"""
from .data import DataSpec, Dataset, load_dataset, standardize
from .evaluate import evaluate_models, dca_summary, reproducibility, to_probability
from .models import FTTransformer, HybridTransformer, build_model
from .train import cross_validate, run_seeds, train_fold

__version__ = "0.1.0"

__all__ = [
    "DataSpec", "Dataset", "load_dataset", "standardize",
    "FTTransformer", "HybridTransformer", "build_model",
    "train_fold", "cross_validate", "run_seeds",
    "evaluate_models", "dca_summary", "reproducibility", "to_probability",
    "__version__",
]
