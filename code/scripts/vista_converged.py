"""Load the archived VISTA experiment module with a fully converged logistic classifier.

The principal results use Linux-fitted logistic classifiers with newton-cholesky and tol=1e-10.
This module treats a convergence warning as an error. Every other constructor argument,
the class order, development-only standardization, and the remaining functions are unchanged.
The archived script retains the earlier L-BFGS setting at tol=1e-4.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

SOLVER = {"solver": "newton-cholesky", "tol": 1e-10, "max_iter": 1000}


def load(path: Path, converged: bool = True):
    spec = importlib.util.spec_from_file_location("original_common_utility", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if converged:
        def fit_logistic(y_dev, labels_dev, alpha):
            mean = y_dev.mean(axis=0)
            scale = y_dev.std(axis=0, ddof=0).copy()
            scale[scale == 0] = 1.0
            model = LogisticRegression(
                C=1.0 / float(alpha), class_weight="balanced", dual=False, fit_intercept=True,
                intercept_scaling=1, l1_ratio=0.0, n_jobs=None, penalty="deprecated",
                random_state=20260812, verbose=0, warm_start=False, **SOLVER)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit((y_dev - mean) / scale, mod.CLASSES[labels_dev])
            if not np.array_equal(model.classes_, mod.CLASSES):
                raise RuntimeError(f"Logistic class order mismatch: {model.classes_}")
            return mod.Decoder(model=model, mean=mean, scale=scale, kind="logistic")
        mod.fit_logistic = fit_logistic
    return mod
