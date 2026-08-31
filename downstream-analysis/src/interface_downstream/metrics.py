"""Regression metrics without scikit-learn dependencies."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    target_std = float(np.std(y_true, ddof=0))
    prediction_std = float(np.std(y_pred, ddof=0))
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual**2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae_over_target_std": mae / target_std if target_std > 0 else math.nan,
        "pearson": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if target_std > 0 and prediction_std > 0 else math.nan
        ),
        "target_test_mean": float(np.mean(y_true)),
        "target_test_std": target_std,
    }

