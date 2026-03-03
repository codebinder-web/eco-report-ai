"""Forecast evaluation metrics for EcoReport AI.

All metrics handle edge cases (zeros, NaN, Inf) safely.
MAPE uses a safe denominator to prevent division by zero when actual = 0.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPSILON = 1e-8  # Safe denominator for MAPE


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAE as a float. Returns NaN if inputs are empty or all NaN.
    """
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(math.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Percentage Error with safe division.

    Denominator: max(|y_true|, ε) where ε = 1e-8 to avoid division by zero.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE in percentage points (0–100+). Returns NaN if empty.
    """
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    denominators = np.maximum(np.abs(yt), _EPSILON)
    return float(np.mean(np.abs(yt - yp) / denominators) * 100)


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE in one call.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict with keys "mae", "rmse", "mape".
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into mean ± std summary.

    Args:
        fold_metrics: List of metric dicts, one per CV fold.

    Returns:
        Dict with keys "mae", "rmse", "mape", each containing "mean" and "std".
    """
    result: dict[str, dict[str, float]] = {}
    for key in ("mae", "rmse", "mape"):
        values = [m[key] for m in fold_metrics if not math.isnan(m.get(key, float("nan")))]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            result[key] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return result


def _clean(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and remove NaN-paired observations.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Tuple of cleaned (y_true, y_pred) numpy arrays.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    # Align lengths
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    # Remove positions where either is NaN or Inf
    valid = np.isfinite(yt) & np.isfinite(yp)
    return yt[valid], yp[valid]
