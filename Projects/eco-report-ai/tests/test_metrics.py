"""Tests for the evaluation metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eco_report_ai.evaluation.metrics import (
    _EPSILON,
    aggregate_fold_metrics,
    compute_all_metrics,
    mae,
    mape,
    rmse,
)


class TestMAE:
    """Tests for the mae() function."""

    def test_perfect_prediction(self) -> None:
        """MAE should be 0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """MAE of [1,2,3] vs [2,3,4] = 1.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_empty_returns_nan(self) -> None:
        """Empty arrays should return NaN."""
        result = mae(np.array([]), np.array([]))
        assert math.isnan(result)

    def test_all_nan_returns_nan(self) -> None:
        """All-NaN input should return NaN."""
        y = np.array([float("nan")] * 3)
        result = mae(y, y)
        assert math.isnan(result)

    def test_handles_negative_values(self) -> None:
        """MAE should work with negative values."""
        y_true = np.array([-2.0, -1.0, 0.0])
        y_pred = np.array([-1.0, 0.0, 1.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)


class TestRMSE:
    """Tests for the rmse() function."""

    def test_perfect_prediction(self) -> None:
        assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """RMSE of errors [1, 1, 1] = 1.0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_penalizes_large_errors_more_than_mae(self) -> None:
        """RMSE should be >= MAE for the same errors."""
        y_true = np.array([1.0, 1.0, 10.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_empty_returns_nan(self) -> None:
        assert math.isnan(rmse(np.array([]), np.array([])))


class TestMAPE:
    """Tests for the mape() function."""

    def test_perfect_prediction(self) -> None:
        assert mape(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """MAPE: y_true=[2,4], y_pred=[1,3] → mean(|2-1|/2, |4-3|/4)*100 = mean(50,25) = 37.5."""
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([1.0, 3.0])
        assert mape(y_true, y_pred) == pytest.approx(37.5)

    def test_zero_denominator_handled_safely(self) -> None:
        """MAPE should not raise on actual=0 (uses epsilon denominator)."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 1.5])
        result = mape(y_true, y_pred)
        assert math.isfinite(result)

    def test_empty_returns_nan(self) -> None:
        assert math.isnan(mape(np.array([]), np.array([])))

    def test_epsilon_value_is_small(self) -> None:
        """_EPSILON must be a very small positive number."""
        assert _EPSILON > 0
        assert _EPSILON < 1e-5


class TestComputeAllMetrics:
    """Tests for compute_all_metrics()."""

    def test_returns_all_keys(self) -> None:
        """Dict should contain mae, rmse, mape."""
        result = compute_all_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert set(result.keys()) == {"mae", "rmse", "mape"}

    def test_values_are_non_negative(self) -> None:
        """All metrics should be non-negative for well-formed inputs."""
        result = compute_all_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]))
        for k, v in result.items():
            assert v >= 0, f"{k} is negative: {v}"


class TestAggregateFoldMetrics:
    """Tests for aggregate_fold_metrics()."""

    def test_mean_and_std_computed(self) -> None:
        """Should compute mean and std across folds."""
        fold_metrics = [
            {"mae": 1.0, "rmse": 2.0, "mape": 10.0},
            {"mae": 3.0, "rmse": 4.0, "mape": 20.0},
        ]
        result = aggregate_fold_metrics(fold_metrics)
        assert result["mae"]["mean"] == pytest.approx(2.0)
        assert result["mae"]["std"] == pytest.approx(1.0)
        assert result["rmse"]["mean"] == pytest.approx(3.0)

    def test_empty_list_returns_nan(self) -> None:
        """Empty fold list should return NaN for all metrics."""
        result = aggregate_fold_metrics([])
        for key in ("mae", "rmse", "mape"):
            assert math.isnan(result[key]["mean"])

    def test_single_fold(self) -> None:
        """Single fold should return std=0."""
        fold_metrics = [{"mae": 1.5, "rmse": 2.5, "mape": 5.0}]
        result = aggregate_fold_metrics(fold_metrics)
        assert result["mae"]["mean"] == pytest.approx(1.5)
        assert result["mae"]["std"] == pytest.approx(0.0)
