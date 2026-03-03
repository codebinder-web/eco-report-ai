"""Rolling-origin cross-validation for time-series models.

Implements the leakage-free evaluation strategy described in design_decisions.md.
Training windows always precede test windows — no shuffling, ever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from eco_report_ai.evaluation.metrics import aggregate_fold_metrics, compute_all_metrics

logger = logging.getLogger(__name__)


class ForecastModel(Protocol):
    """Protocol that all EcoReport AI models must satisfy for backtesting."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model on training data."""
        ...

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions for test features."""
        ...

    @property
    def name(self) -> str:
        """Return the model name."""
        ...


@dataclass
class FoldResult:
    """Result from a single rolling-origin fold.

    Attributes:
        fold_idx: Zero-based fold index.
        train_start: First training date.
        train_end: Last training date.
        test_start: First test date.
        test_end: Last test date.
        y_true: Actual target values.
        y_pred: Predicted target values.
        metrics: Dict of MAE, RMSE, MAPE for this fold.
    """

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregated result across all folds for one model.

    Attributes:
        model_name: Name of the evaluated model.
        fold_results: Per-fold results.
        aggregate_metrics: Mean ± std of each metric across folds.
    """

    model_name: str
    fold_results: list[FoldResult]
    aggregate_metrics: dict[str, dict[str, float]]


class RollingOriginCV:
    """Rolling-origin (expanding window) cross-validator for time-series.

    Args:
        n_folds: Number of evaluation folds.
        forecast_horizon: Number of steps to forecast per fold.
    """

    def __init__(self, n_folds: int = 5, forecast_horizon: int = 6) -> None:
        self.n_folds = n_folds
        self.forecast_horizon = forecast_horizon

    def split(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate (train_idx, test_idx) pairs for rolling-origin CV.

        The test windows are carved from the end of the series.
        Each fold's training window expands by one forecast horizon.

        Args:
            df: DataFrame with monotonic DatetimeIndex.

        Returns:
            List of (train_index, test_index) tuples.

        Raises:
            ValueError: If the dataset is too short for the requested splits.
        """
        n = len(df)
        required = self.n_folds * self.forecast_horizon + self.forecast_horizon
        if n < required:
            raise ValueError(
                f"Dataset too short ({n} rows) for {self.n_folds} folds × "
                f"{self.forecast_horizon} horizon = {required} minimum rows."
            )

        splits = []
        for fold in range(self.n_folds):
            # test window is carved from the tail
            test_end_idx = n - fold * self.forecast_horizon
            test_start_idx = test_end_idx - self.forecast_horizon
            train_end_idx = test_start_idx

            if train_end_idx < self.forecast_horizon:
                logger.warning("Fold %d: training window too small, skipping.", fold)
                continue

            train_idx = df.index[:train_end_idx]
            test_idx = df.index[test_start_idx:test_end_idx]
            splits.append((train_idx, test_idx))

        # Reverse so fold 0 = earliest
        splits = list(reversed(splits))
        logger.info(
            "Generated %d rolling-origin folds (horizon=%d).",
            len(splits),
            self.forecast_horizon,
        )
        return splits

    def run_backtest(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str] | None = None,
    ) -> BacktestResult:
        """Run rolling-origin backtesting for a single model.

        Args:
            model: Model implementing fit(X, y) and predict(X) interface.
            X: Full feature DataFrame with DatetimeIndex.
            y: Target Series with same index.
            feature_cols: Subset of X columns to use; None = use all.

        Returns:
            BacktestResult with per-fold and aggregate metrics.
        """
        model_name = getattr(model, "name", type(model).__name__)
        logger.info("Backtesting %s ...", model_name)

        if feature_cols:
            X = X[feature_cols]

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].copy()
        y = y.loc[common_idx].copy()

        splits = self.split(X)
        fold_results: list[FoldResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.loc[train_idx].dropna()
            y_train = y.loc[X_train.index]
            X_test = X.loc[test_idx].dropna()
            y_test = y.loc[X_test.index]

            if len(X_train) < 10 or len(X_test) == 0:
                logger.warning("Fold %d: insufficient data, skipping.", fold_idx)
                continue

            try:
                model.fit(X_train, y_train)
                y_pred = np.asarray(model.predict(X_test), dtype=float)
            except Exception as exc:
                logger.error("Fold %d: model failed — %s", fold_idx, exc)
                continue

            metrics = compute_all_metrics(y_test.values, y_pred)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                y_true=y_test.values,
                y_pred=y_pred,
                metrics=metrics,
            )
            fold_results.append(fold_result)

            logger.info(
                "Fold %d — MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
                fold_idx,
                metrics["mae"],
                metrics["rmse"],
                metrics["mape"],
            )

        agg = aggregate_fold_metrics([fr.metrics for fr in fold_results])
        logger.info(
            "%s backtest complete — mean RMSE=%.4f ± %.4f",
            model_name,
            agg["rmse"]["mean"],
            agg["rmse"]["std"],
        )

        return BacktestResult(
            model_name=model_name,
            fold_results=fold_results,
            aggregate_metrics=agg,
        )
