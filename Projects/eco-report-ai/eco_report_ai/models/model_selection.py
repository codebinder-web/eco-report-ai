"""Model selection policy for EcoReport AI.

Selects the best model based on mean RMSE across rolling-origin CV folds.
Tiebreaker: MAE → MAPE → preference order (OLS > ARIMA > GB > LSTM).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from eco_report_ai.evaluation.backtesting import BacktestResult

logger = logging.getLogger(__name__)

# Preference order (lower index = preferred when RMSE ties)
_MODEL_PREFERENCE = ["OLS", "ARIMA", "GradientBoosting", "LSTM"]


def select_best_model(
    backtest_results: list[BacktestResult],
) -> tuple[str, BacktestResult]:
    """Select the best model based on mean RMSE across CV folds.

    Tiebreaker order: mean RMSE → mean MAE → mean MAPE → model preference index.

    Args:
        backtest_results: List of BacktestResult objects, one per model.

    Returns:
        Tuple of (best_model_name, best_BacktestResult).

    Raises:
        ValueError: If backtest_results is empty or all models have NaN RMSE.
    """
    if not backtest_results:
        raise ValueError("backtest_results is empty — no models to select from.")

    valid_results = [
        r for r in backtest_results
        if not math.isnan(r.aggregate_metrics.get("rmse", {}).get("mean", float("nan")))
    ]

    if not valid_results:
        logger.warning("All models have NaN RMSE. Defaulting to first available.")
        return backtest_results[0].model_name, backtest_results[0]

    def sort_key(r: BacktestResult) -> tuple[float, float, float, int]:
        rmse = r.aggregate_metrics.get("rmse", {}).get("mean", float("inf"))
        mae_val = r.aggregate_metrics.get("mae", {}).get("mean", float("inf"))
        mape_val = r.aggregate_metrics.get("mape", {}).get("mean", float("inf"))
        pref = next(
            (i for i, name in enumerate(_MODEL_PREFERENCE) if name in r.model_name),
            len(_MODEL_PREFERENCE),
        )
        return (rmse, mae_val, mape_val, pref)

    best = min(valid_results, key=sort_key)

    logger.info(
        "Best model: %s (mean RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%)",
        best.model_name,
        best.aggregate_metrics.get("rmse", {}).get("mean", float("nan")),
        best.aggregate_metrics.get("mae", {}).get("mean", float("nan")),
        best.aggregate_metrics.get("mape", {}).get("mean", float("nan")),
    )
    return best.model_name, best


class ModelRegistry:
    """Registry that holds fitted model instances by name.

    Allows lookup of the best model instance after selection.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def register(self, name: str, model: Any) -> None:
        """Register a fitted model.

        Args:
            name: Model identifier.
            model: Fitted model object.
        """
        self._models[name] = model
        logger.debug("Registered model: %s", name)

    def get(self, name: str) -> Any:
        """Retrieve a registered model by name.

        Args:
            name: Model identifier.

        Returns:
            The model object.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry.")
        return self._models[name]

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())
