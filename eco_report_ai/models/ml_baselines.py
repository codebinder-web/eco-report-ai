"""Machine learning baseline models for EcoReport AI.

Implements GradientBoostingRegressor and RandomForestRegressor with
hyperparameter search on training folds only (preventing leakage).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


class GradientBoostingModel:
    """Gradient Boosting or Random Forest regressor with randomized hyperparameter search.

    Hyperparameter search is performed exclusively on training data using
    TimeSeriesSplit to maintain temporal ordering.

    Args:
        variant: "gradient_boosting" or "random_forest".
        n_iter_search: Number of RandomizedSearchCV iterations.
        cv_folds: Number of CV folds within hyperparameter search.
        seed: Random seed for reproducibility.
        name: Human-readable model name.
    """

    def __init__(
        self,
        variant: Literal["gradient_boosting", "random_forest"] = "gradient_boosting",
        n_iter_search: int = 20,
        cv_folds: int = 3,
        seed: int = 42,
        name: str = "GradientBoosting",
    ) -> None:
        self.variant = variant
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.seed = seed
        self.name = name
        self._model: Any = None
        self._feature_names: list[str] = []

    def _build_search_space(self) -> dict[str, Any]:
        """Define the hyperparameter search space."""
        if self.variant == "gradient_boosting":
            return {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "min_samples_leaf": [1, 2, 4, 8],
            }
        else:
            return {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": [0.5, 0.7, "sqrt"],
            }

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GradientBoostingModel":
        """Fit the model with hyperparameter search.

        Drops NaN rows before fitting. Feature names are stored for
        feature importance extraction.

        Args:
            X_train: Numeric feature DataFrame.
            y_train: Target Series.

        Returns:
            Self.
        """
        # Drop NaN
        combined = pd.concat([X_train, y_train], axis=1).dropna()
        y_col = y_train.name or "__target__"
        combined.columns = list(X_train.columns) + [y_col]
        X_clean = combined[list(X_train.columns)]
        y_clean = combined[y_col]

        self._feature_names = list(X_clean.columns)

        base_estimator: Any
        if self.variant == "gradient_boosting":
            base_estimator = GradientBoostingRegressor(random_state=self.seed)
        else:
            base_estimator = RandomForestRegressor(random_state=self.seed, n_jobs=-1)

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=self._build_search_space(),
            n_iter=min(self.n_iter_search, 10) if len(X_clean) < 100 else self.n_iter_search,
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=self.seed,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_clean.values, y_clean.values)
        self._model = search.best_estimator_

        logger.info(
            "%s fitted: best_params=%s, best_neg_RMSE=%.4f",
            self.name,
            search.best_params_,
            (-search.best_score_) ** 0.5,
        )
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X_test: Feature DataFrame with same columns as training data.

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        X_aligned = X_test[self._feature_names].ffill().fillna(0)
        return self._model.predict(X_aligned.values)

    def feature_importances(self) -> pd.Series:
        """Return feature importances from the fitted model.

        Returns:
            Series with feature names as index, sorted descending.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importances().")
        importances = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
            name="importance",
        )
        return importances.sort_values(ascending=False)
