"""Econometric models: OLS with HAC robust standard errors and ARIMA/SARIMAX.

Design rationale (from design_decisions.md ADR-005, ADR-007):
- OLS provides interpretability and a classical econometric baseline.
- HAC standard errors correct for autocorrelation and heteroskedasticity
  in macroeconomic time series (Newey-West, 1987).
- ARIMA captures univariate time-series dynamics; order selected by AIC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── OLS Model ─────────────────────────────────────────────────────────────────


@dataclass
class OLSSummary:
    """Summary of OLS estimation results.

    Attributes:
        formula: The regression formula used.
        coefficients: Dict mapping variable name to coefficient estimate.
        std_errors: HAC standard errors.
        p_values: Two-sided p-values.
        t_stats: t-statistics.
        r_squared: Coefficient of determination.
        adj_r_squared: Adjusted R².
        nobs: Number of observations.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        durbin_watson: Durbin-Watson statistic for residual autocorrelation (2 = no AC).
        residuals: In-sample residuals as a Series.
        raw_result: The underlying statsmodels RegressionResults object.
    """

    formula: str
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    t_stats: dict[str, float]
    r_squared: float
    adj_r_squared: float
    nobs: int
    aic: float
    bic: float
    durbin_watson: float
    residuals: pd.Series
    raw_result: Any


class OLSModel:
    """OLS regression with optional HAC (Newey-West) robust standard errors.

    Args:
        formula: Patsy/statsmodels formula string (e.g., "cpi_yoy ~ fedfunds + unrate").
        hac_lags: Number of Newey-West lags for HAC correction. 0 = no HAC.
        name: Human-readable model name.
    """

    def __init__(
        self,
        formula: str = "cpi_yoy ~ fedfunds + unrate + cpi_yoy_lag_1 + cpi_yoy_lag_3",
        hac_lags: int = 6,
        name: str = "OLS",
    ) -> None:
        self.formula = formula
        self.hac_lags = hac_lags
        self.name = name
        self._result: Any = None
        self._train_data: Optional[pd.DataFrame] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "OLSModel":
        """Fit the OLS model on training data.

        Args:
            X_train: Feature DataFrame (must contain columns referenced in formula).
            y_train: Target Series.

        Returns:
            Self for method chaining.
        """
        import statsmodels.formula.api as smf

        # Merge features and target for formula interface
        train_df = X_train.copy()
        train_df[y_train.name or "target"] = y_train
        train_df = train_df.dropna()

        cov_kwargs: dict = {}
        if self.hac_lags > 0:
            cov_kwargs = {"cov_type": "HAC", "cov_kwds": {"maxlags": self.hac_lags}}

        try:
            model = smf.ols(self.formula, data=train_df)
            if cov_kwargs:
                self._result = model.fit(**cov_kwargs)
            else:
                self._result = model.fit()
            self._train_data = train_df
            logger.info(
                "OLS fitted: R²=%.4f, adj-R²=%.4f, n=%d, HAC=%s",
                self._result.rsquared,
                self._result.rsquared_adj,
                int(self._result.nobs),
                f"yes (lags={self.hac_lags})" if self.hac_lags > 0 else "no",
            )
        except Exception as exc:
            logger.error("OLS fit failed: %s", exc)
            raise
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate OLS predictions.

        Args:
            X_test: Feature DataFrame with same columns as training data.

        Returns:
            Array of predicted values.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        preds = self._result.predict(X_test)
        return np.asarray(preds, dtype=float)

    def get_summary(self) -> OLSSummary:
        """Extract a structured summary of the fitted OLS model.

        Returns:
            OLSSummary dataclass with all key statistics.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before get_summary().")

        from statsmodels.stats.stattools import durbin_watson as _dw

        r = self._result
        params = r.params
        dw = float(_dw(r.resid))
        return OLSSummary(
            formula=self.formula,
            coefficients=params.to_dict(),
            std_errors=r.bse.to_dict(),
            p_values=r.pvalues.to_dict(),
            t_stats=r.tvalues.to_dict(),
            r_squared=float(r.rsquared),
            adj_r_squared=float(r.rsquared_adj),
            nobs=int(r.nobs),
            aic=float(r.aic),
            bic=float(r.bic),
            durbin_watson=dw,
            residuals=r.resid,
            raw_result=r,
        )


# ── ARIMA Model ───────────────────────────────────────────────────────────────


@dataclass
class ARIMAForecast:
    """Forecast output from the ARIMA model.

    Attributes:
        order: (p, d, q) order used.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        forecast: Point forecast values as a Series.
        conf_int_lower: Lower 95% confidence interval.
        conf_int_upper: Upper 95% confidence interval.
    """

    order: tuple[int, int, int]
    aic: float
    bic: float
    forecast: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series


class ARIMAModel:
    """ARIMA / SARIMAX model with automatic order selection via AIC grid search.

    Order (p, d, q) is selected by grid search. Differencing order d is
    determined via the ADF test (if d_auto=True) or fixed from config.

    Args:
        max_p: Maximum AR order to search.
        max_q: Maximum MA order to search.
        seasonal: Whether to include a seasonal component.
        seasonal_period: Seasonal periodicity (12 for monthly data).
        name: Human-readable model name.
    """

    def __init__(
        self,
        max_p: int = 3,
        max_q: int = 2,
        seasonal: bool = False,
        seasonal_period: int = 12,
        name: str = "ARIMA",
    ) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.name = name
        self._result: Any = None
        self._best_order: tuple[int, int, int] = (1, 1, 1)
        self._y_train: Optional[pd.Series] = None

    def _determine_d(self, y: pd.Series) -> int:
        """Determine differencing order using the ADF test.

        Args:
            y: Target time series.

        Returns:
            0 if series is stationary, 1 otherwise.
        """
        from statsmodels.tsa.stattools import adfuller

        try:
            adf_result = adfuller(y.dropna(), autolag="AIC")
            p_value = adf_result[1]
            d = 0 if p_value < 0.05 else 1
            logger.info("ADF test p-value=%.4f -> d=%d", p_value, d)
            return d
        except Exception:
            return 1

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ARIMAModel":
        """Fit ARIMA by grid-searching (p, d, q) to minimise AIC.

        Args:
            X_train: Feature DataFrame (not used by ARIMA; kept for API consistency).
            y_train: Target time series.

        Returns:
            Self.
        """
        from statsmodels.tsa.arima.model import ARIMA

        self._y_train = y_train.dropna()
        d = self._determine_d(self._y_train)

        best_aic = float("inf")
        best_result = None
        best_order = (1, d, 1)

        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(self._y_train, order=(p, d, q))
                    res = m.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_result = res
                        best_order = (p, d, q)
                except Exception:
                    continue

        if best_result is None:
            logger.warning("ARIMA grid search failed for all orders. Fitting (1,%d,1).", d)
            m = ARIMA(self._y_train, order=(1, d, 1))
            best_result = m.fit()
            best_order = (1, d, 1)

        self._result = best_result
        self._best_order = best_order
        logger.info(
            "ARIMA(%d,%d,%d) selected: AIC=%.2f BIC=%.2f",
            *best_order,
            best_result.aic,
            best_result.bic,
        )
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate ARIMA predictions for the test window.

        Args:
            X_test: Feature DataFrame (used only to determine forecast length).

        Returns:
            Array of h-step-ahead point forecasts.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        h = len(X_test)
        fc = self._result.forecast(steps=h)
        return np.asarray(fc, dtype=float)

    def forecast(self, horizon: int) -> ARIMAForecast:
        """Generate a multi-step-ahead forecast with confidence intervals.

        Args:
            horizon: Number of steps ahead to forecast.

        Returns:
            ARIMAForecast with point forecast and 95% CIs.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before forecast().")

        fc_result = self._result.get_forecast(steps=horizon)
        fc_mean = fc_result.predicted_mean
        conf = fc_result.conf_int(alpha=0.05)

        # Build future date index
        if self._y_train is not None and isinstance(self._y_train.index, pd.DatetimeIndex):
            last_date = self._y_train.index[-1]
            future_idx = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS",
            )
            fc_mean.index = future_idx
            conf.index = future_idx

        return ARIMAForecast(
            order=self._best_order,
            aic=float(self._result.aic),
            bic=float(self._result.bic),
            forecast=fc_mean,
            conf_int_lower=conf.iloc[:, 0],
            conf_int_upper=conf.iloc[:, 1],
        )
