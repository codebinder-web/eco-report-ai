"""Pipeline orchestrator for EcoReport AI.

Implements the full end-to-end pipeline:
  Data → Features → Models → Backtesting → Selection → Report

Architecture follows the course principle:
  Plan (config) → Compute (deterministic) → Generate (constrained) → Validate → Render
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from eco_report_ai.config import PipelineConfig, load_config
from eco_report_ai.data.loaders import load_macro_data
from eco_report_ai.evaluation.backtesting import BacktestResult, RollingOriginCV
from eco_report_ai.features.build_features import build_features
from eco_report_ai.models.econometrics import ARIMAModel, OLSModel
from eco_report_ai.models.lstm import LSTMForecaster
from eco_report_ai.models.ml_baselines import GradientBoostingModel
from eco_report_ai.models.model_selection import ModelRegistry, select_best_model
from eco_report_ai.reporting.charts import (
    plot_forecast,
    plot_historical_series,
    plot_model_comparison,
)
from eco_report_ai.reporting.nlg import NarrativeGenerator
from eco_report_ai.reporting.report_writer import ReportWriter
from eco_report_ai.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def _set_global_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _recursive_forecast(
    model_obj: Any,
    X_last: pd.Series,
    y_history: pd.Series,
    horizon: int,
    future_dates: pd.DatetimeIndex,
    target_col: str = "cpi_yoy",
) -> pd.Series:
    """Recursive multi-step forecast that updates target-lag features after each step.

    After predicting step k the lag columns for *target_col* are updated with the
    just-predicted value before predicting step k+1.  All exogenous features
    (UNRATE, FEDFUNDS and their lags / rolling statistics) are held constant at
    their last observed values — the standard assumption when no separate exogenous
    forecasts are available.

    Args:
        model_obj: Fitted model exposing ``predict(X: DataFrame) -> np.ndarray``.
        X_last: Last known feature row as a named Series (index = column names).
        y_history: Full historical target series used to seed the lag buffer.
        horizon: Number of steps ahead to forecast.
        future_dates: DatetimeIndex of length ``horizon`` for the returned Series.
        target_col: Name of the target variable (identifies its lag columns).

    Returns:
        Series of length ``horizon`` with recursive predictions.
    """
    lag_pattern = re.compile(rf"^{re.escape(target_col)}_lag_(\d+)$")

    # Collect all lag columns for the target that exist in the feature set
    all_lag_cols = [c for c in X_last.index if lag_pattern.match(c)]
    max_lag_needed = (
        max(int(lag_pattern.match(c).group(1)) for c in all_lag_cols)
        if all_lag_cols
        else 0
    )

    # Seed the rolling buffer with enough historical y values to cover all lags
    y_buf: list[float] = (
        list(y_history.values[-max_lag_needed:]) if max_lag_needed > 0 else []
    )

    current_row = X_last.copy()
    predictions: list[float] = []

    for _step in range(horizon):
        X_step = pd.DataFrame([current_row.values], columns=list(current_row.index))
        try:
            pred = float(model_obj.predict(X_step)[0])
        except Exception as exc:
            logger.warning("Recursive forecast step %d failed (%s); using last value.", _step, exc)
            pred = float(y_buf[-1]) if y_buf else float(y_history.iloc[-1])

        predictions.append(pred)
        y_buf.append(pred)

        # Update target lag columns for the next step using the rolling buffer
        for col in all_lag_cols:
            m = lag_pattern.match(col)
            if m:
                lag_n = int(m.group(1))
                if lag_n <= len(y_buf):
                    current_row[col] = y_buf[-lag_n]
        # Exogenous features are intentionally held constant (last-observed assumption).

    return pd.Series(predictions, index=future_dates, name="forecast")


class EcoReportPipeline:
    """Full end-to-end pipeline orchestrator.

    Args:
        config: Validated PipelineConfig instance.
        config_path: Path to the config file (for reproducibility logging).
    """

    def __init__(
        self,
        config: PipelineConfig,
        config_path: str = "config.yaml",
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.run_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()
        _set_global_seeds(config.seed)
        logger.info("Pipeline initialized (run_id=%s, seed=%d)", self.run_id, config.seed)

    def run(self, dry_run: bool = False, override_source: Optional[str] = None) -> dict[str, Any]:
        """Execute the full pipeline.

        Args:
            dry_run: If True, log the execution plan without running it.
            override_source: Override config.data.source ("fred"|"sample"|"auto").

        Returns:
            Evidence store dict containing all computed results.
        """
        logger.info("=" * 60)
        logger.info("EcoReport AI Pipeline — run_id=%s", self.run_id)
        logger.info("=" * 60)

        if dry_run:
            logger.info("[DRY RUN] Pipeline would execute: Load → Features → Models → Report")
            return {"run_id": self.run_id, "dry_run": True}

        # ── Step 1: Load data ─────────────────────────────────────────────────
        logger.info("Step 1/6: Loading data ...")
        df_raw, source_used = load_macro_data(
            self.config.data, override_source=override_source
        )
        logger.info(
            "Data loaded: %d rows (%s to %s), source=%s",
            len(df_raw),
            df_raw.index.min().strftime("%Y-%m"),
            df_raw.index.max().strftime("%Y-%m"),
            source_used,
        )

        # ── Step 2: Feature engineering ───────────────────────────────────────
        logger.info("Step 2/6: Building features ...")
        df_features = build_features(df_raw, self.config.features)
        target_col = self.config.features.target_column
        y = df_features[target_col].dropna()
        X = df_features.drop(columns=[target_col]).loc[y.index]

        # Select numeric feature columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]

        logger.info("Features: %d rows × %d columns", len(X), len(X.columns))

        # ── Step 3: Fit & backtest models ─────────────────────────────────────
        logger.info("Step 3/6: Fitting and backtesting models ...")
        backtest_results, registry = self._fit_and_backtest(X, y, df_features)

        # ── Step 4: Select best model ─────────────────────────────────────────
        logger.info("Step 4/6: Selecting best model ...")
        best_name, best_result = select_best_model(backtest_results)
        best_model_obj = registry.get(best_name)

        # ── Step 5: Generate final forecast ───────────────────────────────────
        logger.info("Step 5/6: Generating final forecast ...")
        forecast_series, conf_lower, conf_upper = self._generate_final_forecast(
            best_model_obj, best_name, X, y, df_features
        )

        # ── Step 6: Build evidence store & write report ───────────────────────
        logger.info("Step 6/6: Building evidence store and writing report ...")
        evidence = self._build_evidence_store(
            df_raw=df_raw,
            df_features=df_features,
            y=y,
            source_used=source_used,
            backtest_results=backtest_results,
            registry=registry,
            best_name=best_name,
            best_result=best_result,
            best_model_obj=best_model_obj,
            forecast_series=forecast_series,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
        )

        # ── Charts ────────────────────────────────────────────────────────────
        ensure_dir(self.config.reporting.figures_dir)
        _plot_all(
            df_raw=df_raw,
            y=y,
            forecast_series=forecast_series,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
            best_name=best_name,
            model_metrics=evidence.get("model_metrics", {}),
            config=self.config,
        )

        # ── Write report ──────────────────────────────────────────────────────
        nlg = NarrativeGenerator(
            llm_model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            enabled_if_key=self.config.llm.enabled_if_key,
        )
        writer = ReportWriter(
            output_dir=self.config.reporting.output_dir,
            figures_dir=self.config.reporting.figures_dir,
            report_filename=self.config.reporting.report_filename,
            nlg=nlg,
        )
        md_path, json_path = writer.write_all(evidence)

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info("  Report : %s", md_path)
        logger.info("  JSON   : %s", json_path)
        logger.info("=" * 60)

        return evidence

    def _fit_and_backtest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_features: pd.DataFrame,
    ) -> tuple[list[BacktestResult], ModelRegistry]:
        """Fit all enabled models and run rolling-origin backtesting.

        Args:
            X: Feature matrix.
            y: Target series.
            df_features: Full feature DataFrame (for ARIMA which is univariate).

        Returns:
            Tuple of (backtest_results list, ModelRegistry).
        """
        cv = RollingOriginCV(
            n_folds=self.config.evaluation.n_folds,
            forecast_horizon=self.config.evaluation.forecast_horizon,
        )
        registry = ModelRegistry()
        results: list[BacktestResult] = []
        cfg = self.config.models

        # ── OLS ───────────────────────────────────────────────────────────────
        if cfg.ols.enabled:
            try:
                ols = OLSModel(
                    formula=cfg.ols.formula,
                    hac_lags=cfg.ols.hac_lags,
                    name="OLS",
                )
                ols_result = cv.run_backtest(ols, X, y)
                results.append(ols_result)
                # Fit on full data for final use
                ols.fit(X, y)
                registry.register("OLS", ols)
            except Exception as exc:
                logger.error("OLS failed: %s", exc)

        # ── ARIMA ─────────────────────────────────────────────────────────────
        if cfg.arima.enabled:
            try:
                arima = ARIMAModel(
                    max_p=cfg.arima.max_p,
                    max_q=cfg.arima.max_q,
                    seasonal=cfg.arima.seasonal,
                    seasonal_period=cfg.arima.seasonal_period,
                    name="ARIMA",
                )
                arima_result = cv.run_backtest(arima, X, y)
                results.append(arima_result)
                arima.fit(X, y)
                registry.register("ARIMA", arima)
            except Exception as exc:
                logger.error("ARIMA failed: %s", exc)

        # ── Gradient Boosting ─────────────────────────────────────────────────
        if cfg.gradient_boosting.enabled:
            try:
                gb = GradientBoostingModel(
                    variant=cfg.gradient_boosting.variant,
                    n_iter_search=cfg.gradient_boosting.n_iter_search,
                    cv_folds=cfg.gradient_boosting.cv_folds,
                    seed=self.config.seed,
                    name="GradientBoosting",
                )
                gb_result = cv.run_backtest(gb, X, y)
                results.append(gb_result)
                gb.fit(X, y)
                registry.register("GradientBoosting", gb)
            except Exception as exc:
                logger.error("GradientBoosting failed: %s", exc)

        # ── LSTM ──────────────────────────────────────────────────────────────
        if cfg.lstm.enabled:
            try:
                lstm = LSTMForecaster(
                    lookback=cfg.lstm.lookback,
                    hidden_size=cfg.lstm.hidden_size,
                    num_layers=cfg.lstm.num_layers,
                    dropout=cfg.lstm.dropout,
                    batch_size=cfg.lstm.batch_size,
                    max_epochs=cfg.lstm.max_epochs,
                    patience=cfg.lstm.patience,
                    learning_rate=cfg.lstm.learning_rate,
                    val_split=cfg.lstm.val_split,
                    seed=self.config.seed,
                    name="LSTM",
                )
                lstm_result = cv.run_backtest(lstm, X, y)
                results.append(lstm_result)
                lstm.fit(X, y)
                registry.register("LSTM", lstm)
            except Exception as exc:
                logger.error("LSTM failed: %s", exc)

        if not results:
            raise RuntimeError("All models failed during backtesting. Check logs for details.")

        return results, registry

    def _generate_final_forecast(
        self,
        best_model_obj: Any,
        best_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        df_features: pd.DataFrame,
    ) -> tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
        """Generate the final out-of-sample forecast from the best model.

        Args:
            best_model_obj: Fitted model object.
            best_name: Model name string.
            X: Full feature matrix.
            y: Full target series.
            df_features: Full feature DataFrame.

        Returns:
            Tuple of (forecast, conf_lower, conf_upper). CIs are None if unavailable.
        """
        horizon = self.config.evaluation.final_forecast_horizon
        last_date = y.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        # For ARIMA, use the model's own forecast method for CIs
        if best_name == "ARIMA" and hasattr(best_model_obj, "forecast"):
            try:
                fc_result = best_model_obj.forecast(horizon)
                return fc_result.forecast, fc_result.conf_int_lower, fc_result.conf_int_upper
            except Exception as exc:
                logger.warning("ARIMA CI forecast failed (%s); using point forecast.", exc)

        # Recursive multi-step forecast: update target-lag features after each step.
        # Exogenous features (UNRATE, FEDFUNDS, their lags/rolling stats) are held
        # constant at last observed values — documented assumption in methodology.md.
        last_X_row = X.iloc[-1].copy()  # named Series: feature names as index
        target_col = self.config.features.target_column
        try:
            forecast_series = _recursive_forecast(
                model_obj=best_model_obj,
                X_last=last_X_row,
                y_history=y,
                horizon=horizon,
                future_dates=future_dates,
                target_col=target_col,
            )
        except Exception as exc:
            logger.error("Recursive forecast failed: %s. Falling back to last known value.", exc)
            forecast_series = pd.Series(
                [float(y.iloc[-1])] * horizon, index=future_dates, name="forecast"
            )

        return forecast_series, None, None

    def _build_evidence_store(
        self,
        df_raw: pd.DataFrame,
        df_features: pd.DataFrame,
        y: pd.Series,
        source_used: str,
        backtest_results: list[BacktestResult],
        registry: ModelRegistry,
        best_name: str,
        best_result: BacktestResult,
        best_model_obj: Any,
        forecast_series: pd.Series,
        conf_lower: Optional[pd.Series],
        conf_upper: Optional[pd.Series],
    ) -> dict[str, Any]:
        """Build the complete evidence store for report generation.

        Args:
            All pipeline outputs.

        Returns:
            Evidence store dict.
        """
        # ── Descriptive stats ─────────────────────────────────────────────────
        cpi_yoy = df_features.get("cpi_yoy", y)
        unrate = df_raw.get("UNRATE", pd.Series(dtype=float))
        fedfunds = df_raw.get("FEDFUNDS", pd.Series(dtype=float))

        # ── OLS summary — always extracted from registry, not just when OLS wins ─
        ols_summary: dict[str, Any] = {}
        ols_r2 = float("nan")
        ols_adj_r2 = float("nan")
        ols_nobs = 0
        ols_formula = self.config.models.ols.formula

        try:
            ols_summary = self._try_get_ols_summary(registry)
            if ols_summary:
                ols_r2 = ols_summary.get("r_squared", float("nan"))
                ols_adj_r2 = ols_summary.get("adj_r_squared", float("nan"))
                ols_nobs = ols_summary.get("nobs", 0)
        except Exception as exc:
            logger.warning("OLS summary extraction failed: %s", exc)

        # ── Model metrics ─────────────────────────────────────────────────────
        model_metrics: dict[str, dict[str, float]] = {}
        for r in backtest_results:
            model_metrics[r.model_name] = {
                "mae": r.aggregate_metrics.get("mae", {}).get("mean", float("nan")),
                "rmse": r.aggregate_metrics.get("rmse", {}).get("mean", float("nan")),
                "mape": r.aggregate_metrics.get("mape", {}).get("mean", float("nan")),
            }

        # ── Forecast dict ─────────────────────────────────────────────────────
        forecast_values = {
            str(ts.strftime("%Y-%m")): float(val)
            for ts, val in forecast_series.items()
        }

        # ── Library versions ──────────────────────────────────────────────────
        import importlib

        def _get_version(pkg: str) -> str:
            try:
                return importlib.metadata.version(pkg)
            except Exception:
                return "unknown"

        evidence: dict[str, Any] = {
            # Run metadata
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config.hash(),
            "config_path": str(self.config_path),
            "seed": self.config.seed,
            "data_source": source_used,
            # Library versions
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pandas_version": _get_version("pandas"),
            "statsmodels_version": _get_version("statsmodels"),
            "sklearn_version": _get_version("scikit-learn"),
            "torch_version": _get_version("torch"),
            # Data stats
            "start_date": str(y.index.min().strftime("%Y-%m")),
            "end_date": str(y.index.max().strftime("%Y-%m")),
            "n_obs": int(len(y)),
            "n_features": int(len(df_features.columns)),
            # CPI YoY stats
            "cpi_yoy_mean": float(cpi_yoy.mean()) if len(cpi_yoy) > 0 else float("nan"),
            "cpi_yoy_std": float(cpi_yoy.std()) if len(cpi_yoy) > 0 else float("nan"),
            "cpi_yoy_min": float(cpi_yoy.min()) if len(cpi_yoy) > 0 else float("nan"),
            "cpi_yoy_max": float(cpi_yoy.max()) if len(cpi_yoy) > 0 else float("nan"),
            "cpi_yoy_latest": float(cpi_yoy.iloc[-1]) if len(cpi_yoy) > 0 else float("nan"),
            # UNRATE stats
            "unrate_mean": float(unrate.mean()) if len(unrate) > 0 else float("nan"),
            "unrate_std": float(unrate.std()) if len(unrate) > 0 else float("nan"),
            "unrate_min": float(unrate.min()) if len(unrate) > 0 else float("nan"),
            "unrate_max": float(unrate.max()) if len(unrate) > 0 else float("nan"),
            # FEDFUNDS stats
            "fedfunds_mean": float(fedfunds.mean()) if len(fedfunds) > 0 else float("nan"),
            "fedfunds_std": float(fedfunds.std()) if len(fedfunds) > 0 else float("nan"),
            "fedfunds_min": float(fedfunds.min()) if len(fedfunds) > 0 else float("nan"),
            "fedfunds_max": float(fedfunds.max()) if len(fedfunds) > 0 else float("nan"),
            # Model evaluation
            "n_models": len(backtest_results),
            "n_folds": self.config.evaluation.n_folds,
            "model_metrics": model_metrics,
            "best_model": best_name,
            "best_rmse": best_result.aggregate_metrics.get("rmse", {}).get("mean", float("nan")),
            "best_mae": best_result.aggregate_metrics.get("mae", {}).get("mean", float("nan")),
            "best_mape": best_result.aggregate_metrics.get("mape", {}).get("mean", float("nan")),
            # OLS
            "formula": ols_formula,
            "hac_lags": self.config.models.ols.hac_lags,
            "r_squared": ols_r2,
            "adj_r_squared": ols_adj_r2,
            "nobs": ols_nobs,
            **ols_summary,
            # Forecast
            "forecast_horizon": self.config.evaluation.final_forecast_horizon,
            "forecast_start": str(forecast_series.index[0].strftime("%Y-%m")) if len(forecast_series) > 0 else "N/A",
            "forecast_values": forecast_values,
            "forecast_conf_lower": (
                {str(ts.strftime("%Y-%m")): float(v) for ts, v in conf_lower.items()}
                if conf_lower is not None else {}
            ),
            "forecast_conf_upper": (
                {str(ts.strftime("%Y-%m")): float(v) for ts, v in conf_upper.items()}
                if conf_upper is not None else {}
            ),
        }

        return evidence

    def _try_get_ols_summary(self, registry: ModelRegistry) -> dict[str, Any]:
        """Extract OLS summary from the model registry.

        OLS is always reported as an econometric baseline regardless of which model
        was selected as best.  Returns an empty dict if OLS was not fitted.

        Args:
            registry: The ModelRegistry populated during _fit_and_backtest.

        Returns:
            Dict with OLS coefficients, diagnostics, and fit statistics; empty if
            OLS was not run or failed.
        """
        from eco_report_ai.models.econometrics import OLSModel

        try:
            ols_obj = registry.get("OLS")
        except KeyError:
            logger.info("OLS not in registry (disabled or failed). Skipping OLS summary.")
            return {}

        if not isinstance(ols_obj, OLSModel) or ols_obj._result is None:
            return {}

        try:
            summary = ols_obj.get_summary()
            return {
                "ols_coefficients": summary.coefficients,
                "ols_std_errors": summary.std_errors,
                "ols_p_values": summary.p_values,
                "ols_t_stats": summary.t_stats,
                "r_squared": summary.r_squared,
                "adj_r_squared": summary.adj_r_squared,
                "nobs": summary.nobs,
                "durbin_watson": summary.durbin_watson,
            }
        except Exception as exc:
            logger.warning("OLS summary extraction: %s", exc)
            return {}


def _plot_all(
    df_raw: pd.DataFrame,
    y: pd.Series,
    forecast_series: pd.Series,
    conf_lower: Optional[pd.Series],
    conf_upper: Optional[pd.Series],
    best_name: str,
    model_metrics: dict[str, dict[str, float]],
    config: PipelineConfig,
) -> None:
    """Generate and save all report charts."""
    cfg = config.reporting

    try:
        labels = {
            "cpi_yoy": "CPI YoY (%)",
            "CPIAUCSL": "CPI Level",
            "UNRATE": "Unemployment (%)",
            "FEDFUNDS": "Fed Funds (%)",
        }
        # Always include CPI YoY (the forecast target) plus available exogenous series
        plot_df = df_raw.copy()
        plot_df["cpi_yoy"] = y  # add the computed target series
        cols_to_plot = ["cpi_yoy"] + [c for c in ["UNRATE", "FEDFUNDS"] if c in df_raw.columns]
        plot_historical_series(
            df=plot_df,
            columns=cols_to_plot,
            labels=labels,
            figures_dir=cfg.figures_dir,
            dpi=cfg.figure_dpi,
            style=cfg.figure_style,
        )
    except Exception as exc:
        logger.error("Historical chart failed: %s", exc)

    try:
        plot_forecast(
            historical=y,
            forecast=forecast_series,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
            model_name=best_name,
            figures_dir=cfg.figures_dir,
            dpi=cfg.figure_dpi,
            style=cfg.figure_style,
        )
    except Exception as exc:
        logger.error("Forecast chart failed: %s", exc)

    try:
        model_mean_metrics = {
            name: {"mae": v["mae"], "rmse": v["rmse"], "mape": v["mape"]}
            for name, v in model_metrics.items()
        }
        plot_model_comparison(
            model_metrics=model_mean_metrics,
            figures_dir=cfg.figures_dir,
            dpi=cfg.figure_dpi,
            style=cfg.figure_style,
        )
    except Exception as exc:
        logger.error("Comparison chart failed: %s", exc)


def run_pipeline(
    config_path: str = "config.yaml",
    dry_run: bool = False,
    override_source: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience function to load config and run the pipeline.

    Args:
        config_path: Path to the YAML config file.
        dry_run: If True, only log the plan without executing.
        override_source: Override data source ("fred"|"sample"|"auto").

    Returns:
        Evidence store dict.
    """
    config = load_config(config_path)
    pipeline = EcoReportPipeline(config, config_path=config_path)
    return pipeline.run(dry_run=dry_run, override_source=override_source)
