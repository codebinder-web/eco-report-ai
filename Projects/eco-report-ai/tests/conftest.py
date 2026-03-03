"""Shared pytest fixtures for EcoReport AI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eco_report_ai.config import (
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    PipelineConfig,
    ReportingConfig,
)


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """Monthly date index from 2000-01 to 2019-12 (240 months)."""
    return pd.date_range("2000-01-01", periods=240, freq="MS")


@pytest.fixture
def sample_macro_df(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Realistic synthetic macro DataFrame with CPIAUCSL, UNRATE, FEDFUNDS."""
    np.random.seed(42)
    n = len(sample_dates)

    cpiaucsl = np.cumprod(1 + np.random.normal(0.002, 0.001, n)) * 169.3
    unrate = np.clip(5.0 + np.cumsum(np.random.normal(0, 0.05, n)), 3.0, 12.0)
    fedfunds = np.clip(3.0 + np.cumsum(np.random.normal(0, 0.1, n)), 0.07, 6.5)

    return pd.DataFrame(
        {"CPIAUCSL": cpiaucsl, "UNRATE": unrate, "FEDFUNDS": fedfunds},
        index=sample_dates,
    )


@pytest.fixture
def features_config() -> FeaturesConfig:
    """Minimal features config for fast tests."""
    return FeaturesConfig(max_lag=3, rolling_windows=[3], use_differencing=False)


@pytest.fixture
def sample_features_df(sample_macro_df: pd.DataFrame, features_config: FeaturesConfig) -> pd.DataFrame:
    """Pre-built feature DataFrame from sample macro data."""
    from eco_report_ai.features.build_features import build_features

    return build_features(sample_macro_df, features_config)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Minimal PipelineConfig using temp directories."""
    return PipelineConfig(
        seed=42,
        data=DataConfig(
            sample_csv_path="data/sample_macro.csv",
            source="sample",
            start_date="2000-01-01",
        ),
        features=FeaturesConfig(max_lag=3, rolling_windows=[3]),
        evaluation=EvaluationConfig(n_folds=3, forecast_horizon=3, final_forecast_horizon=6),
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "reports"),
            figures_dir=str(tmp_path / "reports" / "figures"),
        ),
    )
