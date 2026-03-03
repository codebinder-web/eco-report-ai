"""Feature engineering for the EcoReport AI forecasting pipeline.

Constructs lag features, rolling statistics, and the CPI YoY target variable
from the raw macro DataFrame. All operations are designed to be leakage-free:
rolling windows use only past observations (no look-ahead).
"""

from __future__ import annotations

import logging

import pandas as pd

from eco_report_ai.config import FeaturesConfig

logger = logging.getLogger(__name__)

# Columns to engineer features from (in addition to the target)
FEATURE_SOURCE_COLS = ["CPIAUCSL", "UNRATE", "FEDFUNDS"]


def compute_cpi_yoy(df: pd.DataFrame, cpi_col: str = "CPIAUCSL") -> pd.Series:
    """Compute CPI Year-over-Year percentage change.

    Formula: (CPI_t / CPI_{t-12} - 1) * 100

    Args:
        df: DataFrame with a CPIAUCSL column and monthly DatetimeIndex.
        cpi_col: Name of the CPI level column.

    Returns:
        Series named "cpi_yoy" with the same index.

    Raises:
        KeyError: If cpi_col is not in df.
    """
    if cpi_col not in df.columns:
        raise KeyError(f"Column '{cpi_col}' not found in DataFrame.")

    yoy = df[cpi_col].pct_change(periods=12) * 100
    yoy.name = "cpi_yoy"
    return yoy


def add_lags(
    df: pd.DataFrame,
    columns: list[str],
    max_lag: int,
) -> pd.DataFrame:
    """Add lag features for specified columns up to max_lag periods.

    Args:
        df: Input DataFrame.
        columns: Column names to lag.
        max_lag: Maximum number of lag periods.

    Returns:
        DataFrame with appended lag columns (original columns preserved).
    """
    result = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Lag: column '%s' not found, skipping.", col)
            continue
        for lag in range(1, max_lag + 1):
            result[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return result


def add_rolling(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features.

    Uses min_periods=1 to avoid NaN at the start of the series.
    shift(1) ensures no look-ahead: the rolling window ends at t-1.

    Args:
        df: Input DataFrame.
        columns: Column names to compute rolling statistics for.
        windows: List of rolling window sizes (in months).

    Returns:
        DataFrame with appended rolling feature columns.
    """
    result = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Rolling: column '%s' not found, skipping.", col)
            continue
        for w in windows:
            shifted = df[col].shift(1)  # prevent look-ahead
            result[f"{col}_roll_mean_{w}"] = (
                shifted.rolling(window=w, min_periods=1).mean()
            )
            result[f"{col}_roll_std_{w}"] = (
                shifted.rolling(window=w, min_periods=1).std()
            )
    return result


def build_features(
    df: pd.DataFrame,
    config: FeaturesConfig,
) -> pd.DataFrame:
    """Build the full feature matrix including target variable.

    Steps:
        1. Compute CPI YoY target.
        2. Rename raw series to lowercase for ease of use in OLS formula.
        3. Add lag features up to max_lag.
        4. Add rolling mean/std features.
        5. Optionally first-difference features.
        6. Drop rows with NaN introduced by lag/rolling operations.

    Args:
        df: Raw macro DataFrame (CPIAUCSL, UNRATE, FEDFUNDS) with monthly index.
        config: FeaturesConfig instance.

    Returns:
        Processed feature DataFrame with target column and all engineered features.
        Index is the same monthly DatetimeIndex; first `max_lag + max_window` rows
        may be dropped due to NaN from lag/rolling.
    """
    result = df.copy()

    # ── 1. Compute target ─────────────────────────────────────────────────────
    result["cpi_yoy"] = compute_cpi_yoy(df)
    logger.info("Computed CPI YoY target variable.")

    # ── 2. Lowercase column aliases for OLS formula compatibility ─────────────
    result["fedfunds"] = result["FEDFUNDS"]
    result["unrate"] = result["UNRATE"]
    result["cpiaucsl"] = result["CPIAUCSL"]

    # ── 3. Lag features ───────────────────────────────────────────────────────
    lag_cols = [config.target_column, "fedfunds", "unrate"]
    result = add_lags(result, lag_cols, config.max_lag)
    logger.info("Added lags 1..%d for %s.", config.max_lag, lag_cols)

    # ── 4. Rolling features ───────────────────────────────────────────────────
    roll_cols = [config.target_column, "fedfunds", "unrate"]
    result = add_rolling(result, roll_cols, config.rolling_windows)
    logger.info("Added rolling stats with windows %s.", config.rolling_windows)

    # ── 5. Optional differencing ──────────────────────────────────────────────
    if config.use_differencing:
        for col in ["fedfunds", "unrate", config.target_column]:
            if col in result.columns:
                result[f"{col}_diff1"] = result[col].diff(1)
        logger.info("Applied first differencing.")

    # ── 6. Drop initial NaN rows ──────────────────────────────────────────────
    before = len(result)
    # Require at minimum target + primary regressors to be non-NaN
    required_non_null = [
        config.target_column,
        "fedfunds",
        "unrate",
        f"{config.target_column}_lag_1",
    ]
    result = result.dropna(subset=required_non_null)
    dropped = before - len(result)
    if dropped:
        logger.info("Dropped %d rows with NaN in required columns.", dropped)

    logger.info(
        "Feature matrix built: %d rows × %d columns.", len(result), len(result.columns)
    )
    return result
