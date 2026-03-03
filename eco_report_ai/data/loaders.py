"""Data loading orchestration for EcoReport AI.

Implements the FRED-first, CSV-fallback strategy described in design_decisions.md.
Transparently selects the data source based on environment and config,
enforces monthly frequency, handles missing values, and validates the schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from eco_report_ai.config import DataConfig
from eco_report_ai.data.fred_client import FREDClient
from eco_report_ai.data.schema import DataQualityError, summarize_missingness, validate_dataframe
from eco_report_ai.utils.dates import enforce_monthly_freq, safe_date_filter

logger = logging.getLogger(__name__)


def load_macro_data(
    config: DataConfig,
    override_source: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Load macroeconomic data according to config source strategy.

    Tries FRED if source is "auto" or "fred" and the API key is available,
    otherwise falls back to the local sample CSV.

    Args:
        config: DataConfig instance from the pipeline config.
        override_source: If provided, overrides config.source ("auto"|"fred"|"sample").

    Returns:
        Tuple of (processed DataFrame, source_used string).

    Raises:
        DataQualityError: If loaded data fails schema validation.
        FileNotFoundError: If sample CSV is requested but not found.
    """
    source = override_source or config.source
    client = FREDClient()

    if source in ("auto", "fred") and client.is_available():
        try:
            df = load_from_fred(config, client)
            source_used = "fred"
            logger.info("Data loaded from FRED API.")
        except Exception as exc:
            if source == "fred":
                raise
            logger.warning(
                "FRED fetch failed (%s). Falling back to sample CSV.", exc
            )
            df = load_from_csv(config)
            source_used = "sample_csv"
    else:
        if source == "fred":
            logger.warning(
                "source='fred' but FRED_API_KEY is not set. "
                "Falling back to sample CSV."
            )
        df = load_from_csv(config)
        source_used = "sample_csv"

    df = _postprocess(df, config)
    validate_dataframe(df)

    missingness = summarize_missingness(df)
    for col, pct in missingness.items():
        if pct > 0:
            logger.warning("Column '%s' has %.1f%% missing values after processing.", col, pct)

    return df, source_used


def load_from_fred(config: DataConfig, client: Optional[FREDClient] = None) -> pd.DataFrame:
    """Fetch macro series directly from FRED.

    Args:
        config: DataConfig with series IDs and date range.
        client: Optional pre-configured FREDClient.

    Returns:
        Raw DataFrame from FRED with DatetimeIndex.
    """
    if client is None:
        client = FREDClient()
    return client.fetch_all(
        config.fred_series,
        start_date=config.start_date,
        end_date=config.end_date,
    )


def load_from_csv(config: DataConfig) -> pd.DataFrame:
    """Load macro data from the local sample CSV.

    Args:
        config: DataConfig with sample_csv_path.

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(config.sample_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample CSV not found at '{csv_path}'. "
            "Run the data generation script or provide a FRED_API_KEY."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    logger.info("Loaded sample CSV: %d rows from %s", len(df), csv_path)
    return df


def _postprocess(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Apply frequency enforcement, date filtering, and missing-value imputation.

    Args:
        df: Raw loaded DataFrame.
        config: DataConfig for date range.

    Returns:
        Processed DataFrame ready for feature engineering.
    """
    # Enforce monthly frequency
    df = enforce_monthly_freq(df)

    # Date range filter
    df = safe_date_filter(df, config.start_date, config.end_date)

    # Convert all columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing value strategy
    missing_pcts = summarize_missingness(df)
    for col in df.columns:
        pct = missing_pcts.get(col, 0.0)
        if pct == 0:
            continue
        if pct <= 5.0:
            df[col] = df[col].ffill().bfill()
            logger.info("Column '%s': forward/back-filled (%.1f%% missing).", col, pct)
        else:
            df[col] = df[col].interpolate(method="time")
            df[col] = df[col].ffill().bfill()
            logger.warning(
                "Column '%s': linear-interpolated (%.1f%% missing > 5%%).", col, pct
            )

    # Drop any remaining all-NaN rows (should be rare)
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        logger.warning("Dropped %d all-NaN rows.", before - len(df))

    return df
