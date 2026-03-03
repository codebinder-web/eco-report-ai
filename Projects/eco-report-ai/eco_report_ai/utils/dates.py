"""Date and time-series utilities for EcoReport AI."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def enforce_monthly_freq(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Ensure the DataFrame has a monthly DateTimeIndex at month start ('MS').

    If the index is not already a DatetimeIndex, `date_col` is used as the
    source column.

    Args:
        df: Input DataFrame.
        date_col: Column name to use as the date index if the index is not
            already a DatetimeIndex.

    Returns:
        DataFrame with a monthly MS DatetimeIndex, resampled to fill gaps.

    Raises:
        ValueError: If neither the index nor `date_col` yields a DatetimeIndex.
    """
    if date_col is not None and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame must have a DatetimeIndex or a date_col must be provided."
        )

    # Normalize to month start (to_timestamp() defaults to period start = MS)
    df.index = df.index.to_period("M").to_timestamp()
    df.index.name = "date"

    # Resample to fill any gaps in the monthly index
    original_freq = pd.infer_freq(df.index)
    df = df.resample("MS").last()

    if original_freq != "MS":
        logger.info(
            "Resampled from inferred frequency '%s' to monthly ('MS'). "
            "Rows: %d",
            original_freq,
            len(df),
        )

    return df


def parse_date_range(
    start: str | None,
    end: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse start/end date strings to Timestamps.

    Args:
        start: ISO date string for range start (e.g., "2000-01-01").
        end: ISO date string for range end; defaults to today.

    Returns:
        Tuple of (start_ts, end_ts) as pd.Timestamp objects.
    """
    start_ts = pd.Timestamp(start) if start else pd.Timestamp("2000-01-01")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    return start_ts, end_ts


def safe_date_filter(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """Filter DataFrame to a date range, clipping to available dates.

    Args:
        df: DataFrame with DatetimeIndex.
        start: Start date string.
        end: End date string.

    Returns:
        Filtered DataFrame.
    """
    start_ts, end_ts = parse_date_range(start, end)
    return df.loc[start_ts:end_ts]
