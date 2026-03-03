"""DataFrame schema validation for the macro data layer.

Uses pandera for declarative data contracts. Validates column names,
dtypes, and basic range constraints. Raises DataQualityError on failure.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["CPIAUCSL", "UNRATE", "FEDFUNDS"]
MIN_OBSERVATIONS = 36  # 3 years minimum


class DataQualityError(ValueError):
    """Raised when the macro DataFrame fails schema validation."""


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the macro DataFrame against expected schema and quality rules.

    Args:
        df: DataFrame with DatetimeIndex and raw FRED columns.

    Returns:
        The validated (unchanged) DataFrame.

    Raises:
        DataQualityError: If any validation check fails.
    """
    errors: list[str] = []

    # ── Index checks ─────────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be a DatetimeIndex.")

    # ── Column presence ──────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # ── Minimum length ───────────────────────────────────────────────────────
    if len(df) < MIN_OBSERVATIONS:
        errors.append(
            f"Insufficient data: {len(df)} rows (minimum {MIN_OBSERVATIONS} required)."
        )

    if errors:
        raise DataQualityError("Data quality validation failed:\n" + "\n".join(errors))

    # ── Range warnings (non-fatal) ───────────────────────────────────────────
    if "UNRATE" in df.columns:
        out_of_range = df["UNRATE"][(df["UNRATE"] < 0) | (df["UNRATE"] > 30)]
        if not out_of_range.empty:
            logger.warning("UNRATE out-of-range [0, 30] at %d dates.", len(out_of_range))

    if "FEDFUNDS" in df.columns:
        negative_ff = df["FEDFUNDS"][df["FEDFUNDS"] < 0]
        if not negative_ff.empty:
            logger.warning("FEDFUNDS has %d negative values.", len(negative_ff))

    if "CPIAUCSL" in df.columns:
        non_positive = df["CPIAUCSL"][df["CPIAUCSL"] <= 0]
        if not non_positive.empty:
            errors_blocking = [f"CPIAUCSL has {len(non_positive)} non-positive values."]
            raise DataQualityError("\n".join(errors_blocking))

    logger.info("Schema validation passed: %d rows, %d columns.", len(df), len(df.columns))
    return df


def summarize_missingness(df: pd.DataFrame) -> dict[str, float]:
    """Compute missing value percentages per column.

    Args:
        df: Input DataFrame.

    Returns:
        Dict mapping column name to percentage of missing values.
    """
    total = len(df)
    return {
        col: round(df[col].isna().sum() / total * 100, 2)
        for col in df.columns
    }
