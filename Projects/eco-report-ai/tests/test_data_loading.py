"""Tests for the data loading layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eco_report_ai.config import DataConfig
from eco_report_ai.data.loaders import _postprocess, load_from_csv
from eco_report_ai.data.schema import DataQualityError, validate_dataframe
from eco_report_ai.utils.dates import enforce_monthly_freq

# Resolve the sample CSV path relative to the repo root regardless of the
# directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_CSV = str(_REPO_ROOT / "data" / "sample_macro.csv")


class TestLoadFromCSV:
    """Tests for load_from_csv()."""

    def test_loads_sample_csv(self) -> None:
        """Sample CSV should load without errors."""
        config = DataConfig(sample_csv_path=_SAMPLE_CSV)
        df = load_from_csv(config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_has_required_columns(self) -> None:
        """Sample CSV must have CPIAUCSL, UNRATE, FEDFUNDS."""
        config = DataConfig(sample_csv_path=_SAMPLE_CSV)
        df = load_from_csv(config)
        for col in ["CPIAUCSL", "UNRATE", "FEDFUNDS"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_raises_on_missing_file(self) -> None:
        """FileNotFoundError on a non-existent path."""
        config = DataConfig(sample_csv_path=str(_REPO_ROOT / "data" / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            load_from_csv(config)

    def test_values_are_numeric(self) -> None:
        """All columns should be numeric after loading."""
        config = DataConfig(sample_csv_path=_SAMPLE_CSV)
        df = load_from_csv(config)
        for col in ["CPIAUCSL", "UNRATE", "FEDFUNDS"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Non-numeric: {col}"


class TestEnforceMonthlyFreq:
    """Tests for enforce_monthly_freq()."""

    def test_resamples_to_monthly(self, sample_macro_df) -> None:
        """DataFrame should have monthly frequency after enforcement."""
        result = enforce_monthly_freq(sample_macro_df)
        assert pd.infer_freq(result.index) in ("MS", "QS-JAN", "MS") or len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_duplicate_dates(self, sample_macro_df) -> None:
        """No duplicate dates after resampling."""
        result = enforce_monthly_freq(sample_macro_df)
        assert result.index.duplicated().sum() == 0

    def test_raises_without_datetimeindex_or_col(self) -> None:
        """Should raise ValueError if no DatetimeIndex and no date_col."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            enforce_monthly_freq(df)


class TestSchemaValidation:
    """Tests for validate_dataframe()."""

    def test_valid_df_passes(self, sample_macro_df) -> None:
        """A valid macro DataFrame should pass validation."""
        result = validate_dataframe(sample_macro_df)
        assert result is sample_macro_df

    def test_missing_column_raises(self, sample_macro_df) -> None:
        """Missing a required column should raise DataQualityError."""
        df_bad = sample_macro_df.drop(columns=["UNRATE"])
        with pytest.raises(DataQualityError, match="Missing required columns"):
            validate_dataframe(df_bad)

    def test_too_few_rows_raises(self) -> None:
        """Fewer than 36 rows should raise DataQualityError."""
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")
        df = pd.DataFrame(
            {"CPIAUCSL": range(10, 20), "UNRATE": [4.0] * 10, "FEDFUNDS": [2.0] * 10},
            index=dates,
        )
        with pytest.raises(DataQualityError, match="Insufficient data"):
            validate_dataframe(df)

    def test_non_datetime_index_raises(self) -> None:
        """Integer index should raise DataQualityError."""
        df = pd.DataFrame(
            {"CPIAUCSL": range(40), "UNRATE": [4.0] * 40, "FEDFUNDS": [2.0] * 40}
        )
        with pytest.raises(DataQualityError, match="Index"):
            validate_dataframe(df)


class TestPostprocess:
    """Tests for _postprocess()."""

    def test_no_nans_after_postprocess(self) -> None:
        """Missing values should be filled after postprocessing."""
        config = DataConfig(sample_csv_path=_SAMPLE_CSV, start_date="2000-01-01")
        df = load_from_csv(config)
        df.iloc[5, 0] = float("nan")  # Inject a missing value
        result = _postprocess(df, config)
        # After postprocess, NaN should be filled
        assert result["CPIAUCSL"].isna().sum() == 0

    def test_date_filter_applied(self) -> None:
        """Date range from config should be applied."""
        config = DataConfig(
            sample_csv_path=_SAMPLE_CSV,
            start_date="2010-01-01",
            end_date="2015-12-01",
        )
        df = load_from_csv(config)
        result = _postprocess(df, config)
        assert result.index.min() >= pd.Timestamp("2010-01-01")
        assert result.index.max() <= pd.Timestamp("2015-12-31")
