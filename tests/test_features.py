"""Tests for the feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eco_report_ai.config import FeaturesConfig
from eco_report_ai.features.build_features import (
    add_lags,
    add_rolling,
    build_features,
    compute_cpi_yoy,
)


class TestComputeCpiYoy:
    """Tests for compute_cpi_yoy()."""

    def test_correct_formula(self) -> None:
        """YoY at month 12 should equal (CPI_12 / CPI_0 - 1) * 100."""
        dates = pd.date_range("2000-01-01", periods=24, freq="MS")
        cpi = pd.Series(range(100, 124), index=dates, name="CPIAUCSL")
        df = pd.DataFrame({"CPIAUCSL": cpi})

        result = compute_cpi_yoy(df)
        expected_at_12 = (cpi.iloc[12] / cpi.iloc[0] - 1) * 100
        assert abs(result.iloc[12] - expected_at_12) < 1e-10

    def test_first_12_are_nan(self) -> None:
        """First 12 values should be NaN (no 12-month history)."""
        dates = pd.date_range("2000-01-01", periods=24, freq="MS")
        cpi = pd.Series(range(100, 124), index=dates, name="CPIAUCSL")
        df = pd.DataFrame({"CPIAUCSL": cpi})

        result = compute_cpi_yoy(df)
        assert result.iloc[:12].isna().all()

    def test_raises_on_missing_col(self) -> None:
        """Should raise KeyError if CPI column is absent."""
        df = pd.DataFrame({"OTHER": [1, 2, 3]})
        with pytest.raises(KeyError):
            compute_cpi_yoy(df)

    def test_returns_named_series(self) -> None:
        """Result should be named 'cpi_yoy'."""
        dates = pd.date_range("2000-01-01", periods=24, freq="MS")
        df = pd.DataFrame({"CPIAUCSL": range(100, 124)}, index=dates)
        result = compute_cpi_yoy(df)
        assert result.name == "cpi_yoy"


class TestAddLags:
    """Tests for add_lags()."""

    def test_lag_columns_created(self, sample_macro_df) -> None:
        """Lag columns should be present after add_lags."""
        result = add_lags(sample_macro_df, ["UNRATE"], max_lag=3)
        for lag in range(1, 4):
            assert f"UNRATE_lag_{lag}" in result.columns

    def test_lag_values_correct(self, sample_macro_df) -> None:
        """Lag 1 at row i should equal original at row i-1."""
        result = add_lags(sample_macro_df, ["UNRATE"], max_lag=1)
        assert result["UNRATE_lag_1"].iloc[5] == pytest.approx(sample_macro_df["UNRATE"].iloc[4])

    def test_original_columns_preserved(self, sample_macro_df) -> None:
        """Original columns should not be modified."""
        original_unrate = sample_macro_df["UNRATE"].copy()
        result = add_lags(sample_macro_df, ["UNRATE"], max_lag=2)
        pd.testing.assert_series_equal(result["UNRATE"], original_unrate)

    def test_missing_col_logs_warning(self, sample_macro_df, caplog) -> None:
        """Missing source column should not crash but log a warning."""
        result = add_lags(sample_macro_df, ["NONEXISTENT"], max_lag=1)
        assert "NONEXISTENT_lag_1" not in result.columns


class TestAddRolling:
    """Tests for add_rolling()."""

    def test_rolling_columns_created(self, sample_macro_df) -> None:
        """Rolling mean and std columns should be created."""
        result = add_rolling(sample_macro_df, ["UNRATE"], windows=[3, 6])
        assert "UNRATE_roll_mean_3" in result.columns
        assert "UNRATE_roll_std_3" in result.columns
        assert "UNRATE_roll_mean_6" in result.columns

    def test_no_look_ahead(self, sample_macro_df) -> None:
        """Rolling window should be shifted by 1 to prevent look-ahead."""
        result = add_rolling(sample_macro_df, ["FEDFUNDS"], windows=[3])
        # Row i's rolling mean should be based on rows i-3 to i-1 (shifted)
        # So row 3's value should be mean of rows 0,1,2
        expected = sample_macro_df["FEDFUNDS"].iloc[:3].mean()
        actual = result["FEDFUNDS_roll_mean_3"].iloc[3]
        assert abs(actual - expected) < 1e-10


class TestBuildFeatures:
    """Integration tests for build_features()."""

    def test_cpi_yoy_column_present(self, sample_macro_df, features_config) -> None:
        """Output DataFrame should contain 'cpi_yoy' column."""
        result = build_features(sample_macro_df, features_config)
        assert "cpi_yoy" in result.columns

    def test_no_nan_in_required_cols(self, sample_macro_df, features_config) -> None:
        """Required columns should not contain NaN after build_features."""
        result = build_features(sample_macro_df, features_config)
        for col in ["cpi_yoy", "fedfunds", "unrate", "cpi_yoy_lag_1"]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_output_shape_reasonable(self, sample_macro_df, features_config) -> None:
        """Output should have fewer rows than input (NaN rows dropped) but more columns."""
        result = build_features(sample_macro_df, features_config)
        assert len(result) < len(sample_macro_df)
        assert len(result.columns) > len(sample_macro_df.columns)

    def test_no_future_leakage_in_lags(self, sample_macro_df, features_config) -> None:
        """Lag 1 at any row must equal the target at the previous row."""
        result = build_features(sample_macro_df, features_config)
        result_clean = result.dropna(subset=["cpi_yoy", "cpi_yoy_lag_1"])
        if len(result_clean) > 1:
            idx = result_clean.index
            for i in range(1, min(5, len(idx))):
                expected = result_clean["cpi_yoy"].loc[idx[i - 1]]
                actual = result_clean["cpi_yoy_lag_1"].loc[idx[i]]
                assert abs(actual - expected) < 1e-10, "Lag 1 does not match previous row"
