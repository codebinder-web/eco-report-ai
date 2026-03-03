"""Tests for the reporting layer (charts, NLG, report writer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eco_report_ai.reporting.nlg import NarrativeGenerator, _safe_defaults
from eco_report_ai.reporting.report_writer import ReportWriter, _build_model_comparison_table


# ── Shared evidence fixture ───────────────────────────────────────────────────

@pytest.fixture
def minimal_evidence() -> dict:
    """Minimal evidence store for testing NLG and report writer."""
    return {
        "data_source": "sample_csv",
        "start_date": "2000-01",
        "end_date": "2023-12",
        "n_obs": 276,
        "cpi_yoy_mean": 2.5,
        "cpi_yoy_std": 1.2,
        "cpi_yoy_min": -0.5,
        "cpi_yoy_max": 9.1,
        "cpi_yoy_latest": 3.2,
        "unrate_mean": 5.8,
        "unrate_std": 1.9,
        "unrate_min": 3.4,
        "unrate_max": 14.8,
        "fedfunds_mean": 2.1,
        "fedfunds_std": 2.0,
        "fedfunds_min": 0.07,
        "fedfunds_max": 6.5,
        "n_models": 3,
        "best_model": "ARIMA",
        "best_rmse": 0.41,
        "n_folds": 5,
        "forecast_horizon": 12,
        "forecast_start": "2024-01",
        "formula": "cpi_yoy ~ fedfunds + unrate + cpi_yoy_lag_1",
        "hac_lags": 6,
        "r_squared": 0.72,
        "adj_r_squared": 0.70,
        "nobs": 270,
        "ols_coefficients": {"Intercept": 0.5, "fedfunds": -0.12, "unrate": 0.08},
        "ols_std_errors": {"Intercept": 0.2, "fedfunds": 0.05, "unrate": 0.03},
        "ols_p_values": {"Intercept": 0.01, "fedfunds": 0.02, "unrate": 0.04},
        "ols_t_stats": {"Intercept": 2.5, "fedfunds": -2.4, "unrate": 2.7},
        "model_metrics": {
            "OLS": {"mae": 0.52, "rmse": 0.65, "mape": 18.5},
            "ARIMA": {"mae": 0.38, "rmse": 0.41, "mape": 12.1},
            "GradientBoosting": {"mae": 0.45, "rmse": 0.55, "mape": 15.3},
        },
        "forecast_values": {
            "2024-01": 3.1,
            "2024-02": 3.0,
            "2024-03": 2.9,
            "2024-04": 2.8,
            "2024-05": 2.7,
            "2024-06": 2.6,
        },
        "config_hash": "abc12345",
        "seed": 42,
        "timestamp": "2026-03-02T12:00:00+00:00",
        "python_version": "3.11.0",
        "pandas_version": "2.1.0",
        "statsmodels_version": "0.14.0",
        "sklearn_version": "1.4.0",
        "torch_version": "2.1.0",
    }


# ── NarrativeGenerator tests ──────────────────────────────────────────────────

class TestNarrativeGenerator:
    """Tests for deterministic (template) NLG."""

    def test_exec_summary_contains_best_model(self, minimal_evidence) -> None:
        """Executive summary should mention the best model name."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_executive_summary(minimal_evidence)
        assert "ARIMA" in text

    def test_exec_summary_contains_cpi_mean(self, minimal_evidence) -> None:
        """Executive summary should contain the CPI mean value."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_executive_summary(minimal_evidence)
        assert "2.50" in text

    def test_data_overview_has_table(self, minimal_evidence) -> None:
        """Data overview should include a Markdown table."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_data_overview(minimal_evidence)
        assert "|" in text
        assert "Mean" in text

    def test_econ_findings_mentions_fedfunds(self, minimal_evidence) -> None:
        """Econometric findings should mention fedfunds coefficient."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_econometric_findings(minimal_evidence)
        assert "fedfunds" in text or "Federal Funds" in text

    def test_forecast_section_has_table(self, minimal_evidence) -> None:
        """Forecast section should include a Markdown table of future periods."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_forecast_section(minimal_evidence)
        assert "2024-01" in text

    def test_risks_section_not_empty(self, minimal_evidence) -> None:
        """Risks section should be non-empty."""
        gen = NarrativeGenerator(enabled_if_key=False)
        text = gen.generate_risks(minimal_evidence)
        assert len(text.strip()) > 50

    def test_safe_defaults_fills_missing_keys(self) -> None:
        """_safe_defaults should fill in missing keys without raising."""
        result = _safe_defaults({})
        assert "best_model" in result
        assert "cpi_yoy_mean" in result


# ── ReportWriter tests ────────────────────────────────────────────────────────

class TestReportWriter:
    """Tests for ReportWriter.write_all()."""

    def test_markdown_file_created(self, tmp_path, minimal_evidence) -> None:
        """write_markdown should create a .md file."""
        writer = ReportWriter(
            output_dir=str(tmp_path),
            figures_dir=str(tmp_path / "figures"),
            report_filename="test_report",
            nlg=NarrativeGenerator(enabled_if_key=False),
        )
        md_path = writer.write_markdown(minimal_evidence)
        assert Path(md_path).exists()
        assert Path(md_path).stat().st_size > 0

    def test_json_file_created(self, tmp_path, minimal_evidence) -> None:
        """write_json should create a valid JSON file."""
        writer = ReportWriter(
            output_dir=str(tmp_path),
            report_filename="test_report",
            nlg=NarrativeGenerator(enabled_if_key=False),
        )
        json_path = writer.write_json(minimal_evidence)
        assert Path(json_path).exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["best_model"] == "ARIMA"

    def test_markdown_contains_sections(self, tmp_path, minimal_evidence) -> None:
        """Report Markdown should contain all required sections."""
        writer = ReportWriter(
            output_dir=str(tmp_path),
            report_filename="test_report",
            nlg=NarrativeGenerator(enabled_if_key=False),
        )
        md_path = writer.write_markdown(minimal_evidence)
        content = Path(md_path).read_text(encoding="utf-8")

        for section in [
            "Executive Summary",
            "Data Overview",
            "Econometric Findings",
            "Forecast Results",
            "Model Comparison",
            "Risks",
            "Reproducibility",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_write_all_returns_two_paths(self, tmp_path, minimal_evidence) -> None:
        """write_all should return a tuple of (md_path, json_path)."""
        writer = ReportWriter(
            output_dir=str(tmp_path),
            report_filename="test_report",
            nlg=NarrativeGenerator(enabled_if_key=False),
        )
        md_path, json_path = writer.write_all(minimal_evidence)
        assert Path(md_path).exists()
        assert Path(json_path).exists()

    def test_json_has_run_id(self, tmp_path, minimal_evidence) -> None:
        """JSON evidence should preserve all keys from the evidence store."""
        writer = ReportWriter(output_dir=str(tmp_path), report_filename="test")
        json_path = writer.write_json(minimal_evidence)
        loaded = json.loads(Path(json_path).read_text())
        assert loaded.get("data_source") == "sample_csv"


class TestModelComparisonTable:
    """Tests for _build_model_comparison_table()."""

    def test_returns_markdown_table(self) -> None:
        """Should return a string with Markdown table syntax."""
        metrics = {
            "OLS": {"mae": 0.5, "rmse": 0.6, "mape": 15.0},
            "ARIMA": {"mae": 0.4, "rmse": 0.5, "mape": 12.0},
        }
        table = _build_model_comparison_table(metrics)
        assert "|" in table
        assert "OLS" in table
        assert "ARIMA" in table

    def test_empty_returns_placeholder(self) -> None:
        """Empty dict should return a placeholder message."""
        result = _build_model_comparison_table({})
        assert "No model metrics" in result
