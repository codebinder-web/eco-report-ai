"""End-to-end integration tests for the EcoReport AI pipeline.

These tests run the full pipeline on the sample CSV (no API keys required)
and assert that the key outputs are correct and not degenerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repo root — works regardless of the directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")
_SAMPLE_CSV = str(_REPO_ROOT / "data" / "sample_macro.csv")


@pytest.fixture(scope="module")
def pipeline_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the full pipeline once and return the evidence store.

    Uses a temp directory for outputs so we do not overwrite reports/
    during testing.  The fixture is module-scoped so the pipeline runs
    only once for all tests in this file.
    """
    from eco_report_ai.config import (
        DataConfig,
        EvaluationConfig,
        FeaturesConfig,
        LLMConfig,
        PipelineConfig,
        ReportingConfig,
    )
    from eco_report_ai.pipeline import EcoReportPipeline

    tmp = tmp_path_factory.mktemp("reports")

    config = PipelineConfig(
        seed=42,
        data=DataConfig(
            sample_csv_path=_SAMPLE_CSV,
            source="sample",
            start_date="2000-01-01",
        ),
        features=FeaturesConfig(max_lag=6, rolling_windows=[3, 6]),
        evaluation=EvaluationConfig(
            n_folds=3,
            forecast_horizon=3,
            final_forecast_horizon=12,
        ),
        reporting=ReportingConfig(
            output_dir=str(tmp / "reports"),
            figures_dir=str(tmp / "reports" / "figures"),
        ),
        llm=LLMConfig(enabled_if_key=False),
    )

    pipeline = EcoReportPipeline(config, config_path=_CONFIG_PATH)
    evidence = pipeline.run(override_source="sample")

    md_path = Path(config.reporting.output_dir) / "latest_report.md"
    json_path = Path(config.reporting.output_dir) / "latest_report.json"
    return {
        "evidence": evidence,
        "md_path": md_path,
        "json_path": json_path,
    }


class TestPipelineOutputs:
    """Tests that verify pipeline-level outputs from the full run."""

    def test_evidence_has_best_model(self, pipeline_evidence: dict) -> None:
        """Evidence store must name a best model."""
        evidence = pipeline_evidence["evidence"]
        assert evidence.get("best_model") not in (None, "N/A", "")

    def test_report_files_created(self, pipeline_evidence: dict) -> None:
        """Markdown and JSON report files must exist and be non-empty."""
        md_path: Path = pipeline_evidence["md_path"]
        json_path: Path = pipeline_evidence["json_path"]
        assert md_path.exists(), f"Missing markdown report: {md_path}"
        assert json_path.exists(), f"Missing JSON report: {json_path}"
        assert md_path.stat().st_size > 0, "Markdown report is empty"
        assert json_path.stat().st_size > 0, "JSON report is empty"

    def test_forecast_has_correct_length(self, pipeline_evidence: dict) -> None:
        """Forecast dict must have exactly final_forecast_horizon entries."""
        evidence = pipeline_evidence["evidence"]
        forecast = evidence.get("forecast_values", {})
        assert len(forecast) == 12, (
            f"Expected 12 forecast periods, got {len(forecast)}"
        )

    def test_forecast_is_not_flat(self, pipeline_evidence: dict) -> None:
        """Recursive forecasting must produce non-constant predictions.

        A flat forecast (all identical values) is a sign the lag features
        were not updated between steps.  We allow a small tolerance for
        models that legitimately converge quickly, but require at least
        two distinct values across 12 steps.
        """
        evidence = pipeline_evidence["evidence"]
        forecast = evidence.get("forecast_values", {})
        values = list(forecast.values())
        assert len(values) >= 2, "Forecast is empty"
        unique_values = set(round(v, 6) for v in values)
        assert len(unique_values) > 1, (
            f"Forecast is perfectly flat ({values[0]:.4f} for all {len(values)} steps). "
            "Lag features were not updated during recursive multi-step forecasting."
        )

    def test_model_metrics_present(self, pipeline_evidence: dict) -> None:
        """At least one model must have MAE, RMSE, MAPE metrics."""
        evidence = pipeline_evidence["evidence"]
        model_metrics = evidence.get("model_metrics", {})
        assert len(model_metrics) > 0, "No model metrics recorded"
        for name, metrics in model_metrics.items():
            assert "mae" in metrics and "rmse" in metrics, (
                f"Model '{name}' missing MAE/RMSE"
            )

    def test_ols_summary_always_present(self, pipeline_evidence: dict) -> None:
        """OLS coefficients must appear in the evidence even if OLS is not best."""
        evidence = pipeline_evidence["evidence"]
        coeffs = evidence.get("ols_coefficients")
        assert coeffs is not None and len(coeffs) > 0, (
            "OLS summary missing from evidence store. "
            "Check _try_get_ols_summary — it should use the registry directly."
        )

    def test_durbin_watson_in_evidence(self, pipeline_evidence: dict) -> None:
        """Durbin-Watson statistic must be recorded in the evidence store."""
        evidence = pipeline_evidence["evidence"]
        dw = evidence.get("durbin_watson")
        assert dw is not None, "durbin_watson missing from evidence"
        assert isinstance(dw, float), f"Expected float, got {type(dw)}"
        assert 0.0 <= dw <= 4.0, f"DW out of valid range [0,4]: {dw}"

    def test_n_obs_is_positive(self, pipeline_evidence: dict) -> None:
        """Pipeline must have loaded a non-trivial number of observations."""
        evidence = pipeline_evidence["evidence"]
        assert evidence.get("n_obs", 0) > 50

    def test_best_rmse_is_finite(self, pipeline_evidence: dict) -> None:
        """Best RMSE must be a finite positive number."""
        import math
        evidence = pipeline_evidence["evidence"]
        rmse = evidence.get("best_rmse", float("nan"))
        assert not math.isnan(rmse) and not math.isinf(rmse), (
            f"best_rmse is not finite: {rmse}"
        )
        assert rmse > 0
