"""Natural Language Generation for EcoReport AI.

Implements a two-tier NLG strategy:
1. Deterministic template NLG (always available, offline-safe)
2. Optional LLM enhancement via OpenAI (if OPENAI_API_KEY is set)

Core design principle from course_raw_notes.md:
  "The LLM must not compute. It must only describe."
  "Numerical hallucination is a structural property of generative modeling."

All numeric values passed to the LLM come from the pre-computed evidence store.
The LLM is explicitly instructed not to generate numbers not provided to it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Template sentences ─────────────────────────────────────────────────────────

_EXEC_SUMMARY_TEMPLATE = """\
This report presents a comprehensive macroeconomic analysis of U.S. inflation dynamics
using data from {data_source} covering {start_date} to {end_date} ({n_obs} monthly observations).

CPI inflation (year-over-year) averaged **{cpi_yoy_mean:.2f}%** over the sample period,
with a range of {cpi_yoy_min:.2f}% to {cpi_yoy_max:.2f}%. The most recent observation
stands at **{cpi_yoy_latest:.2f}%**.

Among {n_models} models evaluated via rolling-origin cross-validation ({n_folds} folds),
the **{best_model}** achieved the lowest RMSE of **{best_rmse:.4f}** percentage points.
A {forecast_horizon}-month forward forecast is presented in Section 4.
"""

_DATA_OVERVIEW_TEMPLATE = """\
The analysis draws on three monthly FRED series:
- **CPIAUCSL**: Consumer Price Index (All Urban Consumers, 1982–84=100)
- **UNRATE**: Civilian Unemployment Rate (%)
- **FEDFUNDS**: Effective Federal Funds Rate (%)

| Statistic | CPI YoY (%) | Unemployment (%) | Fed Funds (%) |
|-----------|-------------|-----------------|---------------|
| Mean      | {cpi_yoy_mean:.2f} | {unrate_mean:.2f} | {fedfunds_mean:.2f} |
| Std Dev   | {cpi_yoy_std:.2f} | {unrate_std:.2f} | {fedfunds_std:.2f} |
| Min       | {cpi_yoy_min:.2f} | {unrate_min:.2f} | {fedfunds_min:.2f} |
| Max       | {cpi_yoy_max:.2f} | {unrate_max:.2f} | {fedfunds_max:.2f} |

Data frequency: monthly. Missing value strategy: forward-fill then linear interpolation.
"""

_ECONOMETRIC_TEMPLATE = """\
An OLS regression was estimated with Newey-West HAC standard errors (lags={hac_lags}) to
correct for autocorrelation and heteroskedasticity common in macro time series.

**Regression Formula**: `{formula}`

| Variable | Coefficient | Std Error (HAC) | t-stat | p-value |
|----------|-------------|-----------------|--------|---------|
{coeff_table}

**Model fit**: R² = {r_squared:.4f}, Adjusted R² = {adj_r_squared:.4f}, N = {nobs}

**Residual diagnostics**: Durbin-Watson = {durbin_watson_str}
*(DW ≈ 2 indicates no serial correlation; DW < 1.5 suggests positive autocorrelation in residuals.)*

> **Note on R²**: A very high R² is expected in specifications with a lagged dependent
> variable (AR component). The AR1 term alone accounts for most explained variation;
> the incremental contribution of FEDFUNDS and UNRATE should be assessed via partial
> F-tests or an encompassing test.

**Interpretation**: {interpretation}
"""

_FORECAST_TEMPLATE = """\
The **{best_model}** model was selected as the best-performing model based on minimum
mean RMSE across rolling-origin cross-validation folds.

**{forecast_horizon}-Month Ahead Forecast** (from {forecast_start}):

| Month | Forecast (CPI YoY %) |
|-------|---------------------|
{forecast_table}

The forecast suggests inflation will {trend_description} over the next {forecast_horizon} months.
"""

_RISKS_TEMPLATE = """\
1. **Model uncertainty**: All forecasts are conditional on historical patterns continuing.
   Structural breaks (policy changes, external shocks) may invalidate projections.
2. **Data limitations**: Sample data covers {start_date}–{end_date}. Out-of-sample
   performance may differ from backtested metrics.
3. **LSTM instability**: Deep learning forecasts are sensitive to initialization and
   may not extrapolate well beyond the training distribution.
4. **Omitted variables**: This analysis uses only three macro series. GDP growth,
   commodity prices, and supply-chain indicators are omitted.
5. **Hallucination prevention**: LLM narrative (if enabled) is constrained to
   pre-computed evidence. All numeric claims are validated against the evidence store.
"""

_REPRODUCIBILITY_TEMPLATE = """\
```bash
# Reproduce this exact report:
python -m eco_report_ai run --config config.yaml --data-source {data_source}
```

| Parameter | Value |
|-----------|-------|
| Config hash | `{config_hash}` |
| Random seed | {seed} |
| Data source | {data_source} |
| Run timestamp | {timestamp} |
| Python | {python_version} |
| pandas | {pandas_version} |
| statsmodels | {statsmodels_version} |
| scikit-learn | {sklearn_version} |
| torch | {torch_version} |
"""


class NarrativeGenerator:
    """Generates structured narrative sections for the economic report.

    Implements the hybrid NLG strategy: deterministic templates first,
    optional LLM enhancement if OPENAI_API_KEY is set.

    Args:
        llm_model: OpenAI model identifier.
        temperature: LLM sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens for LLM response.
        enabled_if_key: Auto-enable LLM when API key is present.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        enabled_if_key: bool = True,
    ) -> None:
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._use_llm = enabled_if_key and bool(os.getenv("OPENAI_API_KEY", ""))
        if self._use_llm:
            logger.info("LLM narrative enhancement enabled (model=%s).", llm_model)
        else:
            logger.info("Using deterministic template NLG (OPENAI_API_KEY not set).")

    def generate_executive_summary(self, evidence: dict[str, Any]) -> str:
        """Generate the Executive Summary section.

        Args:
            evidence: Pre-computed evidence store dict.

        Returns:
            Formatted Markdown string for the executive summary.
        """
        text = _EXEC_SUMMARY_TEMPLATE.format(**_safe_defaults(evidence))
        if self._use_llm:
            text = self._enhance_with_llm(
                section="executive_summary",
                template_text=text,
                evidence=evidence,
            )
        return text

    def generate_data_overview(self, evidence: dict[str, Any]) -> str:
        """Generate the Data Overview section."""
        return _DATA_OVERVIEW_TEMPLATE.format(**_safe_defaults(evidence))

    def generate_econometric_findings(self, evidence: dict[str, Any]) -> str:
        """Generate the Econometric Findings section."""
        # Build coefficient table rows
        coeffs = evidence.get("ols_coefficients", {})
        std_errors = evidence.get("ols_std_errors", {})
        p_values = evidence.get("ols_p_values", {})
        t_stats = evidence.get("ols_t_stats", {})

        rows: list[str] = []
        for var in coeffs:
            coef = coeffs.get(var, float("nan"))
            se = std_errors.get(var, float("nan"))
            tv = t_stats.get(var, float("nan"))
            pv = p_values.get(var, float("nan"))
            sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
            rows.append(
                f"| {var} | {coef:.4f}{sig} | {se:.4f} | {tv:.2f} | {pv:.4f} |"
            )

        coeff_table = "\n".join(rows) if rows else "| — | — | — | — | — |"

        fedfunds_coef = coeffs.get("fedfunds", coeffs.get("FEDFUNDS", 0.0))
        unrate_coef = coeffs.get("unrate", coeffs.get("UNRATE", 0.0))

        # Fed funds rate direction
        ff_direction = "decrease" if fedfunds_coef < 0 else "increase"
        ff_sentence = (
            f"A 1 percentage-point increase in the Federal Funds Rate is associated with "
            f"a **{fedfunds_coef:.4f}** pp {ff_direction} in CPI YoY (all else equal), "
            f"{'consistent with a contractionary monetary policy channel' if fedfunds_coef < 0 else 'suggesting limited or lagged monetary transmission in this sample'}."
        )

        # Phillips Curve direction — sign matters
        if unrate_coef < 0:
            pc_sentence = (
                f"A 1 pp rise in unemployment is associated with a "
                f"**{unrate_coef:.4f}** pp decline in inflation, consistent with the "
                f"standard Phillips Curve (higher unemployment → lower inflation)."
            )
        else:
            pc_sentence = (
                f"A 1 pp rise in unemployment is associated with a "
                f"**{unrate_coef:.4f}** pp increase in inflation — a positive sign that runs "
                f"counter to the standard Phillips Curve. Possible explanations include "
                f"supply-side inflation episodes in the sample (e.g., 2021–2022), omitted "
                f"variables (commodity prices, supply shocks), or the short sample period "
                f"conflating unemployment and inflation cycles."
            )

        interpretation = f"{ff_sentence} {pc_sentence}"

        ev = _safe_defaults(evidence)
        ev["coeff_table"] = coeff_table
        ev["interpretation"] = interpretation

        # Pre-format DW so the template doesn't apply a float format spec to "N/A"
        dw_raw = evidence.get("durbin_watson")
        import math as _math
        ev["durbin_watson_str"] = (
            f"{dw_raw:.3f}" if isinstance(dw_raw, float) and not _math.isnan(dw_raw)
            else "N/A"
        )

        return _ECONOMETRIC_TEMPLATE.format(**ev)

    def generate_forecast_section(self, evidence: dict[str, Any]) -> str:
        """Generate the Forecast Results section."""
        forecast_values = evidence.get("forecast_values", {})

        forecast_rows: list[str] = []
        for period, val in list(forecast_values.items())[:12]:
            forecast_rows.append(f"| {period} | {val:.2f} |")
        forecast_table = "\n".join(forecast_rows) if forecast_rows else "| — | — |"

        # Determine trend description
        fc_list = list(forecast_values.values())
        if len(fc_list) >= 2:
            first, last = fc_list[0], fc_list[-1]
            if last > first + 0.2:
                trend_description = "gradually increase"
            elif last < first - 0.2:
                trend_description = "gradually decline toward target"
            else:
                trend_description = "remain relatively stable"
        else:
            trend_description = "evolve as indicated"

        ev = _safe_defaults(evidence)
        ev["forecast_table"] = forecast_table
        ev["trend_description"] = trend_description
        return _FORECAST_TEMPLATE.format(**ev)

    def generate_risks(self, evidence: dict[str, Any]) -> str:
        """Generate the Risks & Limitations section."""
        return _RISKS_TEMPLATE.format(**_safe_defaults(evidence))

    def generate_reproducibility(self, evidence: dict[str, Any]) -> str:
        """Generate the Reproducibility section."""
        return _REPRODUCIBILITY_TEMPLATE.format(**_safe_defaults(evidence))

    def _enhance_with_llm(
        self,
        section: str,
        template_text: str,
        evidence: dict[str, Any],
    ) -> str:
        """Optionally enhance a section's narrative using OpenAI.

        The LLM receives the template text and evidence store. It is
        instructed to improve phrasing WITHOUT changing any numbers
        or introducing new numeric claims.

        Args:
            section: Section identifier for logging.
            template_text: Deterministic template output.
            evidence: Pre-computed evidence store.

        Returns:
            Enhanced text, or original template_text on any error.
        """
        try:
            from openai import OpenAI

            client = OpenAI()

            import json

            # Remove large/complex objects from evidence for the prompt
            safe_evidence = {
                k: v for k, v in evidence.items()
                if isinstance(v, (int, float, str, bool)) or (isinstance(v, dict) and len(str(v)) < 200)
            }

            system_prompt = (
                "You are an expert macroeconomic analyst writing a professional economic report. "
                "You will receive a draft section and supporting evidence. "
                "Your task: improve the prose quality and flow of the draft. "
                "CRITICAL RULES: "
                "(1) Do NOT change any numeric values. "
                "(2) Do NOT introduce numbers not present in the evidence. "
                "(3) Keep the same Markdown structure (headers, tables, bold). "
                "(4) Output only the improved section text, nothing else."
            )

            user_prompt = (
                f"Evidence store (JSON):\n{json.dumps(safe_evidence, default=str, indent=2)}\n\n"
                f"Draft section (improve prose, keep all numbers):\n{template_text}"
            )

            response = client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            enhanced = response.choices[0].message.content or template_text
            logger.info("LLM enhanced section '%s' (%d tokens used).", section, response.usage.total_tokens if response.usage else 0)
            return enhanced

        except Exception as exc:
            logger.warning("LLM enhancement failed for '%s': %s. Using template.", section, exc)
            return template_text


def _safe_defaults(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of evidence with safe defaults for missing keys.

    This prevents KeyError in format() calls when a metric was not computed.

    Args:
        evidence: Pre-computed evidence store.

    Returns:
        Dict with all required template keys filled.
    """
    defaults: dict[str, Any] = {
        "data_source": "sample CSV",
        "start_date": "2000-01",
        "end_date": "2023-12",
        "n_obs": 0,
        "cpi_yoy_mean": float("nan"),
        "cpi_yoy_std": float("nan"),
        "cpi_yoy_min": float("nan"),
        "cpi_yoy_max": float("nan"),
        "cpi_yoy_latest": float("nan"),
        "unrate_mean": float("nan"),
        "unrate_std": float("nan"),
        "unrate_min": float("nan"),
        "unrate_max": float("nan"),
        "fedfunds_mean": float("nan"),
        "fedfunds_std": float("nan"),
        "fedfunds_min": float("nan"),
        "fedfunds_max": float("nan"),
        "n_models": 0,
        "best_model": "N/A",
        "best_rmse": float("nan"),
        "n_folds": 5,
        "forecast_horizon": 12,
        "forecast_start": "N/A",
        "formula": "N/A",
        "hac_lags": 6,
        "r_squared": float("nan"),
        "adj_r_squared": float("nan"),
        "nobs": 0,
        "durbin_watson": float("nan"),
        "durbin_watson_str": "N/A",
        "config_hash": "N/A",
        "seed": 42,
        "timestamp": "N/A",
        "python_version": "N/A",
        "pandas_version": "N/A",
        "statsmodels_version": "N/A",
        "sklearn_version": "N/A",
        "torch_version": "N/A",
    }
    merged = {**defaults, **evidence}
    # Ensure float NaN values are represented as a string for format()
    for k, v in merged.items():
        if isinstance(v, float) and (v != v):  # NaN check
            merged[k] = "N/A"
    return merged
