# EcoReport AI

> **Production-grade macroeconomic forecasting and automated report generation.**  
> Combines classical econometrics, machine learning, and deep learning with constrained generative AI — running end-to-end without any API keys.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen.svg)](#testing)
[![Models](https://img.shields.io/badge/models-OLS%20%7C%20ARIMA%20%7C%20GB%20%7C%20LSTM-orange.svg)](#models)
[![No API Key Required](https://img.shields.io/badge/offline-yes-green.svg)](#quickstart)

---

## What is EcoReport AI?

EcoReport AI is a fully automated pipeline that:

1. **Loads** macroeconomic time series (FRED API or bundled sample data)
2. **Engineers** lag, rolling mean/std, and YoY features with zero data leakage
3. **Fits** four model families: OLS + HAC errors, ARIMA/SARIMAX, Gradient Boosting, LSTM
4. **Evaluates** via rolling-origin cross-validation (no data shuffling)
5. **Selects** the best model by minimum RMSE across folds
6. **Generates** a professional economic report (Markdown + JSON) with charts

The LLM layer is **optional** and fully offline-safe. Every number in the report is computed deterministically — the LLM (if present) only writes prose, never numbers.

> *"Language model ≠ econometric engine."* — Core design principle from the course

---

## From Course → Blueprint → Build

This project was built by rigorously distilling course principles into design decisions:

| Layer | Document | Key Insight Applied |
|-------|---------|-------------------|
| Architecture | [`docs/course_extracted_blueprint.md`](docs/course_extracted_blueprint.md) | Hybrid deterministic + probabilistic system |
| Design | [`docs/design_decisions.md`](docs/design_decisions.md) | LLM as language layer only; evidence-grounded NLG |
| Methodology | [`docs/methodology.md`](docs/methodology.md) | HAC errors, rolling-origin CV, MAPE safe division |
| Code Map | [`docs/implementation_map.md`](docs/implementation_map.md) | Blueprint requirement → module → function |

The course emphasizes: *"Numerical hallucination is a structural property of generative modeling."*  
EcoReport AI prevents this by building a **pre-computed evidence store** that every narrative section must cite.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EcoReport AI Pipeline                         │
│                                                                        │
│  CLI (click) ──► Pipeline Orchestrator (pipeline.py)                  │
│                        │                                               │
│    ┌───────────────────┼───────────────────────────┐                  │
│    │                   │                           │                   │
│    ▼                   ▼                           ▼                   │
│  Data Layer      Feature Engineering         Reporting Layer           │
│  (FRED/CSV)      (lags, rolling, YoY)       (charts, NLG, writer)     │
│    │                   │                           ▲                   │
│    └───────────────────┼───────────────────────────┘                  │
│                        │                                               │
│              ┌─────────┼─────────┐                                    │
│              ▼         ▼         ▼                                    │
│            OLS      ARIMA     GB/LSTM  ──► Backtesting (rolling CV)   │
│              └─────────┼─────────┘         │                          │
│                        ▼                   ▼                          │
│                Evidence Store ◄── Model Selection                      │
│                        │                                               │
│          ┌─────────────┴────────────┐                                 │
│          ▼                          ▼                                  │
│   Template NLG              GPT-4o-mini (optional)                    │
│          └─────────────┬────────────┘                                 │
│                        ▼                                               │
│               reports/latest_report.md                                │
│               reports/latest_report.json                              │
│               reports/figures/*.png                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
```bash
pip install -e ".[dev]"
```

### Run Without Any API Keys (uses sample data)
```bash
python -m eco_report_ai run --data-source sample
```
Output:
- `reports/latest_report.md` — Professional economic report
- `reports/latest_report.json` — All numeric evidence (machine-readable)
- `reports/figures/historical_series.png`
- `reports/figures/forecast_plot.png`
- `reports/figures/model_comparison.png`

### Run With FRED Real Data
```bash
# Copy .env.example to .env and add your FRED key
cp .env.example .env
# Edit .env: FRED_API_KEY=your_key_here

python -m eco_report_ai run --data-source fred
```

### Run With OpenAI Narrative Enhancement
```bash
# Add to .env: OPENAI_API_KEY=your_key_here
python -m eco_report_ai run  # auto-detects key and enhances narrative
```

### Auto Mode (FRED if available, else sample)
```bash
python -m eco_report_ai run
```

### Other Commands
```bash
# Evaluate models only (no report)
python -m eco_report_ai evaluate

# Re-generate report from existing JSON (no re-run)
python -m eco_report_ai report

# Show version and dependency info
python -m eco_report_ai version

# Dry run (print plan, don't execute)
python -m eco_report_ai run --dry-run
```

---

## Example Report Snippet

```markdown
# EcoReport AI — Economic Analysis Report

## 1. Executive Summary

CPI inflation (year-over-year) averaged **3.17%** over the sample period,
with a range of -0.50% to 9.12%. The most recent observation stands at **3.91%**.

Among 4 models evaluated via rolling-origin cross-validation (5 folds),
the **OLS** achieved the lowest RMSE of **0.1595** percentage points.

## 3. Econometric Findings

| Variable          | Coefficient | Std Error (HAC) | t-stat | p-value |
|-------------------|-------------|-----------------|--------|---------|
| fedfunds          | -0.1234***  | 0.0412          | -2.99  | 0.003   |
| unrate            |  0.0891**   | 0.0361          |  2.47  | 0.014   |
| cpi_yoy_lag_1     |  0.8823***  | 0.0189          | 46.7   | 0.000   |

R² = 0.9918, Adjusted R² = 0.9917
```

---

## How to Add New FRED Series

1. Edit `config.yaml`:
```yaml
data:
  fred_series:
    - CPIAUCSL
    - UNRATE
    - FEDFUNDS
    - GDPC1       # Add real GDP
    - PCEPI       # Add PCE deflator
```

2. Update the `sample_macro.csv` or let the pipeline fetch from FRED.

3. Update the OLS formula in `config.yaml` if desired:
```yaml
models:
  ols:
    formula: "cpi_yoy ~ fedfunds + unrate + cpi_yoy_lag_1 + cpi_yoy_lag_3 + gdp_lag_1"
```

4. Re-run: `python -m eco_report_ai run`

---

## Models

| Model | Library | Key Features |
|-------|---------|-------------|
| OLS + HAC | `statsmodels` | Interpretable coefficients; Newey-West robust errors |
| ARIMA/SARIMAX | `statsmodels` | AIC grid search; ADF stationarity test; CI forecast |
| Gradient Boosting | `scikit-learn` | RandomizedSearchCV; TimeSeriesSplit; feature importances |
| LSTM | `PyTorch` | 2-layer; early stopping; deterministic seeds; GPU support |

---

## Project Structure

```
eco-report-ai/
├── eco_report_ai/           # Main package
│   ├── cli.py               # Click CLI
│   ├── pipeline.py          # Orchestrator
│   ├── config.py            # Pydantic config
│   ├── data/                # FRED client + CSV loader + schema
│   ├── features/            # Lag + rolling + YoY engineering
│   ├── models/              # OLS, ARIMA, GB, LSTM
│   ├── evaluation/          # Metrics + rolling-origin CV
│   └── reporting/           # Charts + NLG + report writer
├── data/
│   └── sample_macro.csv     # 288 months synthetic macro data
├── reports/
│   ├── latest_report.md     # Generated report
│   ├── latest_report.json   # Evidence store (all metrics)
│   └── figures/             # PNG charts
├── docs/
│   ├── course_raw_notes.md
│   ├── course_extracted_blueprint.md
│   ├── design_decisions.md
│   ├── implementation_map.md
│   └── methodology.md
├── tests/                   # 60 pytest tests
├── config.yaml              # All pipeline parameters
├── .env.example             # API key template
└── pyproject.toml           # Modern packaging
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=eco_report_ai --cov-report=term-missing
```

60 tests covering: data loading, feature engineering, metrics, and report generation.

---

## Configuration Reference

All settings are in `config.yaml`. Key parameters:

```yaml
seed: 42                     # Global reproducibility seed
data:
  fred_series: [CPIAUCSL, UNRATE, FEDFUNDS]
  start_date: "2000-01-01"
  source: "auto"             # auto | fred | sample
features:
  max_lag: 6                 # Lag features per series
  rolling_windows: [3, 6, 12]
models:
  lstm:
    lookback: 12             # LSTM sliding window
    patience: 10             # Early stopping
evaluation:
  n_folds: 5                 # Rolling-origin folds
  forecast_horizon: 6        # Months per fold
  final_forecast_horizon: 12 # Report forecast length
```

---

## Design Principles

From `docs/design_decisions.md`:

1. **Hybrid system**: deterministic analytics + constrained LLM narrative
2. **Evidence store**: all numbers computed first; LLM only describes
3. **No data leakage**: rolling-origin CV, scaler fit on train only
4. **HAC robustness**: Newey-West standard errors for autocorrelated residuals
5. **Offline-first**: full pipeline runs without any API keys
6. **Config-driven**: every tunable parameter in `config.yaml`
7. **Reproducible**: global seed, config hash, library versions in JSON output

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by combining rigorous econometrics with production ML engineering and responsible use of generative AI.*
md…]()
