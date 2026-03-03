# EcoReport AI — Course-Extracted Blueprint

> **Source**: `docs/course_raw_notes.md` — Lectures 1–N on Generative AI, LLM Architecture, and Production System Design  
> **Distilled by**: Senior Econometrics + ML + GenAI Engineer  
> **Purpose**: Authoritative specification for the EcoReport AI implementation

---

## 1. Problem Statement & User Story

### Problem Statement
Macroeconomic analysis requires transforming raw time-series data (CPI, unemployment, interest rates) into interpretable forecasts and structured reports. This process today is:

- **Manual and slow**: economists spend hours formatting charts and writing narratives
- **Error-prone**: hand-copied statistics introduced inconsistencies
- **Non-reproducible**: ad-hoc scripts with no logging or version control
- **LLM-unsafe by default**: naive use of ChatGPT for report writing leads to numeric hallucination

### Core Insight from Course
> *"Language model ≠ econometric engine."* (Lecture 1)  
> *"Numerical hallucination is a structural property of generative modeling."* (Lecture 1)

The solution is a **hybrid system**: deterministic statistical computation + constrained LLM narrative layer.

### User Stories
| Actor | Goal | Acceptance Criterion |
|-------|------|----------------------|
| Economist | Run the pipeline and get a professional report | `python -m eco_report_ai run` produces `reports/latest_report.md` |
| Data Scientist | Evaluate multiple forecasting models | Backtesting results appear in JSON with MAE/RMSE/MAPE |
| Developer | Add a new FRED series | Edit `config.yaml` → re-run pipeline |
| Auditor | Verify all numbers in the report | Every stat links back to `evidence_store` in JSON |
| Student | Run without any API keys | Falls back to sample CSV; report still generated |

---

## 2. Inputs / Outputs

### Inputs
| Source | Series | Description | Fallback |
|--------|--------|-------------|---------|
| FRED API | `CPIAUCSL` | Consumer Price Index (all urban) | `data/sample_macro.csv` |
| FRED API | `UNRATE` | Civilian Unemployment Rate | `data/sample_macro.csv` |
| FRED API | `FEDFUNDS` | Effective Federal Funds Rate | `data/sample_macro.csv` |
| Config | `config.yaml` | All tunable parameters | Defaults baked in |
| Env | `FRED_API_KEY` | FRED data access (optional) | Local CSV |
| Env | `OPENAI_API_KEY` | LLM narrative enhancement (optional) | Template NLG |

### Outputs
| File | Description |
|------|-------------|
| `reports/latest_report.md` | Professional Markdown economic report |
| `reports/latest_report.json` | All numeric evidence + metadata (machine-readable) |
| `reports/figures/historical_series.png` | Historical multi-series line chart |
| `reports/figures/forecast_plot.png` | Best model forecast with confidence intervals |
| `reports/figures/model_comparison.png` | Bar chart comparing MAE/RMSE across models |
| `logs/eco_report_ai.log` | Structured log of entire run |

---

## 3. Data Sources & Fallback Plan

### Primary: FRED API
- Requires `FRED_API_KEY` environment variable
- Library: `fredapi` Python package
- Series fetched at monthly frequency
- Date range: configurable via `config.yaml` (default: 2000-01-01 to present)

### Fallback: Local Sample CSV
- File: `data/sample_macro.csv`
- Contains synthetic but realistic monthly macro data (2000–2023)
- Columns: `date, CPIAUCSL, UNRATE, FEDFUNDS`
- Loaded automatically when `FRED_API_KEY` is absent or FRED fetch fails

### Frequency Alignment
- All series resampled to monthly frequency (`MS` — month start)
- Missing values handled via **time-aware forward-fill** then linear interpolation
- Documented in `docs/methodology.md`

---

## 4. Preprocessing Rules

| Step | Rule | Rationale |
|------|------|-----------|
| Frequency enforcement | Resample all series to `MS` | Ensure consistent time index |
| Missing values (< 5%) | Forward-fill (`ffill`) then `bfill` | Time-series safe; avoids future-data leakage |
| Missing values (> 5%) | Linear interpolation | Documented; flags raised in report |
| Outlier detection | IQR method; flag only, not remove | Preserve economic signal; note in report |
| Target construction | CPI YoY (%) = `(CPI_t / CPI_{t-12} - 1) * 100` | Standard macroeconomic practice |
| Train/test split | Chronological; no shuffle | Prevent data leakage |
| Scaling | `MinMaxScaler` for LSTM only | Tree/OLS models use unscaled features |
| Differencing (optional) | Configurable via `config.yaml` | Stationarity enforcement for ARIMA |

---

## 5. Feature Engineering Plan

### Base Features
| Feature | Description |
|---------|-------------|
| `cpi_yoy` | CPI Year-over-Year % change (target variable) |
| `unrate` | Unemployment rate (level) |
| `fedfunds` | Federal funds rate (level) |

### Lag Features
- `cpi_yoy_lag_1` through `cpi_yoy_lag_{max_lag}` (default: 6 months)
- `unrate_lag_1` through `unrate_lag_{max_lag}`
- `fedfunds_lag_1` through `fedfunds_lag_{max_lag}`
- `max_lag` configurable in `config.yaml`

### Rolling Window Features
- Rolling mean: windows from `config.yaml` (default: `[3, 6, 12]` months)
- Rolling std: same windows
- Applied to: CPI YoY, UNRATE, FEDFUNDS

### Optional Differencing
- First-difference of all variables if `features.use_differencing: true`
- Produces stationarity-ready features for ARIMA

### Feature Selection
- Features used by each model documented in `reports/latest_report.json`
- OLS uses a subset defined in `config.yaml`
- ML/LSTM use all engineered features

---

## 6. Modeling Plan

### 6.1 OLS with HAC Robust Standard Errors
- **Library**: `statsmodels`
- **Formula**: `cpi_yoy ~ fedfunds + unrate + cpi_yoy_lag_1 + cpi_yoy_lag_3`
- **HAC errors**: Newey-West (lags = 6) to correct for heteroskedasticity and autocorrelation
- **Outputs**: coefficients, p-values, R², adjusted R², HAC std errors, residual diagnostics
- **Why OLS**: interpretability, classic econometrics baseline, produces causal-framing for report

### 6.2 ARIMA / SARIMAX
- **Library**: `statsmodels`
- **Strategy**: auto-grid search over `(p,d,q)` with `d` from ADF test result
- **SARIMAX**: optional seasonal component `(P,D,Q,s=12)` for monthly CPI
- **Outputs**: forecast + 95% confidence intervals, AIC/BIC
- **Why ARIMA**: gold standard univariate time-series forecasting

### 6.3 ML Baseline (Gradient Boosting)
- **Library**: `scikit-learn`
- **Model**: `GradientBoostingRegressor` (default) or `RandomForestRegressor` (configurable)
- **Hyperparameters**: grid search via `RandomizedSearchCV` on training folds only
- **Outputs**: feature importances, predictions
- **Why GB**: non-linear relationships, handles mixed-frequency signals

### 6.4 LSTM (Deep Learning)
- **Library**: `PyTorch` (CPU/GPU)
- **Architecture**: 2-layer LSTM → Dense(1)
- **Input**: sliding window of length `config.lstm.lookback` (default: 12 months)
- **Training**: early stopping on validation loss, patience = 10 epochs
- **Seeds**: `torch.manual_seed`, `numpy.random.seed`, Python `random.seed` — all set from `config.seed`
- **Device**: auto-detect CUDA, fallback CPU
- **Outputs**: scaled predictions (inverse-transformed before evaluation)
- **Why LSTM**: captures long-range temporal dependencies; demonstrates DL in time series

---

## 7. Backtesting & Evaluation Methodology

### Rolling-Origin Cross-Validation
```
Train[0..T-h]  →  Predict[T-h+1..T]
Train[0..T-h+s] →  Predict[T-h+s+1..T+s]
...  (n_folds iterations)
```
- **Key property**: training window always ends before test window begins (no leakage)
- **n_folds**: configurable (default: 5)
- **Horizon**: configurable (default: 6 months)
- **Never shuffle**

### Metrics (computed per fold, then aggregated)
| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | `mean(|y_true - y_pred|)` | Interpretable in CPI units |
| RMSE | `sqrt(mean((y_true-y_pred)²))` | Penalizes large errors |
| MAPE | `mean(|y_true-y_pred|/max(|y_true|, ε)) * 100` | Safe division with ε=1e-8 |

### Reporting
- Per-model: mean ± std across folds for each metric
- Final comparison table in report

---

## 8. Model Selection Policy

```
best_model = argmin(mean_RMSE_across_folds)
```

Tiebreaker order: MAE → MAPE → model preference (OLS > ARIMA > GB > LSTM for interpretability).

The winning model is used for:
- Final out-of-sample forecast (configurable horizon, default 12 months ahead)
- Report "Forecast Results" section

All models' metrics are reported in the comparison section.

---

## 9. Report Structure & Visuals

### Markdown Report (`reports/latest_report.md`)
```
# EcoReport AI — Economic Analysis Report
## 1. Executive Summary
## 2. Data Overview
   - Series coverage (start/end dates, obs count)
   - Missing data summary
   - Descriptive statistics table
## 3. Econometric Findings
   - OLS regression table (coefficients, HAC std errors, p-values)
   - Model interpretation narrative
## 4. Forecast Results
   - Best model identification
   - Forecast table (horizon × period)
   - Confidence interval discussion
## 5. Model Comparison
   - Table: Model | MAE | RMSE | MAPE
## 6. Risks & Limitations
## 7. Reproducibility
   - Exact CLI command used
   - Config hash
   - Library versions
## Appendix: Figures
```

### Charts
| File | Description |
|------|-------------|
| `historical_series.png` | Multi-line plot: CPI YoY, UNRATE, FEDFUNDS (2000–present) |
| `forecast_plot.png` | Historical + best model forecast + 95% CI shading |
| `model_comparison.png` | Grouped bar chart: MAE/RMSE/MAPE per model |

---

## 10. CLI Specification

```bash
# Full pipeline run (auto-detects FRED key or falls back to CSV)
python -m eco_report_ai run

# Use sample data explicitly
python -m eco_report_ai run --data-source sample

# Use FRED with specific start date
python -m eco_report_ai run --data-source fred --start-date 2005-01-01

# Run with a specific config file
python -m eco_report_ai run --config my_config.yaml

# Just show what would run (dry run)
python -m eco_report_ai run --dry-run

# Evaluate all models only (no report)
python -m eco_report_ai evaluate

# Generate report from existing results
python -m eco_report_ai report

# Show version info
python -m eco_report_ai version
```

### CLI Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--data-source` | `auto` | `auto`, `fred`, `sample` |
| `--start-date` | Config value | Data start date |
| `--end-date` | Config value | Data end date |
| `--output-dir` | `reports/` | Output directory |
| `--dry-run` | `false` | Print plan without executing |
| `--log-level` | `INFO` | Log verbosity |

---

## 11. Testing Plan

### Unit Tests
| File | What it tests |
|------|--------------|
| `tests/test_data_loading.py` | CSV loading, date parsing, frequency enforcement, missing value handling |
| `tests/test_features.py` | Lag creation, rolling window computation, target variable calculation |
| `tests/test_metrics.py` | MAE/RMSE/MAPE computation, edge cases (zeros, perfect prediction) |
| `tests/test_reporting.py` | Report section generation, JSON serialization, figure file creation |

### Integration Tests
- End-to-end pipeline test using sample CSV (no API keys required)
- Asserts report files exist and are non-empty
- Asserts JSON has required keys

### Test Fixtures
- `tests/conftest.py` with shared fixtures (sample DataFrame, mock config)

### Coverage Target
- ≥ 80% line coverage on all source modules

---

## 12. Future Extensions

| Extension | Description | Effort |
|-----------|-------------|--------|
| RAG Enhancement | Embed historical reports; retrieve context for LLM | Medium |
| Domain Fine-Tuning | Fine-tune LLM on Fed/ECB/BOE publications | High |
| More FRED Series | Add GDP, PCE, yield curve, ISM | Low |
| Vector Autoregression (VAR) | Multivariate econometric model | Medium |
| Bayesian Structural Time Series | Uncertainty quantification | High |
| Dashboard UI | Streamlit/Dash frontend for report exploration | Medium |
| MLflow Integration | Experiment tracking, model registry | Medium |
| Airflow/Prefect DAG | Scheduled monthly pipeline runs | Medium |
| LLM Validator | Regex + embedding-based numeric hallucination checker | Medium |

---

*Blueprint distilled from `docs/course_raw_notes.md` — all design decisions trace back to course principles.*
