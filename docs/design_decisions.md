# EcoReport AI — Architecture & Design Decisions (ADR)

> **Document type**: Architecture Decision Record (ADR)  
> **Principles source**: `docs/course_raw_notes.md` + industry best practices  
> **Last updated**: 2026-03-02

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EcoReport AI Pipeline                         │
│                                                                        │
│  CLI (click)                                                           │
│      │                                                                 │
│      ▼                                                                 │
│  Pipeline Orchestrator (pipeline.py)                                   │
│      │                                                                 │
│      ├─► Data Layer ──────────────────────────────────────────────┐   │
│      │     fred_client.py  │  loaders.py  │  schema.py            │   │
│      │                                                             │   │
│      ├─► Feature Engineering ─────────────────────────────────────┤   │
│      │     build_features.py                                       │   │
│      │                                                             │   │
│      ├─► Models ──────────────────────────────────────────────────┤   │
│      │     econometrics.py  │  ml_baselines.py  │  lstm.py        │   │
│      │                                                             │   │
│      ├─► Evaluation ──────────────────────────────────────────────┤   │
│      │     metrics.py  │  backtesting.py                          │   │
│      │                                                             │   │
│      ├─► Model Selection ─────────────────────────────────────────┤   │
│      │     model_selection.py                                      │   │
│      │                                                             │   │
│      └─► Reporting Layer ─────────────────────────────────────────┘   │
│            charts.py  │  nlg.py  │  report_writer.py                  │
│                │                        │                              │
│                ▼                        ▼                              │
│         reports/figures/        reports/latest_report.md              │
│                                  reports/latest_report.json           │
└──────────────────────────────────────────────────────────────────────┘
```

### Layered Architecture Rationale
The system is split into six distinct layers. Each layer has a single responsibility, enabling independent testing, replacement, and extension. This directly implements the architectural separation taught in the course:

> *"Statistical Engine: compute metrics deterministically. LLM: generate structured narrative. Validator: enforce citation & numeric consistency."* — Course Lecture 1

---

## 2. Key Decisions

### ADR-001: Hybrid Deterministic + Probabilistic Architecture

**Status**: Accepted  
**Context**: LLMs generate plausible but potentially false numeric content. Economic reports require exact, reproducible numbers.  
**Decision**: All statistical computation is performed by the Python analytics layer. The LLM (if present) receives only pre-computed evidence and is restricted to narrative generation.  
**Alternatives considered**:
- *LLM-only*: Rejected — hallucination risk is a structural property of autoregressive models
- *No LLM*: Rejected — reduces report quality; offline template NLG is the fallback  
**Consequences**: System is reliable without an API key. LLM adds polish, not substance.

---

### ADR-002: Config-Driven Everything

**Status**: Accepted  
**Context**: Hard-coded parameters make experimentation difficult and are a code smell.  
**Decision**: All tunable values (lags, windows, model params, series names, date ranges, seed) live in `config.yaml`. Loaded via `config.py` using `pydantic` for validation.  
**Alternatives**: Environment variables only (rejected — too fragile), argparse only (rejected — not persistent).  
**Consequences**: One-line changes to `config.yaml` reconfigure the entire pipeline.

---

### ADR-003: FRED-First with Local CSV Fallback

**Status**: Accepted  
**Context**: Real macro data requires a FRED API key. The system must run without one.  
**Decision**: `loaders.py` attempts FRED → catches any exception → falls back to `data/sample_macro.csv`.  
**Alternatives**: Mock data only (rejected — limits realism); require FRED key (rejected — violates "no API key" requirement).  
**Consequences**: Seamless offline development and portfolio demos.

---

### ADR-004: Rolling-Origin Backtesting (No Shuffle)

**Status**: Accepted  
**Context**: Standard cross-validation shuffles data, creating data leakage in time series (using future data to predict the past).  
**Decision**: Implement rolling-origin CV. Training window always precedes test window. No shuffling.  
**Alternatives**: Walk-forward validation (similar but simpler; considered equivalent here), expanding window (chosen as default — uses all available history).  
**Course principle**: *"Never shuffle"* for time-series evaluation.  
**Consequences**: Leakage-free evaluation. Metrics are trustworthy.

---

### ADR-005: Model Suite Selection

**Status**: Accepted  
**Context**: Need breadth (classical → ML → DL) without over-engineering.  
**Decision**:
| Model | Why included | Why this model |
|-------|-------------|----------------|
| OLS + HAC | Interpretability, econometric baseline | Standard for macro analysis |
| ARIMA/SARIMAX | Univariate TS gold standard | Captures autocorrelation structure |
| Gradient Boosting | Non-linear ML baseline | Handles feature interactions |
| LSTM | Deep learning demonstration | Captures long-range dependencies |

**Why NOT others**:
- *Prophet*: Good but Facebook-specific; requires extra install; not econometric tradition
- *XGBoost*: Would be fine; GB is included in scikit-learn, reducing dependencies
- *Transformer for TS*: Over-engineered for monthly macro data at this scale
- *GARCH*: Volatility modeling; less relevant for level/YoY forecasting here

---

### ADR-006: MAPE with Safe Division

**Status**: Accepted  
**Context**: CPI YoY can theoretically be 0% (deflation) causing division by zero in MAPE.  
**Decision**: `MAPE = mean(|actual - pred| / max(|actual|, ε)) * 100` with `ε = 1e-8`.  
**Alternatives**: Skip MAPE (rejected — widely used), use sMAPE (considered; MAPE is more standard).  
**Consequences**: Numerically stable; documented in methodology.

---

### ADR-007: HAC Robust Standard Errors in OLS

**Status**: Accepted  
**Context**: Macroeconomic time series exhibit autocorrelation and heteroskedasticity. OLS standard errors under these conditions are biased.  
**Decision**: Apply Newey-West HAC correction via `statsmodels` `cov_type='HAC'` with `maxlags=6`.  
**Course alignment**: Econometric rigor is required; plain OLS standard errors would be misleading.  
**Consequences**: Inference (p-values, confidence intervals) is statistically valid.

---

### ADR-008: LLM Usage Policy

**Status**: Accepted  
**Context**: OpenAI API adds cost and dependency. The system must work offline.  
**Decision**:
1. If `OPENAI_API_KEY` is set → call GPT-4o-mini with structured prompt containing pre-computed evidence
2. If key is absent → use deterministic template-based NLG (string formatting with computed values)
3. LLM prompt explicitly prohibits free numeric generation
4. Temperature = 0.3 (low, near-deterministic)
5. Structured JSON output enforced via response format  
**Guardrails implemented**:
- Prompt includes evidence store (all numbers cited by key name)
- No speculative numeric generation allowed in prompt
- Fallback is production-quality (not a placeholder)

---

### ADR-009: Reproducibility Strategy

**Status**: Accepted  
**Context**: ML experiments must be reproducible. Seeds, configs, and versions must be logged.  
**Decision**:
- Global seed from `config.seed` (default: `42`) applied to: Python `random`, `numpy`, `torch`
- All config values logged at pipeline start
- `reports/latest_report.json` includes: config snapshot, library versions, run timestamp
- Deterministic LSTM training via seeded initialization
**Consequences**: Re-running with same config produces identical results (CPU; GPU may vary slightly).

---

### ADR-010: LSTM Architecture Choices

**Status**: Accepted  
**Context**: LSTM must be "production-quality" without being over-engineered for monthly macro data.  
**Decision**:
- 2 LSTM layers (depth without vanishing gradient risk)
- Hidden size: 64 (configurable)
- Dropout: 0.2 (regularization)
- Sliding window lookback: 12 months (1 year of history)
- Early stopping: patience=10 epochs on validation loss
- Batch size: 32
- Max epochs: 200 (early stopping usually triggers earlier)
- Optimizer: Adam with lr=1e-3
- GPU auto-detection; CPU fallback

---

## 3. How We Avoid Leakage in Time Series

| Leakage Type | Prevention |
|-------------|-----------|
| Future data in training | Strict chronological split; rolling-origin CV |
| Scaling on full dataset | Scaler fit only on training fold; applied to test |
| Feature engineering using future values | All lags/rolling windows computed before train/test split |
| LSTM sequence construction | Sliding windows constructed from training portion only |
| Hyperparameter tuning on test set | RandomizedSearchCV applied to training folds only |

---

## 4. Explainability Strategy

| Model | Explainability Method |
|-------|----------------------|
| OLS | Coefficients + HAC p-values directly in report |
| ARIMA | AR/MA coefficients + AIC/BIC in report |
| Gradient Boosting | Feature importance plot (MDI) |
| LSTM | Aggregate prediction + confidence interval; inherently black-box |

Course principle: *"LLM acts as a constrained natural language rendering engine over validated statistical outputs."* The report narrative explains OLS coefficients in plain language using the template/LLM, but all numbers come from the evidence store.

---

## 5. Reproducibility Choices

```yaml
# In config.yaml
seed: 42
data:
  start_date: "2000-01-01"
  fred_series: [CPIAUCSL, UNRATE, FEDFUNDS]
```

Every run logs:
```json
{
  "run_id": "uuid",
  "timestamp": "ISO8601",
  "config_hash": "sha256",
  "library_versions": {...},
  "data_source": "fred|sample",
  "seed": 42
}
```

---

## 6. Data Quality Strategy

| Issue | Detection | Resolution |
|-------|-----------|-----------|
| Missing values | `df.isna().sum()` per column | ffill → bfill → linear interpolation |
| Irregular frequency | Check `pd.infer_freq` | Force resample to `MS` |
| Outliers | IQR-based flagging | Flag in report; do not remove |
| Insufficient history | Assert ≥ 36 months | Raise `DataQualityError` |
| Stationarity | ADF test for ARIMA | Auto-determine `d` parameter |
| Data staleness | Check last date ≥ 1 month ago | Warning logged |

---

## 7. Security & Ethics Notes

| Concern | Mitigation |
|---------|-----------|
| API Key exposure | `.env.example` never contains real keys; `.gitignore` includes `.env` |
| No private data | All data from public FRED API or synthetic CSV |
| LLM prompt injection | Structured prompts; no user input passed to LLM directly |
| Model bias in narrative | Template NLG used for quantitative claims; LLM for phrasing only |
| Overconfident forecasts | Confidence intervals always shown; limitations section mandatory |

---

## 8. "Beyond the Course" MLOps-Like Improvements

| Improvement | Implementation |
|-------------|---------------|
| Structured logging | `logging_config.py` with JSON formatter, file + console handlers |
| Config validation | `pydantic` `BaseSettings` for type-safe config loading |
| Schema validation | `pandera` DataFrameSchema for data quality contracts |
| Evidence store | All computed stats written to `reports/latest_report.json` before narrative generation |
| Separation of concerns | Data / Features / Models / Evaluation / Reporting are independent modules |
| Type hints everywhere | Full type annotations on all public functions |
| Docstrings | Google-style docstrings on all modules, classes, functions |
| Tests | pytest with fixtures; ≥ 80% coverage target |
| `.env.example` | Documents all required/optional environment variables |
| `pyproject.toml` | Modern Python packaging with `[project.scripts]` entry point |

---

## 9. Technology Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Package manager | `pyproject.toml` + pip | Modern, standard |
| Config | `pydantic` + YAML | Type-safe, validated |
| Data | `pandas` + `fredapi` | Industry standard |
| Econometrics | `statsmodels` | OLS, ARIMA, HAC |
| ML | `scikit-learn` | RandomizedSearchCV, GB |
| DL | `PyTorch` | Flexible, GPU support |
| Charts | `matplotlib` + `seaborn` | Reproducible, no frontend needed |
| CLI | `click` | Clean, composable |
| Testing | `pytest` | Standard |
| Logging | `logging` (stdlib) | No extra dependency |
| NLG | Templates + OpenAI optional | Offline-first |

---

*All decisions trace back to course principles in `docs/course_raw_notes.md`.*
