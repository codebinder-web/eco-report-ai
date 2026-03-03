# EcoReport AI — Implementation Map

> Maps every blueprint requirement to code module, file, and key functions/classes.  
> Also links each section to course concepts in `docs/course_raw_notes.md`.

---

## Module Map

| Blueprint Requirement | Module / File | Key Functions / Classes |
|----------------------|--------------|------------------------|
| CLI entrypoint | `eco_report_ai/cli.py` | `cli()`, `run()`, `evaluate()`, `report()`, `version()` |
| Pipeline orchestration | `eco_report_ai/pipeline.py` | `EcoReportPipeline`, `run_pipeline()` |
| Config loading & validation | `eco_report_ai/config.py` | `PipelineConfig`, `load_config()` |
| Structured logging | `eco_report_ai/logging_config.py` | `setup_logging()`, `JsonFormatter` |
| FRED data fetch | `eco_report_ai/data/fred_client.py` | `FREDClient`, `fetch_series()`, `fetch_all()` |
| CSV fallback loading | `eco_report_ai/data/loaders.py` | `load_macro_data()`, `load_from_csv()`, `load_from_fred()` |
| DataFrame schema contract | `eco_report_ai/data/schema.py` | `MacroDataSchema`, `validate_dataframe()` |
| Feature engineering | `eco_report_ai/features/build_features.py` | `build_features()`, `add_lags()`, `add_rolling()`, `compute_cpi_yoy()` |
| OLS + HAC regression | `eco_report_ai/models/econometrics.py` | `OLSModel`, `fit()`, `predict()`, `get_summary()` |
| ARIMA/SARIMAX | `eco_report_ai/models/econometrics.py` | `ARIMAModel`, `fit()`, `forecast()` |
| Gradient Boosting | `eco_report_ai/models/ml_baselines.py` | `GradientBoostingModel`, `fit()`, `predict()` |
| LSTM forecaster | `eco_report_ai/models/lstm.py` | `LSTMForecaster`, `LSTMDataset`, `fit()`, `predict()` |
| Model selection policy | `eco_report_ai/models/model_selection.py` | `select_best_model()`, `ModelRegistry` |
| MAE / RMSE / MAPE | `eco_report_ai/evaluation/metrics.py` | `mae()`, `rmse()`, `mape()`, `compute_all_metrics()` |
| Rolling-origin CV | `eco_report_ai/evaluation/backtesting.py` | `RollingOriginCV`, `run_backtest()` |
| Historical + forecast charts | `eco_report_ai/reporting/charts.py` | `plot_historical_series()`, `plot_forecast()`, `plot_model_comparison()` |
| Template / LLM narrative | `eco_report_ai/reporting/nlg.py` | `NarrativeGenerator`, `generate_executive_summary()`, `generate_section()` |
| MD + JSON report writer | `eco_report_ai/reporting/report_writer.py` | `ReportWriter`, `write_markdown()`, `write_json()` |
| Date utilities | `eco_report_ai/utils/dates.py` | `enforce_monthly_freq()`, `parse_date_range()` |
| File I/O utilities | `eco_report_ai/utils/io.py` | `ensure_dir()`, `save_json()`, `load_json()` |

---

## File → Test Coverage Map

| Source File | Test File | Key Test Cases |
|-------------|-----------|---------------|
| `data/loaders.py` | `tests/test_data_loading.py` | CSV load, date index, freq enforcement, missing values |
| `features/build_features.py` | `tests/test_features.py` | Lag creation, rolling windows, CPI YoY, no leakage |
| `evaluation/metrics.py` | `tests/test_metrics.py` | MAE/RMSE/MAPE correctness, zero-handling, perfect prediction |
| `reporting/report_writer.py` | `tests/test_reporting.py` | File creation, non-empty, JSON keys present |
| `reporting/nlg.py` | `tests/test_reporting.py` | Template fallback, section generation |

---

## Course Concepts Used

| Course Concept | Location in Raw Notes | Applied In |
|---------------|----------------------|-----------|
| Hybrid deterministic + probabilistic architecture | Lecture 1, §6, §10 | `pipeline.py`, `nlg.py`, `report_writer.py` |
| LLM as language layer only (not calculator) | Lecture 1, §6; GPT Evolution, §D | `nlg.py` — NarrativeGenerator |
| Hallucination prevention via evidence store | Lecture 1, §7; Challenges, §1 | `report_writer.py` — evidence_store dict |
| Conditional generation (P(X\|C)) | Lecture 1, §4 | `nlg.py` — structured prompts with pre-computed stats |
| Low-temperature inference | How GenAI Works, §Design Implications | `nlg.py` — `temperature=0.3` |
| Structured JSON output enforcement | Control & Conditioning, §8 | `nlg.py` — `response_format={"type": "json_object"}` |
| Prompt engineering as control system | Prompt Engineering, §3 | `nlg.py` — prompt templates |
| Production pipeline: Plan→Compute→Generate→Validate→Render | GenAI Architectures, §Design Decisions | `pipeline.py` |
| Evaluation dimensions: Quality × Diversity × Speed | Evaluation, §Core | `backtesting.py`, `metrics.py` |
| Guardrails surrounding the model | Challenges, §6 | `report_writer.py`, `schema.py` |
| Next-token numeric risk | GPT Evolution, §B | `nlg.py` — numeric prohibitions in prompt |
| Scaling insight: seed for reproducibility | Improvement pathways | `config.py`, `lstm.py` |
| Tool use (LLM as orchestrator) | Language Applications, §6 | `pipeline.py` — orchestrates tools |
| Deployment doctrine: audit logging | Challenges, §8 | `logging_config.py` |
| Data quality strategy | Training Data Challenges, §1–5 | `schema.py`, `loaders.py` |
| Statistical rigor: HAC errors | Course §9 (Econometrics-Specific) | `econometrics.py` |
| RLHF insight: structured prompts matter | Enterprise Fine-Tuning, §D | `nlg.py` |

---

## Data Flow Diagram

```
config.yaml
    │
    ▼
load_config() ──► PipelineConfig
                        │
                        ▼
              load_macro_data()
              [FRED → CSV fallback]
                        │
                        ▼
              validate_dataframe() [schema]
                        │
                        ▼
              build_features()
              [lags, rolling, CPI YoY]
                        │
                        ▼
         ┌──────────────┴──────────────┐
         ▼              ▼              ▼
    OLSModel      ARIMAModel     GBModel
    ARIMAModel                   LSTMForecaster
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
              RollingOriginCV.run_backtest()
              [MAE, RMSE, MAPE per fold]
                        │
                        ▼
              select_best_model()
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
       generate forecast    charts.py
                        │
                        ▼
              evidence_store (dict)
                        │
                        ▼
              NarrativeGenerator
              [template NLG or GPT-4o-mini]
                        │
                        ▼
              ReportWriter.write_markdown()
              ReportWriter.write_json()
                        │
                        ▼
              reports/latest_report.md
              reports/latest_report.json
              reports/figures/*.png
```

---

## Directory Structure

```
eco-report-ai/
├── eco_report_ai/           # Main package
│   ├── __init__.py
│   ├── __main__.py          # python -m eco_report_ai
│   ├── cli.py               # Click CLI
│   ├── config.py            # Pydantic config model
│   ├── logging_config.py    # Structured logging setup
│   ├── pipeline.py          # Orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fred_client.py
│   │   ├── loaders.py
│   │   └── schema.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── econometrics.py
│   │   ├── lstm.py
│   │   ├── ml_baselines.py
│   │   └── model_selection.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── backtesting.py
│   │   └── metrics.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── charts.py
│   │   ├── nlg.py
│   │   └── report_writer.py
│   └── utils/
│       ├── __init__.py
│       ├── dates.py
│       └── io.py
├── data/
│   └── sample_macro.csv     # Synthetic monthly macro data
├── reports/
│   └── figures/             # Chart output directory
├── docs/
│   ├── course_raw_notes.md
│   ├── course_extracted_blueprint.md
│   ├── design_decisions.md
│   ├── implementation_map.md
│   └── methodology.md
├── tests/
│   ├── conftest.py
│   ├── test_data_loading.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_reporting.py
├── config.yaml
├── .env.example
├── pyproject.toml
└── README.md
```
