"""Configuration loading and validation for EcoReport AI.

All pipeline parameters live in config.yaml and are parsed here into
Pydantic models. This gives us type safety, default values, and clear
documentation of every knob in the system.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ── Sub-models ────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Data source configuration."""

    fred_series: list[str] = Field(default=["CPIAUCSL", "UNRATE", "FEDFUNDS"])
    start_date: str = Field(default="2000-01-01")
    end_date: str | None = Field(default=None)
    sample_csv_path: str = Field(default="data/sample_macro.csv")
    source: Literal["auto", "fred", "sample"] = Field(default="auto")

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, v: str) -> str:
        """Ensure start_date is a valid ISO date string."""
        import datetime

        datetime.date.fromisoformat(v)
        return v


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    max_lag: int = Field(default=6, ge=1, le=24)
    rolling_windows: list[int] = Field(default=[3, 6, 12])
    use_differencing: bool = Field(default=False)
    target_column: str = Field(default="cpi_yoy")


class OLSConfig(BaseModel):
    """OLS model configuration."""

    enabled: bool = Field(default=True)
    formula: str = Field(
        default="cpi_yoy ~ fedfunds + unrate + cpi_yoy_lag_1 + cpi_yoy_lag_3"
    )
    hac_lags: int = Field(default=6, ge=0)


class ARIMAConfig(BaseModel):
    """ARIMA/SARIMAX configuration."""

    enabled: bool = Field(default=True)
    max_p: int = Field(default=3, ge=0)
    max_q: int = Field(default=2, ge=0)
    seasonal: bool = Field(default=False)
    seasonal_period: int = Field(default=12)


class GradientBoostingConfig(BaseModel):
    """Gradient Boosting / Random Forest configuration."""

    enabled: bool = Field(default=True)
    variant: Literal["gradient_boosting", "random_forest"] = Field(
        default="gradient_boosting"
    )
    n_iter_search: int = Field(default=20, ge=1)
    cv_folds: int = Field(default=3, ge=2)


class LSTMConfig(BaseModel):
    """LSTM model configuration."""

    enabled: bool = Field(default=True)
    lookback: int = Field(default=12, ge=1)
    hidden_size: int = Field(default=64, ge=8)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=0.8)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=200, ge=1)
    patience: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    val_split: float = Field(default=0.15, gt=0, lt=1)


class ModelsConfig(BaseModel):
    """All model sub-configurations."""

    ols: OLSConfig = Field(default_factory=OLSConfig)
    arima: ARIMAConfig = Field(default_factory=ARIMAConfig)
    gradient_boosting: GradientBoostingConfig = Field(
        default_factory=GradientBoostingConfig
    )
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)


class EvaluationConfig(BaseModel):
    """Backtesting and evaluation configuration."""

    n_folds: int = Field(default=5, ge=2)
    forecast_horizon: int = Field(default=6, ge=1)
    final_forecast_horizon: int = Field(default=12, ge=1)


class ReportingConfig(BaseModel):
    """Report output configuration."""

    output_dir: str = Field(default="reports")
    figures_dir: str = Field(default="reports/figures")
    report_filename: str = Field(default="latest_report")
    figure_dpi: int = Field(default=150, ge=72)
    figure_style: str = Field(default="seaborn-v0_8-whitegrid")


class LLMConfig(BaseModel):
    """Optional LLM narrative generation configuration."""

    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100)
    enabled_if_key: bool = Field(default=True)


class PipelineConfig(BaseModel):
    """Root configuration model for the entire pipeline."""

    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def hash(self) -> str:
        """Return a SHA-256 hash of the serialized config for reproducibility logging.

        Returns:
            8-character hex digest of the config JSON.
        """
        serialized = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:8]


# ── Public API ────────────────────────────────────────────────────────────────


def load_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Falls back to all defaults if the file does not exist.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A validated PipelineConfig instance.

    Raises:
        yaml.YAMLError: If the file is malformed YAML.
        pydantic.ValidationError: If a config value fails validation.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found — using defaults.", path)
        return PipelineConfig()

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    config = PipelineConfig(**raw)
    logger.info("Config loaded from %s (hash=%s)", path, config.hash())
    return config
