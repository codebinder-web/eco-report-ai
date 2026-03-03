"""Chart generation for EcoReport AI.

Produces publication-quality matplotlib figures:
- Historical multi-series line chart
- Forecast plot with confidence intervals
- Model comparison bar chart

All figures are saved to the configured figures directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#8338EC"]


def _setup_style(style: str) -> None:
    """Apply matplotlib style safely, falling back gracefully."""
    try:
        plt.style.use(style)
    except OSError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass


def plot_historical_series(
    df: pd.DataFrame,
    columns: list[str],
    labels: Optional[dict[str, str]] = None,
    figures_dir: str | Path = "reports/figures",
    dpi: int = 150,
    style: str = "seaborn-v0_8-whitegrid",
) -> str:
    """Plot historical macroeconomic series as a multi-panel figure.

    Args:
        df: DataFrame with DatetimeIndex and macro columns.
        columns: Columns to plot (one panel per column).
        labels: Optional display labels mapping col → label.
        figures_dir: Directory to save the figure.
        dpi: Figure resolution.
        style: Matplotlib style name.

    Returns:
        Path to the saved figure file.
    """
    _setup_style(style)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    labels = labels or {}
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col, color in zip(axes, columns, _PALETTE):
        if col not in df.columns:
            logger.warning("Column '%s' not in DataFrame for historical plot.", col)
            continue
        ax.plot(df.index, df[col], color=color, linewidth=1.8, alpha=0.9)
        ax.set_ylabel(labels.get(col, col), fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle("Macroeconomic Indicators — Historical Series", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path(figures_dir) / "historical_series.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved historical series chart: %s", out_path)
    return str(out_path)


def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    conf_lower: Optional[pd.Series] = None,
    conf_upper: Optional[pd.Series] = None,
    model_name: str = "Best Model",
    figures_dir: str | Path = "reports/figures",
    dpi: int = 150,
    style: str = "seaborn-v0_8-whitegrid",
) -> str:
    """Plot historical target series + model forecast with optional CI shading.

    Args:
        historical: Historical CPI YoY series.
        forecast: Forecasted values with future DatetimeIndex.
        conf_lower: Lower confidence interval (optional).
        conf_upper: Upper confidence interval (optional).
        model_name: Label for the forecast series.
        figures_dir: Directory to save the figure.
        dpi: Figure resolution.
        style: Matplotlib style name.

    Returns:
        Path to the saved figure file.
    """
    _setup_style(style)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Show last 3 years of history for clarity
    recent_history = historical.iloc[-36:] if len(historical) > 36 else historical

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        recent_history.index, recent_history.values,
        color=_PALETTE[0], linewidth=2, label="Historical CPI YoY (%)",
    )

    ax.plot(
        forecast.index, forecast.values,
        color=_PALETTE[1], linewidth=2, linestyle="--", label=f"Forecast ({model_name})",
    )

    if conf_lower is not None and conf_upper is not None:
        ax.fill_between(
            forecast.index,
            conf_lower.values,
            conf_upper.values,
            color=_PALETTE[1],
            alpha=0.15,
            label="95% Confidence Interval",
        )

    # Vertical separator between history and forecast
    if len(recent_history) > 0:
        ax.axvline(
            x=recent_history.index[-1],
            color="gray",
            linewidth=1.2,
            linestyle=":",
            alpha=0.8,
        )

    ax.axhline(y=2.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="2% target")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("CPI YoY (%)", fontsize=10)
    ax.set_title(f"CPI Inflation Forecast — {model_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = Path(figures_dir) / "forecast_plot.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forecast chart: %s", out_path)
    return str(out_path)


def plot_model_comparison(
    model_metrics: dict[str, dict[str, float]],
    figures_dir: str | Path = "reports/figures",
    dpi: int = 150,
    style: str = "seaborn-v0_8-whitegrid",
) -> str:
    """Bar chart comparing MAE, RMSE, and MAPE across all models.

    Args:
        model_metrics: Dict mapping model name → {mae: float, rmse: float, mape: float}.
        figures_dir: Directory to save the figure.
        dpi: Figure resolution.
        style: Matplotlib style name.

    Returns:
        Path to the saved figure file.
    """
    _setup_style(style)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if not model_metrics:
        logger.warning("No model metrics provided for comparison chart.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No model metrics available", ha="center", va="center")
        out_path = Path(figures_dir) / "model_comparison.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return str(out_path)

    models = list(model_metrics.keys())
    metrics_to_plot = ["mae", "rmse", "mape"]
    metric_labels = {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [model_metrics[m].get(metric, float("nan")) for m in models]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(models))]
        bars = ax.bar(models, values, color=colors, alpha=0.85, width=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(v for v in values if not np.isnan(v)),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(metric_labels[metric], fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model Comparison — Rolling-Origin CV Metrics", fontsize=13)
    fig.tight_layout()

    out_path = Path(figures_dir) / "model_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison chart: %s", out_path)
    return str(out_path)
