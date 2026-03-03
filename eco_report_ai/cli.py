"""Command-line interface for EcoReport AI.

Usage:
  python -m eco_report_ai run
  python -m eco_report_ai run --data-source sample
  python -m eco_report_ai run --config my_config.yaml
  python -m eco_report_ai evaluate
  python -m eco_report_ai version
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from eco_report_ai import __version__
from eco_report_ai.logging_config import setup_logging


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Path to the YAML configuration file.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity level.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str, log_level: str) -> None:
    """EcoReport AI — Production macroeconomic forecasting and report generation."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["log_level"] = log_level
    setup_logging(log_level=log_level)


@cli.command()
@click.option(
    "--data-source",
    "data_source",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "fred", "sample"], case_sensitive=False),
    help="Data source: 'auto' (FRED if key available, else sample), 'fred', 'sample'.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Log execution plan without running the pipeline.",
)
@click.pass_context
def run(ctx: click.Context, data_source: str, dry_run: bool) -> None:
    """Run the full EcoReport AI pipeline: data → features → models → report."""
    from eco_report_ai.pipeline import run_pipeline

    config_path = ctx.obj.get("config_path", "config.yaml")

    click.echo(f"EcoReport AI v{__version__} — starting pipeline ...")
    click.echo(f"  Config     : {config_path}")
    click.echo(f"  Data source: {data_source}")
    if dry_run:
        click.echo("  [DRY RUN] — no files will be written.")

    try:
        evidence = run_pipeline(
            config_path=config_path,
            dry_run=dry_run,
            override_source=data_source if data_source != "auto" else None,
        )
        if not dry_run:
            click.secho("Pipeline completed successfully!", fg="green")
            click.echo(f"  Best model : {evidence.get('best_model', 'N/A')}")
            click.echo(f"  Best RMSE  : {evidence.get('best_rmse', 'N/A'):.4f}")
            click.echo(f"  Report     : reports/latest_report.md")
            click.echo(f"  JSON       : reports/latest_report.json")
    except Exception as exc:
        click.secho(f"Pipeline failed: {exc}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Run only the model evaluation (backtesting) without writing a report."""
    from eco_report_ai.config import load_config
    from eco_report_ai.data.loaders import load_macro_data
    from eco_report_ai.evaluation.backtesting import RollingOriginCV
    from eco_report_ai.features.build_features import build_features
    from eco_report_ai.models.econometrics import ARIMAModel, OLSModel
    from eco_report_ai.models.ml_baselines import GradientBoostingModel

    import numpy as np

    config_path = ctx.obj.get("config_path", "config.yaml")
    config = load_config(config_path)

    click.echo("Loading data and running model evaluation ...")
    try:
        df_raw, source = load_macro_data(config.data)
        df_features = build_features(df_raw, config.features)
        target_col = config.features.target_column
        y = df_features[target_col].dropna()
        X = df_features.drop(columns=[target_col]).loc[y.index]
        X = X.select_dtypes(include=[np.number])

        cv = RollingOriginCV(
            n_folds=config.evaluation.n_folds,
            forecast_horizon=config.evaluation.forecast_horizon,
        )

        for ModelCls, name in [
            (lambda: OLSModel(formula=config.models.ols.formula), "OLS"),
            (lambda: ARIMAModel(max_p=config.models.arima.max_p), "ARIMA"),
            (lambda: GradientBoostingModel(), "GradientBoosting"),
        ]:
            try:
                model = ModelCls()
                result = cv.run_backtest(model, X, y)
                click.echo(
                    f"  {name:20s} MAE={result.aggregate_metrics['mae']['mean']:.4f} "
                    f"RMSE={result.aggregate_metrics['rmse']['mean']:.4f} "
                    f"MAPE={result.aggregate_metrics['mape']['mean']:.2f}%"
                )
            except Exception as exc:
                click.echo(f"  {name}: FAILED — {exc}")

        click.secho("Evaluation complete.", fg="green")
    except Exception as exc:
        click.secho(f"Evaluation failed: {exc}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def report(ctx: click.Context) -> None:
    """Generate report from existing results JSON (skip pipeline re-run)."""
    from eco_report_ai.config import load_config
    from eco_report_ai.reporting.nlg import NarrativeGenerator
    from eco_report_ai.reporting.report_writer import ReportWriter
    from eco_report_ai.utils.io import load_json

    config_path = ctx.obj.get("config_path", "config.yaml")
    config = load_config(config_path)

    json_path = Path(config.reporting.output_dir) / f"{config.reporting.report_filename}.json"

    if not json_path.exists():
        click.secho(f"No results JSON found at {json_path}. Run the pipeline first.", fg="red")
        sys.exit(1)

    evidence = load_json(json_path)
    nlg = NarrativeGenerator()
    writer = ReportWriter(
        output_dir=config.reporting.output_dir,
        report_filename=config.reporting.report_filename,
        nlg=nlg,
    )
    md_path = writer.write_markdown(evidence)
    click.secho(f"Report regenerated: {md_path}", fg="green")


@cli.command()
def version() -> None:
    """Display EcoReport AI version information."""
    click.echo(f"EcoReport AI v{__version__}")
    click.echo(f"Python {sys.version}")

    try:
        import pandas as pd
        import numpy as np
        import statsmodels
        import sklearn

        click.echo(f"pandas {pd.__version__}")
        click.echo(f"numpy {np.__version__}")
        click.echo(f"statsmodels {statsmodels.__version__}")
        click.echo(f"scikit-learn {sklearn.__version__}")
    except ImportError:
        pass

    try:
        import torch
        click.echo(f"torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        click.echo("torch: not installed")
