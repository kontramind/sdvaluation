"""
Command-line interface for Data Shapley valuation.

This module provides a Typer-based CLI for running Data Shapley valuation
on synthetic data, specifically designed for MIMIC-III readmission datasets.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core import run_data_valuation

app = typer.Typer(help="Data Shapley valuation for synthetic data")
console = Console()


@app.command(name="shapley")
def data_valuation_mimic_iii(
    train_data: Path = typer.Option(
        ...,
        "-t",
        "--train-data",
        help="Path to synthetic training CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_data: Path = typer.Option(
        ...,
        "-e",
        "--test-data",
        help="Path to real test CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target_column: str = typer.Option(
        "IS_READMISSION_30D",
        "-c",
        "--target-column",
        help="Name of the target column",
    ),
    encoding_config: Optional[Path] = typer.Option(
        None,
        "--encoding-config",
        help="Path to RDT encoding configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    num_samples: int = typer.Option(
        100,
        "-n",
        "--num-samples",
        help="Number of Monte Carlo samples for Shapley computation",
        min=10,
    ),
    max_coalition_size: Optional[int] = typer.Option(
        None,
        "-m",
        "--max-coalition-size",
        help="Maximum coalition size for early truncation (optional)",
        min=10,
    ),
    random_state: int = typer.Option(
        42,
        "-s",
        "--random-state",
        help="Random seed for reproducibility",
    ),
    output_dir: Path = typer.Option(
        "experiments/data_valuation",
        "-o",
        "--output-dir",
        help="Output directory for results",
    ),
    include_features: bool = typer.Option(
        True,
        "--include-features/--no-include-features",
        help="Include feature columns in output CSV",
    ),
    lgbm_params_json: Optional[Path] = typer.Option(
        None,
        "--lgbm-params-json",
        help="Path to JSON file with LGBM hyperparameters and config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """
    Run Data Shapley valuation on MIMIC-III synthetic data.

    This command computes Data Shapley values for synthetic training data
    by evaluating its contribution to model performance on real test data.
    The valuation uses LightGBM as the utility model and supports optional
    hyperparameter configuration.

    Examples:

        Basic usage with default settings:

            $ sdvaluation shapley \\
                -t data/synthetic_train.csv \\
                -e data/real_test.csv \\
                -c IS_READMISSION_30D

        Fast valuation with early truncation:

            $ sdvaluation shapley \\
                -t data/synthetic_train.csv \\
                -e data/real_test.csv \\
                -n 50 \\
                -m 100 \\
                -o experiments/quick_valuation

        Using pre-trained hyperparameters and encoding config:

            $ sdvaluation shapley \\
                -t data/synthetic_train.csv \\
                -e data/real_test.csv \\
                --lgbm-params-json experiments/training/best_params.json \\
                --encoding-config configs/encoding.yaml
    """
    console.print("\n[bold cyan]MIMIC-III Data Shapley Valuation[/bold cyan]\n")

    try:
        # Load LGBM parameters if provided
        lgbm_params = None
        if lgbm_params_json is not None:
            console.print(f"[yellow]Loading LGBM parameters from {lgbm_params_json}[/yellow]")
            with open(lgbm_params_json, "r") as f:
                params_data = json.load(f)
                lgbm_params = params_data.get("best_hyperparameters")

                # Auto-load encoding config if not explicitly provided
                if encoding_config is None:
                    preprocessing_config = params_data.get("preprocessing", {})
                    encoding_config_path = preprocessing_config.get("encoding_config_path")
                    if encoding_config_path:
                        encoding_config = Path(encoding_config_path)
                        console.print(
                            f"[yellow]Auto-loading encoding config from {encoding_config}[/yellow]"
                        )

        # Run the data valuation
        console.print("[green]Starting Data Shapley computation...[/green]\n")

        run_data_valuation(
            train_file=train_data,
            test_file=test_data,
            target_column=target_column,
            encoding_config=encoding_config,
            num_samples=num_samples,
            max_coalition_size=max_coalition_size,
            random_state=random_state,
            output_dir=output_dir,
            include_features=include_features,
            lgbm_params=lgbm_params,
        )

        console.print(
            f"\n[bold green] Data valuation completed successfully![/bold green]"
        )
        console.print(f"[green]Results saved to: {output_dir}[/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during data valuation:[/bold red] {e}\n")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
