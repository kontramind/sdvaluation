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
    n_jobs: int = typer.Option(
        1,
        "-j",
        "--n-jobs",
        help="Number of parallel jobs for computation (1=sequential, -1=all CPUs)",
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

        Parallel execution with 4 CPU cores:

            $ sdvaluation shapley \\
                -t data/synthetic_train.csv \\
                -e data/real_test.csv \\
                -n 20 \\
                -j 4

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
            n_jobs=n_jobs,
        )

        console.print(
            f"\n[bold green] Data valuation completed successfully![/bold green]"
        )
        console.print(f"[green]Results saved to: {output_dir}[/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during data valuation:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@app.command(name="dual-eval")
def dual_evaluation(
    tuning_data: Path = typer.Option(
        ...,
        "--tuning-data",
        help="Path to 40k population data CSV for hyperparameter tuning",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    real_train: Path = typer.Option(
        ...,
        "--real-train",
        help="Path to 10k real training CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    synth_train: Path = typer.Option(
        ...,
        "--synth-train",
        help="Path to 10k synthetic training CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    real_test: Path = typer.Option(
        ...,
        "--real-test",
        help="Path to 10k real test CSV file",
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
    encoding_config: Path = typer.Option(
        ...,
        "--encoding-config",
        help="Path to RDT encoding configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    n_trials: int = typer.Option(
        100,
        "--n-trials",
        "-n",
        help="Number of Bayesian optimization trials",
        min=10,
    ),
    n_folds: int = typer.Option(
        5,
        "--n-folds",
        "-k",
        help="Number of cross-validation folds",
        min=2,
    ),
    output_dir: Path = typer.Option(
        Path("experiments/dual_eval"),
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    threshold_metric: str = typer.Option(
        "f1",
        "--threshold-metric",
        help="Metric to optimize classification threshold: f1, precision, recall, youden",
    ),
    no_leaf_alignment: bool = typer.Option(
        False,
        "--no-leaf-alignment",
        help="Skip leaf alignment analysis (harmful detection)",
    ),
    random_state: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
) -> None:
    """
    Run dual scenario evaluation for synthetic data quality assessment.

    This command compares synthetic data performance under two scenarios:

    Scenario 1 (Optimal): Hyperparameters tuned on 10k real training data
    - Best case performance for both real and synthetic
    - Shows ceiling of synthetic data quality

    Scenario 2 (Deployment): Hyperparameters tuned on 40k population data
    - Realistic deployment scenario with distribution shift
    - Tests robustness of synthetic data to parameter transfer

    The evaluation includes:
    - Hyperparameter tuning on both 40k and 10k datasets
    - Threshold optimization via CV (default: F1 score)
    - Performance comparison (Real vs Synth) in both scenarios
    - Transfer gap analysis (params and thresholds)
    - Optional: Leaf alignment analysis for harmful point detection

    Example usage:

        # Basic usage with default F1 threshold optimization
        sdvaluation dual-eval \\
            --tuning-data population_40k.csv \\
            --real-train real_train_10k.csv \\
            --synth-train synth_train_10k.csv \\
            --real-test real_test_10k.csv \\
            --encoding-config encoding.yaml \\
            --n-trials 100 \\
            --output-dir experiments/dual_eval

        # Optimize for recall (medical use case - minimize false negatives)
        sdvaluation dual-eval \\
            --tuning-data population_40k.csv \\
            --real-train real_train_10k.csv \\
            --synth-train synth_train_10k.csv \\
            --real-test real_test_10k.csv \\
            --encoding-config encoding.yaml \\
            --threshold-metric recall

    Output files:
        - params_40k_*.json: Hyperparameters + threshold from 40k tuning
        - params_10k_*.json: Hyperparameters + threshold from 10k tuning
        - scenario_1_optimal_*.json: Results with 10k-tuned params/threshold
        - scenario_2_deployment_*.json: Results with 40k-tuned params/threshold
        - leaf_alignment_10k_params_*.csv: Harmful detection (10k params)
        - leaf_alignment_40k_params_*.csv: Harmful detection (40k params)
        - summary_*.json: Overall summary with threshold gaps
    """
    from .dual_evaluation import run_dual_evaluation

    try:
        console.print("\n[bold]Dual Scenario Evaluation[/bold]")
        console.print("=" * 60)

        # Run dual evaluation
        results = run_dual_evaluation(
            tuning_data=tuning_data,
            real_train=real_train,
            synth_train=synth_train,
            real_test=real_test,
            target_column=target_column,
            encoding_config=encoding_config,
            output_dir=output_dir,
            n_trials=n_trials,
            n_folds=n_folds,
            threshold_metric=threshold_metric,
            run_leaf_alignment=not no_leaf_alignment,
            random_state=random_state,
        )

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Dual evaluation completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during dual evaluation:[/bold red] {e}\n")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
