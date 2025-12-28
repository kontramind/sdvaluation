"""
Command-line interface for Data Shapley valuation.

This module provides a Typer-based CLI for running Data Shapley valuation
on synthetic data, specifically designed for MIMIC-III readmission datasets.
"""

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core import run_data_valuation
from .tuner import tune_dual_scenario, run_leaf_alignment_baseline

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
        "READMIT",
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
                -c READMIT

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
            f"\n[bold green] Data valuation completed successfully![/bold green]"
        )
        console.print(f"[green]Results saved to: {output_dir}[/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during data valuation:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@app.command(name="tune")
def tune_hyperparameters(
    dseed_dir: Path = typer.Option(
        ...,
        "-d",
        "--dseed-dir",
        help="Path to dseed directory containing training and test data",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    target_column: str = typer.Option(
        "READMIT",
        "-c",
        "--target-column",
        help="Name of the target column",
    ),
    n_trials: int = typer.Option(
        100,
        "-n",
        "--n-trials",
        help="Number of Bayesian optimization trials",
        min=10,
    ),
    n_folds: int = typer.Option(
        5,
        "-k",
        "--n-folds",
        help="Number of cross-validation folds",
        min=2,
    ),
    threshold_metric: str = typer.Option(
        "f1",
        "--threshold-metric",
        help="Metric to optimize classification threshold: f1, precision, recall, youden",
    ),
    n_jobs: int = typer.Option(
        -1,
        "-j",
        "--n-jobs",
        help="Number of parallel jobs (1=sequential, -1=all CPUs)",
    ),
    seed: int = typer.Option(
        42,
        "-s",
        "--seed",
        help="Random seed for reproducibility",
    ),
    output_name: str = typer.Option(
        "hyperparams.json",
        "-o",
        "--output-name",
        help="Output filename for hyperparameters",
    ),
    optimize_metric: str = typer.Option(
        "auroc",
        "--optimize-metric",
        help="Metric to optimize during hyperparameter search: auroc, pr_auc, f1, precision, recall",
    ),
) -> None:
    """
    Tune LightGBM hyperparameters on training data.

    This command performs dual-scenario hyperparameter tuning:

    Scenario 1 (Deployment): Tune on unsampled (population) data
    - Represents realistic deployment with distribution shift
    - Parameters optimized for 40k population distribution

    Scenario 2 (Optimal): Tune on training (real) data
    - Represents best-case performance ceiling
    - Parameters optimized for 10k sample distribution

    The command auto-discovers files in the dseed directory:
    - *_unsampled.csv: 40k population data for deployment tuning
    - *_training.csv: 10k real data for optimal tuning
    - *_encoding.yaml: RDT encoding configuration

    Optimization strategy:
    - Hyperparameters: Optimized using ROC-AUC (threshold-independent)
    - Threshold: Optimized for specified metric (F1/recall/precision)

    Output (saved to dseed_dir/hyperparams.json):
    - deployment.lgbm_params: LightGBM hyperparameters tuned on unsampled
    - deployment.optimal_threshold: Classification threshold
    - optimal.lgbm_params: LightGBM hyperparameters tuned on training
    - optimal.optimal_threshold: Classification threshold
    - comparison: Parameter differences and performance gaps

    Examples:

        Basic usage (auto-discover files):

            $ sdvaluation tune --dseed-dir dseed6765/

        Custom settings with more trials:

            $ sdvaluation tune \\
                --dseed-dir dseed6765/ \\
                --n-trials 200 \\
                --threshold-metric recall \\
                --n-jobs 8

        Batch process all dseeds:

            $ for dseed in dseed*/; do \\
                sdvaluation tune --dseed-dir $dseed; \\
              done
    """
    console.print("\n[bold cyan]LightGBM Hyperparameter Tuning[/bold cyan]")
    console.print("[cyan]Dual Scenario: Deployment + Optimal[/cyan]\n")

    start_time = time.time()

    try:
        # Validate threshold metric
        valid_threshold_metrics = ["f1", "recall", "precision", "youden"]
        if threshold_metric not in valid_threshold_metrics:
            console.print(
                f"[bold red]Error:[/bold red] Invalid threshold metric '{threshold_metric}'. "
                f"Must be one of: {', '.join(valid_threshold_metrics)}"
            )
            raise typer.Exit(code=1)

        # Validate optimize metric
        valid_optimize_metrics = ["auroc", "pr_auc", "f1", "precision", "recall"]
        if optimize_metric not in valid_optimize_metrics:
            console.print(
                f"[bold red]Error:[/bold red] Invalid optimize metric '{optimize_metric}'. "
                f"Must be one of: {', '.join(valid_optimize_metrics)}"
            )
            raise typer.Exit(code=1)

        # Run tuning
        results = tune_dual_scenario(
            dseed_dir=dseed_dir,
            target_column=target_column,
            n_trials=n_trials,
            n_folds=n_folds,
            threshold_metric=threshold_metric,
            n_jobs=n_jobs,
            seed=seed,
            output_name=output_name,
            optimize_metric=optimize_metric,
        )

        # Display summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  CV ROC-AUC:        {results['hyperparams']['best_cv_score']:.4f}")
        console.print(f"  Test ROC-AUC:      {results['hyperparams']['test_evaluation']['auroc']:.4f}")
        console.print(f"  CV→Test Gap:       {results['hyperparams']['test_evaluation']['auroc'] - results['hyperparams']['best_cv_score']:+.4f}")
        console.print(f"  Optimal Threshold: {results['hyperparams']['optimal_threshold']:.3f}")

        # Display execution time
        elapsed_time = time.time() - start_time
        console.print(f"\n[bold]Execution time:[/bold] {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")

        console.print(f"\n[bold green]✓ Hyperparameter tuning completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during hyperparameter tuning:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
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
        "READMIT",
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
    leaf_n_estimators: int = typer.Option(
        500,
        "--leaf-n-estimators",
        help="Number of trees for leaf alignment (more = tighter CIs, default: 500)",
        min=100,
    ),
    n_jobs: int = typer.Option(
        1,
        "--n-jobs",
        "-j",
        help="Number of parallel jobs (1=sequential, -1=all CPUs). Used for LGBM training and leaf alignment.",
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
            leaf_n_estimators=leaf_n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Dual evaluation completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during dual evaluation:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@app.command(name="leaf-alignment")
def leaf_alignment_baseline(
    dseed_dir: Path = typer.Option(
        ...,
        "-d",
        "--dseed-dir",
        help="Path to dseed directory with hyperparams.json",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    target_column: str = typer.Option(
        "READMIT",
        "-c",
        "--target-column",
        help="Name of the target column",
    ),
    n_estimators: int = typer.Option(
        500,
        "--n-estimators",
        help="Number of trees for leaf alignment (more = tighter CIs)",
        min=100,
    ),
    n_jobs: int = typer.Option(
        1,
        "-j",
        "--n-jobs",
        help="Number of parallel jobs (1=sequential, -1=all CPUs)",
    ),
    random_state: int = typer.Option(
        42,
        "-s",
        "--seed",
        help="Random seed for reproducibility",
    ),
    cross_test: bool = typer.Option(
        False,
        "--cross-test",
        help="Include cross-test scenarios (unsampled+optimal params, training+deployment params)",
    ),
) -> None:
    """
    Establish leaf alignment baseline using real training data.

    This command evaluates how well real training data represents real test
    data using leaf co-occurrence analysis. It provides a baseline for later
    comparing synthetic data quality.

    The command runs two scenarios by default:

    Scenario 1 (Deployment Baseline): Unsampled → Test
    - Trains on 40k unsampled (population) data with deployment hyperparams
    - Evaluates on 10k real test data
    - Shows: Population data quality baseline

    Scenario 2 (Optimal Baseline): Training → Test
    - Trains on 10k real training data with optimal hyperparams
    - Evaluates on 10k real test data
    - Shows: Sampled training data quality baseline

    With --cross-test flag, adds two additional scenarios:

    Scenario 3 (Cross-test A): Unsampled → Test (with optimal hyperparams)
    - Tests: How much do optimal hyperparams improve unsampled data?

    Scenario 4 (Cross-test B): Training → Test (with deployment hyperparams)
    - Tests: How robust is training data to suboptimal hyperparams?

    This enables decomposition analysis to separate:
    - Pure data quality effect (unsampled vs training data)
    - Pure hyperparameter effect (deployment vs optimal params)
    - Interaction effects

    Requirements:
    - hyperparams.json must exist in dseed directory (run 'tune' first)
    - Test data must be available

    Output files (saved to dseed directory):
    - leaf_alignment_deployment_baseline.csv: Per-point utilities
    - leaf_alignment_optimal_baseline.csv: Per-point utilities
    - leaf_alignment_cross_*.csv: Cross-test utilities (if --cross-test)
    - leaf_alignment_summary.json: Summary statistics

    Example usage:

        # After running tune command
        $ sdvaluation tune --dseed-dir dseed6765/

        # Standard 2-scenario baseline
        $ sdvaluation leaf-alignment --dseed-dir dseed6765/

        # Full 4-scenario analysis with decomposition
        $ sdvaluation leaf-alignment --dseed-dir dseed6765/ --cross-test

        # With more trees for tighter confidence intervals
        $ sdvaluation leaf-alignment \\
            --dseed-dir dseed6765/ \\
            --n-estimators 1000 \\
            --n-jobs -1 \\
            --cross-test
    """
    try:
        console.print("\n[bold]Leaf Alignment Baseline Analysis[/bold]")
        if cross_test:
            console.print("[cyan](Including cross-test scenarios)[/cyan]")
        console.print("=" * 60)

        # Run baseline analysis
        run_leaf_alignment_baseline(
            dseed_dir=dseed_dir,
            target_column=target_column,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            cross_test=cross_test,
        )

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Baseline analysis completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during baseline analysis:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="eval")
def evaluate_synthetic_data(
    dseed_dir: Path = typer.Option(
        ...,
        "-d",
        "--dseed-dir",
        help="Path to dseed directory with hyperparams.json",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    synthetic_file: Path = typer.Option(
        ...,
        "-s",
        "--synthetic-file",
        help="Path to synthetic training CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target_column: str = typer.Option(
        "READMIT",
        "-c",
        "--target-column",
        help="Name of the target column",
    ),
    n_estimators: int = typer.Option(
        500,
        "--n-estimators",
        min=100,
        help="Number of trees for leaf alignment (more = tighter confidence intervals)",
    ),
    n_jobs: int = typer.Option(
        1,
        "-j",
        "--n-jobs",
        help="Number of parallel jobs (1=sequential, -1=all CPUs)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Custom output path for CSV results (default: dseed_dir/synthetic_evaluation.csv)",
    ),
) -> None:
    """
    Evaluate synthetic data quality using leaf alignment.

    This command evaluates synthetic training data by:
    1. Loading hyperparameters tuned on real training data (from 'tune' command)
    2. Training a model on the synthetic data using those hyperparameters
    3. Measuring leaf co-occurrence with real test data
    4. Identifying beneficial, harmful, and hallucinated synthetic points

    The evaluation produces two outputs:
    - CSV file with per-point utilities and classifications
    - JSON summary with aggregate statistics

    Requirements:
    - hyperparams.json must exist in dseed directory (run 'tune' first)
    - Real test data must be available in dseed directory

    Interpretation:
    - Beneficial points: Synthetic samples that improve model performance
    - Harmful points: Synthetic samples that hurt model performance
    - Hallucinated points: Synthetic samples with no alignment to real test data

    Example usage:

        # After running tune command
        $ sdvaluation tune --dseed-dir dseed55/

        # Evaluate synthetic data
        $ sdvaluation eval \\
            --dseed-dir dseed55/ \\
            --synthetic-file synth_10k.csv

        # With more trees for tighter confidence intervals
        $ sdvaluation eval \\
            --dseed-dir dseed55/ \\
            --synthetic-file synth_10k.csv \\
            --n-estimators 1000 \\
            --n-jobs -1
    """
    try:
        from .tuner import evaluate_synthetic

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Synthetic Data Evaluation - sdvaluation[/bold cyan]")
        console.print("=" * 60)

        start_time = time.time()

        evaluate_synthetic(
            dseed_dir=dseed_dir,
            synthetic_file=synthetic_file,
            target_column=target_column,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            seed=seed,
            output_file=output,
        )

        # Display execution time
        elapsed_time = time.time() - start_time
        console.print(f"\n[bold]Execution time:[/bold] {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Synthetic evaluation completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error during synthetic evaluation:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
