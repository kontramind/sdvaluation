#!/usr/bin/env python3
"""
Compare hyperparameters across multiple dseeds to understand different behaviors.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_hyperparams(dseed_path: Path) -> dict:
    """Load hyperparams.json from a dseed directory."""
    hyperparams_file = dseed_path / "hyperparams.json"
    if not hyperparams_file.exists():
        raise FileNotFoundError(f"No hyperparams.json found in {dseed_path}")

    with open(hyperparams_file, "r") as f:
        return json.load(f)


def compare_hyperparams(dseed_dirs: list[Path]):
    """Compare hyperparameters across multiple dseeds."""

    # Load all hyperparameters
    all_params = {}
    for dseed_dir in dseed_dirs:
        dseed_name = dseed_dir.name
        try:
            all_params[dseed_name] = load_hyperparams(dseed_dir)
        except FileNotFoundError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
            continue

    if not all_params:
        console.print("[red]No valid hyperparameters found![/red]")
        return

    # Display Deployment Parameters
    console.print("\n[bold cyan]═══ Deployment Parameters (Unsampled Data) ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    for dseed_name in sorted(all_params.keys()):
        table.add_column(dseed_name, justify="right")

    # Key LGBM parameters to compare
    key_params = [
        "num_leaves",
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_samples",
        "reg_lambda",
        "reg_alpha",
        "colsample_bytree",
        "subsample",
        "min_split_gain",
    ]

    for param in key_params:
        row = [param]
        for dseed_name in sorted(all_params.keys()):
            params = all_params[dseed_name]
            value = params["deployment"]["lgbm_params"].get(param, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table.add_row(*row)

    # Add threshold
    row = ["optimal_threshold"]
    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]
        threshold = params["deployment"].get("optimal_threshold", "N/A")
        if isinstance(threshold, float):
            row.append(f"{threshold:.4f}")
        else:
            row.append(str(threshold))
    table.add_row(*row, style="bold")

    # Add CV score
    row = ["best_cv_score"]
    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]
        cv_score = params["deployment"].get("best_cv_score", "N/A")
        if isinstance(cv_score, float):
            row.append(f"{cv_score:.4f}")
        else:
            row.append(str(cv_score))
    table.add_row(*row, style="bold green")

    console.print(table)

    # Display Optimal Parameters
    console.print("\n[bold cyan]═══ Optimal Parameters (Training Data) ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    for dseed_name in sorted(all_params.keys()):
        table.add_column(dseed_name, justify="right")

    for param in key_params:
        row = [param]
        for dseed_name in sorted(all_params.keys()):
            params = all_params[dseed_name]
            value = params["optimal"]["lgbm_params"].get(param, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table.add_row(*row)

    # Add threshold
    row = ["optimal_threshold"]
    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]
        threshold = params["optimal"].get("optimal_threshold", "N/A")
        if isinstance(threshold, float):
            row.append(f"{threshold:.4f}")
        else:
            row.append(str(threshold))
    table.add_row(*row, style="bold")

    # Add CV score
    row = ["best_cv_score"]
    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]
        cv_score = params["optimal"].get("best_cv_score", "N/A")
        if isinstance(cv_score, float):
            row.append(f"{cv_score:.4f}")
        else:
            row.append(str(cv_score))
    table.add_row(*row, style="bold green")

    console.print(table)

    # Display Comparison Summary
    console.print("\n[bold cyan]═══ Comparison Summary ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan")
    table.add_column("Training Samples", justify="right")
    table.add_column("Deployment CV", justify="right")
    table.add_column("Optimal CV", justify="right")
    table.add_column("CV Gap", justify="right")
    table.add_column("Threshold Gap", justify="right")

    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]
        training_samples = params["optimal"].get("tuning_samples", "N/A")
        deploy_cv = params["deployment"].get("best_cv_score", 0)
        optimal_cv = params["optimal"].get("best_cv_score", 0)
        cv_gap = params["comparison"].get("cv_score_gap", 0)
        threshold_gap = params["comparison"].get("threshold_gap", 0)

        # Color code CV gap
        if cv_gap < -0.01:
            cv_gap_style = "red"
        elif cv_gap < 0:
            cv_gap_style = "yellow"
        else:
            cv_gap_style = "green"

        table.add_row(
            dseed_name,
            str(training_samples),
            f"{deploy_cv:.4f}",
            f"{optimal_cv:.4f}",
            f"[{cv_gap_style}]{cv_gap:+.4f}[/{cv_gap_style}]",
            f"{threshold_gap:+.4f}",
        )

    console.print(table)

    # Display test performance if available
    console.print("\n[bold cyan]═══ Test Performance (if available) ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan")
    table.add_column("Scenario", style="cyan")
    table.add_column("Test AUROC", justify="right")
    table.add_column("Test F1", justify="right")
    table.add_column("CV→Test Gap", justify="right")

    for dseed_name in sorted(all_params.keys()):
        params = all_params[dseed_name]

        # Deployment test performance
        deploy_test = params["deployment"].get("test_evaluation")
        if deploy_test:
            cv_score = params["deployment"]["best_cv_score"]
            test_auroc = deploy_test.get("test_auroc", 0)
            gap = test_auroc - cv_score
            gap_style = "red" if gap < -0.02 else "yellow" if gap < 0 else "green"

            table.add_row(
                dseed_name,
                "Deployment",
                f"{test_auroc:.4f}",
                f"{deploy_test.get('test_f1', 0):.4f}",
                f"[{gap_style}]{gap:+.4f}[/{gap_style}]",
            )

        # Optimal test performance
        optimal_test = params["optimal"].get("test_evaluation")
        if optimal_test:
            cv_score = params["optimal"]["best_cv_score"]
            test_auroc = optimal_test.get("test_auroc", 0)
            gap = test_auroc - cv_score
            gap_style = "red" if gap < -0.02 else "yellow" if gap < 0 else "green"

            table.add_row(
                "",
                "Optimal",
                f"{test_auroc:.4f}",
                f"{optimal_test.get('test_f1', 0):.4f}",
                f"[{gap_style}]{gap:+.4f}[/{gap_style}]",
            )

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("\n[bold]Usage:[/bold]")
        console.print("  python compare_hyperparams.py <dseed_dir1> <dseed_dir2> ...")
        console.print("\n[bold]Example:[/bold]")
        console.print("  python compare_hyperparams.py ../rd-lake/dseed55 ../rd-lake/dseed2025 ../rd-lake/dseed6765")
        sys.exit(1)

    dseed_dirs = [Path(arg) for arg in sys.argv[1:]]

    # Validate directories
    invalid_dirs = [d for d in dseed_dirs if not d.exists() or not d.is_dir()]
    if invalid_dirs:
        console.print(f"[red]Error: Invalid directories: {invalid_dirs}[/red]")
        sys.exit(1)

    compare_hyperparams(dseed_dirs)
