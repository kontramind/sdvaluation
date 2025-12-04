"""
Generate confusion matrices for Real vs Next-Gen Synthetic data comparison.

Trains LightGBM on full training datasets and evaluates on test set,
producing confusion matrices for direct FP/FN comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console
from rich.table import Table

from sdvaluation.encoding import RDTDatasetEncoder, load_encoding_config

console = Console()


def load_lgbm_params(params_file: Path) -> dict:
    """Load LightGBM parameters from JSON file."""
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Remove parameters that shouldn't be passed to LGBMClassifier
    params.pop("imbalance_method", None)
    params.pop("early_stopping_rounds", None)
    params.pop("optimal_threshold", None)
    params.pop("test_metrics", None)
    params.pop("best_hyperparameters", None)

    # Filter out any remaining dict-valued parameters (metadata fields)
    params = {k: v for k, v in params.items() if not isinstance(v, dict)}

    return params


def train_and_evaluate(
    train_file: Path,
    test_file: Path,
    encoding_config: Path,
    lgbm_params: dict,
    target_column: str = "IS_READMISSION_30D",
    threshold: float = 0.5,
    random_state: int = 42,
) -> dict:
    """
    Train LightGBM on full training data and evaluate on test set.

    Returns:
        Dictionary with predictions, confusion matrix, and metrics.
    """
    console.print(f"[bold]Loading data from:[/bold]")
    console.print(f"  Train: {train_file.name}")
    console.print(f"  Test:  {test_file.name}")

    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Separate features and target
    X_train_original = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test_original = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    console.print(f"  Train samples: {len(train_data):,}")
    console.print(f"  Test samples:  {len(test_data):,}")

    # Encode features
    config = load_encoding_config(encoding_config)
    feature_columns = set(X_train_original.columns)
    filtered_config = {
        "sdtypes": {
            col: dtype
            for col, dtype in config["sdtypes"].items()
            if col in feature_columns
        },
        "transformers": {
            col: transformer
            for col, transformer in config["transformers"].items()
            if col in feature_columns
        },
    }

    encoder = RDTDatasetEncoder(filtered_config)
    encoder.fit(X_train_original)
    X_train = encoder.transform(X_train_original)
    X_test = encoder.transform(X_test_original)

    # Handle class imbalance
    params = lgbm_params.copy()
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    params["scale_pos_weight"] = n_neg / n_pos

    console.print(f"\n[bold]Training LightGBM...[/bold]")
    console.print(f"  Class distribution: {n_pos:,} positive, {n_neg:,} negative")
    console.print(f"  Scale pos weight: {params['scale_pos_weight']:.2f}")

    # Train model
    model = LGBMClassifier(**params, random_state=random_state, verbose=-1)
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute metrics
    tn, fp, fn, tp = cm.ravel()

    # Compute rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def print_confusion_matrix(name: str, results: dict):
    """Pretty print confusion matrix and metrics."""
    cm = results["confusion_matrix"]

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{name:^60}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # Confusion matrix table
    table = Table(title="Confusion Matrix", show_header=True)
    table.add_column("", style="bold")
    table.add_column("Predicted: 0 (Negative)", justify="right")
    table.add_column("Predicted: 1 (Positive)", justify="right")

    table.add_row(
        "Actual: 0 (Negative)",
        f"[green]{results['tn']:,}[/green] (TN)",
        f"[red]{results['fp']:,}[/red] (FP)",
    )
    table.add_row(
        "Actual: 1 (Positive)",
        f"[red]{results['fn']:,}[/red] (FN)",
        f"[green]{results['tp']:,}[/green] (TP)",
    )

    console.print(table)

    # Metrics
    console.print(f"\n[bold]Metrics:[/bold]")
    console.print(f"  Precision: {results['precision']:.4f}")
    console.print(f"  Recall:    {results['recall']:.4f}")
    console.print(f"  F1 Score:  {results['f1']:.4f}")
    console.print(f"  FPR:       {results['fpr']:.4f} (False Positive Rate)")
    console.print(f"  FNR:       {results['fnr']:.4f} (False Negative Rate)")


def compare_confusion_matrices(name1: str, results1: dict, name2: str, results2: dict):
    """Compare two confusion matrices."""
    console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
    console.print(f"[bold yellow]{'Comparison: ' + name1 + ' vs ' + name2:^60}[/bold yellow]")
    console.print(f"[bold yellow]{'='*60}[/bold yellow]")

    # Compute differences
    diff_tn = results2["tn"] - results1["tn"]
    diff_fp = results2["fp"] - results1["fp"]
    diff_fn = results2["fn"] - results1["fn"]
    diff_tp = results2["tp"] - results1["tp"]

    diff_fpr = results2["fpr"] - results1["fpr"]
    diff_fnr = results2["fnr"] - results1["fnr"]
    diff_precision = results2["precision"] - results1["precision"]
    diff_recall = results2["recall"] - results1["recall"]
    diff_f1 = results2["f1"] - results1["f1"]

    # Difference table
    table = Table(title="Confusion Matrix Differences", show_header=True)
    table.add_column("", style="bold")
    table.add_column(name1, justify="right")
    table.add_column(name2, justify="right")
    table.add_column("Difference", justify="right")

    def format_diff(val):
        if val > 0:
            return f"[red]+{val:,}[/red]"
        elif val < 0:
            return f"[green]{val:,}[/green]"
        else:
            return f"{val:,}"

    table.add_row("TN", f"{results1['tn']:,}", f"{results2['tn']:,}", format_diff(diff_tn))
    table.add_row("FP", f"{results1['fp']:,}", f"{results2['fp']:,}", format_diff(diff_fp))
    table.add_row("FN", f"{results1['fn']:,}", f"{results2['fn']:,}", format_diff(diff_fn))
    table.add_row("TP", f"{results1['tp']:,}", f"{results2['tp']:,}", format_diff(diff_tp))

    console.print(table)

    # Metrics comparison
    console.print(f"\n[bold]Metric Changes ({name2} vs {name1}):[/bold]")

    def format_metric_diff(val):
        if abs(val) < 0.001:
            return f"{val:+.4f} (no change)"
        elif val > 0:
            return f"[green]{val:+.4f} (improved)[/green]"
        else:
            return f"[red]{val:+.4f} (degraded)[/red]"

    console.print(f"  Precision: {format_metric_diff(diff_precision)}")
    console.print(f"  Recall:    {format_metric_diff(diff_recall)}")
    console.print(f"  F1 Score:  {format_metric_diff(diff_f1)}")
    console.print(f"  FPR:       {format_metric_diff(-diff_fpr)} (lower is better)")
    console.print(f"  FNR:       {format_metric_diff(-diff_fnr)} (lower is better)")

    # Interpretation
    console.print(f"\n[bold]Interpretation:[/bold]")
    if abs(diff_fp) > 10:
        direction = "increased" if diff_fp > 0 else "decreased"
        console.print(f"  • False Positives {direction} by {abs(diff_fp):,}")
    if abs(diff_fn) > 10:
        direction = "increased" if diff_fn > 0 else "decreased"
        console.print(f"  • False Negatives {direction} by {abs(diff_fn):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and compare confusion matrices for Real vs Next-Gen Synthetic data"
    )
    parser.add_argument(
        "--real-train",
        type=Path,
        required=True,
        help="Path to real training data CSV",
    )
    parser.add_argument(
        "--next-gen-train",
        type=Path,
        required=False,
        help="Path to next-gen synthetic training data CSV (optional for comparison)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to test data CSV (same for both)",
    )
    parser.add_argument(
        "--encoding-config",
        type=Path,
        required=True,
        help="Path to RDT encoding config YAML",
    )
    parser.add_argument(
        "--lgbm-params",
        type=Path,
        required=True,
        help="Path to LightGBM parameters JSON",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="IS_READMISSION_30D",
        help="Name of target column",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load LightGBM parameters
    lgbm_params = load_lgbm_params(args.lgbm_params)

    # Train and evaluate on Real data
    console.print("\n[bold magenta]Processing Real Data...[/bold magenta]")
    real_results = train_and_evaluate(
        args.real_train,
        args.test_file,
        args.encoding_config,
        lgbm_params,
        args.target_column,
        args.threshold,
        args.random_state,
    )

    # Print Real results
    print_confusion_matrix("Real Data (Gen0)", real_results)

    # Train and evaluate on next-gen data if provided
    if args.next_gen_train:
        console.print("\n[bold magenta]Processing Next-Gen Synthetic Data...[/bold magenta]")
        next_gen_results = train_and_evaluate(
            args.next_gen_train,
            args.test_file,
            args.encoding_config,
            lgbm_params,
            args.target_column,
            args.threshold,
            args.random_state,
        )

        # Print next-gen results
        print_confusion_matrix("Next-Gen Synthetic Data", next_gen_results)

        # Compare
        compare_confusion_matrices("Real", real_results, "Next-Gen", next_gen_results)
    else:
        console.print("\n[dim]No next-gen data provided - skipping comparison[/dim]")

    console.print("\n[bold green]✓ Analysis Complete![/bold green]\n")


if __name__ == "__main__":
    main()
