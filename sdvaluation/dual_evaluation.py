"""
Dual Scenario Evaluation for Synthetic Data Quality Assessment.

Compares synthetic data performance under two scenarios:
1. Optimal: Hyperparameters tuned on real training data (best case)
2. Deployment: Hyperparameters tuned on separate population data (realistic)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .encoding import RDTDatasetEncoder, load_encoding_config
from .tuner import tune_hyperparameters

console = Console()


def load_and_encode_data(
    data_file: Path,
    target_column: str,
    encoding_config: Path,
    encoder: Optional[RDTDatasetEncoder] = None,
    fit_encoder: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[RDTDatasetEncoder]]:
    """
    Load and encode a dataset.

    Args:
        data_file: Path to CSV file
        target_column: Name of target column
        encoding_config: Path to RDT encoding config YAML
        encoder: Existing encoder (if None, will create new one)
        fit_encoder: Whether to fit the encoder on this data

    Returns:
        Tuple of (X_encoded, y, encoder)
    """
    # Load data
    data = pd.read_csv(data_file)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Load encoding config
    config = load_encoding_config(encoding_config)

    # Filter config to exclude target column
    feature_columns = set(X.columns)
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

    # Create or use encoder
    if encoder is None:
        encoder = RDTDatasetEncoder(filtered_config)

    # Fit and/or transform
    if fit_encoder:
        encoder.fit(X)

    X_encoded = encoder.transform(X)

    # Reset indexes
    X_encoded = X_encoded.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_encoded, y, encoder


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any],
    threshold: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train LGBM and evaluate on test set.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        params: LGBM hyperparameters
        threshold: Classification threshold
        random_state: Random seed

    Returns:
        Dictionary with metrics and confusion matrix
    """
    # Prepare parameters (remove metadata fields)
    lgbm_params = params.copy()
    lgbm_params.pop('imbalance_method', None)
    lgbm_params.pop('early_stopping_rounds', None)
    lgbm_params['verbose'] = -1  # Suppress output

    # Compute scale_pos_weight for this specific dataset
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    if n_pos > 0 and n_neg > 0:
        lgbm_params['scale_pos_weight'] = n_neg / n_pos

    # Train model
    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Compute metrics
    # Clip probabilities for log loss
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

    metrics = {
        'auroc': float(roc_auc_score(y_test, y_pred_proba)),
        'logloss': float(log_loss(y_test, y_pred_proba_clipped)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        },
        'threshold': threshold,
    }

    return metrics


def print_confusion_matrix(name: str, metrics: Dict[str, Any]) -> None:
    """Display formatted confusion matrix and metrics."""
    from rich.table import Table

    cm = metrics['confusion_matrix']

    console.print(f"\n[bold]{name}[/bold]")
    console.print(f"  Threshold: {metrics['threshold']:.3f}")

    # Confusion matrix table
    table = Table(show_header=True, title="Confusion Matrix")
    table.add_column("", style="bold")
    table.add_column("Predicted: 0", justify="right")
    table.add_column("Predicted: 1", justify="right")

    table.add_row(
        "Actual: 0",
        f"[green]{cm['tn']:,}[/green] (TN)",
        f"[red]{cm['fp']:,}[/red] (FP)",
    )
    table.add_row(
        "Actual: 1",
        f"[red]{cm['fn']:,}[/red] (FN)",
        f"[green]{cm['tp']:,}[/green] (TP)",
    )

    console.print(table)

    # Metrics
    console.print(f"\n  [bold]Performance Metrics:[/bold]")
    console.print(f"    AUROC:     {metrics['auroc']:.4f}")
    console.print(f"    Precision: {metrics['precision']:.4f}")
    console.print(f"    Recall:    {metrics['recall']:.4f}")
    console.print(f"    F1 Score:  {metrics['f1']:.4f}")
    console.print(f"    FPR:       {metrics['fpr']:.4f} (False Positive Rate)")
    console.print(f"    FNR:       {metrics['fnr']:.4f} (False Negative Rate)")


def compare_confusion_matrices(
    name1: str,
    metrics1: Dict[str, Any],
    name2: str,
    metrics2: Dict[str, Any],
) -> None:
    """Display side-by-side comparison of confusion matrices."""
    from rich.table import Table

    console.print(f"\n[bold yellow]{'═' * 70}[/bold yellow]")
    console.print(f"[bold yellow]Comparison: {name1} vs {name2:^70}[/bold yellow]")
    console.print(f"[bold yellow]{'═' * 70}[/bold yellow]")

    cm1 = metrics1['confusion_matrix']
    cm2 = metrics2['confusion_matrix']

    # Compute differences
    diff_tn = cm2['tn'] - cm1['tn']
    diff_fp = cm2['fp'] - cm1['fp']
    diff_fn = cm2['fn'] - cm1['fn']
    diff_tp = cm2['tp'] - cm1['tp']

    diff_fpr = metrics2['fpr'] - metrics1['fpr']
    diff_fnr = metrics2['fnr'] - metrics1['fnr']
    diff_precision = metrics2['precision'] - metrics1['precision']
    diff_recall = metrics2['recall'] - metrics1['recall']
    diff_f1 = metrics2['f1'] - metrics1['f1']
    diff_auroc = metrics2['auroc'] - metrics1['auroc']

    # Confusion matrix differences table
    table = Table(title="Confusion Matrix Comparison", show_header=True)
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

    table.add_row("TN", f"{cm1['tn']:,}", f"{cm2['tn']:,}", format_diff(diff_tn))
    table.add_row("FP", f"{cm1['fp']:,}", f"{cm2['fp']:,}", format_diff(diff_fp))
    table.add_row("FN", f"{cm1['fn']:,}", f"{cm2['fn']:,}", format_diff(diff_fn))
    table.add_row("TP", f"{cm1['tp']:,}", f"{cm2['tp']:,}", format_diff(diff_tp))

    console.print(table)

    # Metrics comparison
    console.print(f"\n[bold]Metric Changes ({name2} vs {name1}):[/bold]")

    def format_metric_diff(val, lower_is_better=False):
        if abs(val) < 0.001:
            return f"{val:+.4f} (no change)"
        elif lower_is_better:
            # For FPR/FNR, negative change is good
            if val < 0:
                return f"[green]{val:+.4f} (improved)[/green]"
            else:
                return f"[red]{val:+.4f} (degraded)[/red]"
        else:
            # For other metrics, positive change is good
            if val > 0:
                return f"[green]{val:+.4f} (improved)[/green]"
            else:
                return f"[red]{val:+.4f} (degraded)[/red]"

    console.print(f"  AUROC:     {format_metric_diff(diff_auroc)}")
    console.print(f"  Precision: {format_metric_diff(diff_precision)}")
    console.print(f"  Recall:    {format_metric_diff(diff_recall)}")
    console.print(f"  F1 Score:  {format_metric_diff(diff_f1)}")
    console.print(f"  FPR:       {format_metric_diff(diff_fpr, lower_is_better=True)}")
    console.print(f"  FNR:       {format_metric_diff(diff_fnr, lower_is_better=True)}")

    # Interpretation
    console.print(f"\n[bold]Key Observations:[/bold]")
    if abs(diff_fp) > 10:
        direction = "increased" if diff_fp > 0 else "decreased"
        color = "red" if diff_fp > 0 else "green"
        console.print(f"  • False Positives [{color}]{direction}[/{color}] by {abs(diff_fp):,}")
    if abs(diff_fn) > 10:
        direction = "increased" if diff_fn > 0 else "decreased"
        color = "red" if diff_fn > 0 else "green"
        console.print(f"  • False Negatives [{color}]{direction}[/{color}] by {abs(diff_fn):,}")


def display_scenario_comparison(
    scenario_name: str,
    real_metrics: Dict[str, Any],
    synth_metrics: Dict[str, Any],
) -> None:
    """Display comparison between real and synthetic performance."""
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print(f"[bold cyan]{scenario_name:^70}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    # Show individual confusion matrices
    print_confusion_matrix("Real Training Data", real_metrics)
    print_confusion_matrix("Synthetic Training Data", synth_metrics)

    # Show detailed comparison
    compare_confusion_matrices("Real", real_metrics, "Synth", synth_metrics)


def display_transfer_gap(
    real_perf_10k: Dict[str, Any],
    real_perf_40k: Dict[str, Any],
) -> None:
    """Display hyperparameter transfer gap analysis."""
    console.print(f"\n[bold yellow]{'═' * 70}[/bold yellow]")
    console.print(f"[bold yellow]{'Hyperparameter Transfer Gap Analysis':^70}[/bold yellow]")
    console.print(f"[bold yellow]{'═' * 70}[/bold yellow]\n")

    auroc_10k = real_perf_10k['auroc']
    auroc_40k = real_perf_40k['auroc']
    transfer_gap = auroc_10k - auroc_40k

    # Determine gap severity
    if abs(transfer_gap) < 0.02:
        gap_status = "[green]Excellent[/green]"
        gap_icon = "✓"
    elif abs(transfer_gap) < 0.05:
        gap_status = "[yellow]Acceptable[/yellow]"
        gap_icon = "⚠"
    else:
        gap_status = "[red]Poor Transfer[/red]"
        gap_icon = "❌"

    console.print(f"[bold]Real data performance:[/bold]")
    console.print(f"  With 10k-tuned params (optimal): {auroc_10k:.4f}")
    console.print(f"  With 40k-tuned params (deployed): {auroc_40k:.4f}")
    console.print(f"\n[bold]Transfer Gap:[/bold] {transfer_gap:+.4f} {gap_icon}")
    console.print(f"[bold]Status:[/bold] {gap_status}")

    if abs(transfer_gap) < 0.02:
        console.print("\n[green]✓ Hyperparameters transfer excellently.[/green]")
        console.print("  Both scenarios provide valid comparison.")
    elif abs(transfer_gap) < 0.05:
        console.print("\n[yellow]⚠ Acceptable transfer gap.[/yellow]")
        console.print("  Minor hyperparameter mismatch, but comparison is still valid.")
    else:
        console.print("\n[red]❌ Poor hyperparameter transfer![/red]")
        console.print("  40k-tuned params may not be optimal for 10k training.")
        console.print("  Synthetic comparison may be confounded by this mismatch.")


def run_dual_evaluation(
    tuning_data: Path,
    real_train: Path,
    synth_train: Path,
    real_test: Path,
    target_column: str = "IS_READMISSION_30D",
    encoding_config: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_trials: int = 100,
    n_folds: int = 5,
    threshold_metric: str = 'f1',
    run_leaf_alignment: bool = True,
    leaf_n_estimators: int = 500,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run dual scenario evaluation.

    Scenario 1 (Optimal): Hyperparameters tuned on real_train (10k)
    Scenario 2 (Deployment): Hyperparameters tuned on tuning_data (40k)

    Args:
        tuning_data: Path to 40k population data for hyperparameter tuning
        real_train: Path to 10k real training data
        synth_train: Path to 10k synthetic training data
        real_test: Path to 10k real test data
        target_column: Name of target column
        encoding_config: Path to RDT encoding config YAML
        output_dir: Output directory for results
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        threshold_metric: Metric to optimize threshold ('f1', 'precision', 'recall', 'youden')
        run_leaf_alignment: Whether to run leaf alignment analysis
        leaf_n_estimators: Number of trees for leaf alignment (more = tighter CIs)
        n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs)
        random_state: Random seed

    Returns:
        Dictionary with all results
    """
    if output_dir is None:
        output_dir = Path("experiments/dual_eval")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print()
    console.print(Panel(
        "[bold cyan]Dual Scenario Evaluation - sdvaluation[/bold cyan]",
        expand=False
    ))

    # ========================================================================
    # Step 1: Load and Encode Data
    # ========================================================================
    console.print("\n[bold]Step 1: Loading and Encoding Data[/bold]")

    # Load tuning data (40k) and fit encoder
    console.print("  [1/4] Loading tuning data (40k)...")
    X_tuning, y_tuning, encoder = load_and_encode_data(
        tuning_data, target_column, encoding_config,
        encoder=None, fit_encoder=True
    )
    console.print(f"    ✓ {len(X_tuning):,} samples, {X_tuning.shape[1]} features")

    # Load real train (10k) using same encoder
    console.print("  [2/4] Loading real train (10k)...")
    X_real_train, y_real_train, _ = load_and_encode_data(
        real_train, target_column, encoding_config,
        encoder=encoder, fit_encoder=False
    )
    console.print(f"    ✓ {len(X_real_train):,} samples")

    # Load synth train (10k) using same encoder
    console.print("  [3/4] Loading synth train (10k)...")
    X_synth_train, y_synth_train, _ = load_and_encode_data(
        synth_train, target_column, encoding_config,
        encoder=encoder, fit_encoder=False
    )
    console.print(f"    ✓ {len(X_synth_train):,} samples")

    # Load real test (10k) using same encoder
    console.print("  [4/4] Loading real test (10k)...")
    X_real_test, y_real_test, _ = load_and_encode_data(
        real_test, target_column, encoding_config,
        encoder=encoder, fit_encoder=False
    )
    console.print(f"    ✓ {len(X_real_test):,} samples")

    # ========================================================================
    # Step 2: Hyperparameter Tuning
    # ========================================================================
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print(f"[bold cyan]{'Phase 1: Hyperparameter Tuning':^70}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    # Tune on 40k
    console.print("[bold][1/2] Tuning on 40k Population Data[/bold]")
    console.print(f"  Running {n_trials} Optuna trials with {n_folds}-fold CV...")
    console.print(f"  Optimizing threshold for: {threshold_metric}")
    result_40k = tune_hyperparameters(
        X_tuning, y_tuning,
        n_trials=n_trials,
        n_folds=n_folds,
        threshold_metric=threshold_metric,
        optimize_threshold=True,
        n_jobs=n_jobs,
        random_state=random_state
    )
    params_40k = result_40k['best_params']
    threshold_40k = result_40k['threshold']
    console.print(f"  [green]✓ Best CV AUROC: {result_40k['cv_score']:.4f}[/green]")
    console.print(f"  [green]✓ Optimal threshold: {threshold_40k:.3f}[/green]")

    # Save params_40k
    params_40k_file = output_dir / f"params_40k_{timestamp}.json"
    with open(params_40k_file, 'w') as f:
        json.dump({
            'params': params_40k,
            'threshold': threshold_40k,
            'threshold_metric': threshold_metric,
            'cv_score': result_40k['cv_score'],
            'n_trials': n_trials,
            'n_folds': n_folds,
            'source': 'tuning_data_40k',
        }, f, indent=2)
    console.print(f"  [green]✓ Saved: {params_40k_file.name}[/green]")

    # Tune on 10k
    console.print("\n[bold][2/2] Tuning on 10k Real Train[/bold]")
    console.print(f"  Running {n_trials} Optuna trials with {n_folds}-fold CV...")
    console.print(f"  Optimizing threshold for: {threshold_metric}")
    result_10k = tune_hyperparameters(
        X_real_train, y_real_train,
        n_trials=n_trials,
        n_folds=n_folds,
        threshold_metric=threshold_metric,
        optimize_threshold=True,
        n_jobs=n_jobs,
        random_state=random_state
    )
    params_10k = result_10k['best_params']
    threshold_10k = result_10k['threshold']
    console.print(f"  [green]✓ Best CV AUROC: {result_10k['cv_score']:.4f}[/green]")
    console.print(f"  [green]✓ Optimal threshold: {threshold_10k:.3f}[/green]")

    # Save params_10k
    params_10k_file = output_dir / f"params_10k_{timestamp}.json"
    with open(params_10k_file, 'w') as f:
        json.dump({
            'params': params_10k,
            'threshold': threshold_10k,
            'threshold_metric': threshold_metric,
            'cv_score': result_10k['cv_score'],
            'n_trials': n_trials,
            'n_folds': n_folds,
            'source': 'real_train_10k',
        }, f, indent=2)
    console.print(f"  [green]✓ Saved: {params_10k_file.name}[/green]")

    # Display threshold comparison
    threshold_gap = abs(threshold_10k - threshold_40k)
    console.print(f"\n[bold]Threshold Comparison:[/bold]")
    console.print(f"  40k-optimized: {threshold_40k:.3f}")
    console.print(f"  10k-optimized: {threshold_10k:.3f}")
    console.print(f"  Gap: {threshold_gap:.3f}")

    # ========================================================================
    # Step 3: Scenario 1 - Optimal (10k-tuned params)
    # ========================================================================
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print(f"[bold cyan]{'Phase 2: Scenario 1 - Optimal (10k-tuned params)':^70}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    console.print("[bold]Training on Real (10k) → Test on Real (10k)[/bold]")
    console.print(f"  Using threshold: {threshold_10k:.3f}")
    real_perf_10k = train_and_evaluate(
        X_real_train, y_real_train,
        X_real_test, y_real_test,
        params_10k,
        threshold=threshold_10k,
        random_state=random_state
    )

    console.print("[bold]Training on Synth (10k) → Test on Real (10k)[/bold]")
    console.print(f"  Using threshold: {threshold_10k:.3f}")
    synth_perf_10k = train_and_evaluate(
        X_synth_train, y_synth_train,
        X_real_test, y_real_test,
        params_10k,
        threshold=threshold_10k,
        random_state=random_state
    )

    display_scenario_comparison(
        "Scenario 1: Optimal (10k-tuned params)",
        real_perf_10k,
        synth_perf_10k
    )

    # Save scenario 1 results
    scenario1_file = output_dir / f"scenario_1_optimal_{timestamp}.json"
    with open(scenario1_file, 'w') as f:
        json.dump({
            'scenario': 'optimal_10k_tuned',
            'params_source': 'real_train_10k',
            'real_performance': real_perf_10k,
            'synth_performance': synth_perf_10k,
        }, f, indent=2)

    # ========================================================================
    # Step 4: Scenario 2 - Deployment (40k-tuned params)
    # ========================================================================
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print(f"[bold cyan]{'Phase 3: Scenario 2 - Deployment (40k-tuned params)':^70}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    console.print("[bold]Training on Real (10k) → Test on Real (10k)[/bold]")
    console.print(f"  Using threshold: {threshold_40k:.3f}")
    real_perf_40k = train_and_evaluate(
        X_real_train, y_real_train,
        X_real_test, y_real_test,
        params_40k,
        threshold=threshold_40k,
        random_state=random_state
    )

    console.print("[bold]Training on Synth (10k) → Test on Real (10k)[/bold]")
    console.print(f"  Using threshold: {threshold_40k:.3f}")
    synth_perf_40k = train_and_evaluate(
        X_synth_train, y_synth_train,
        X_real_test, y_real_test,
        params_40k,
        threshold=threshold_40k,
        random_state=random_state
    )

    display_scenario_comparison(
        "Scenario 2: Deployment (40k-tuned params)",
        real_perf_40k,
        synth_perf_40k
    )

    # Save scenario 2 results
    scenario2_file = output_dir / f"scenario_2_deployment_{timestamp}.json"
    with open(scenario2_file, 'w') as f:
        json.dump({
            'scenario': 'deployment_40k_tuned',
            'params_source': 'tuning_data_40k',
            'real_performance': real_perf_40k,
            'synth_performance': synth_perf_40k,
        }, f, indent=2)

    # ========================================================================
    # Step 5: Transfer Gap Analysis
    # ========================================================================
    display_transfer_gap(real_perf_10k, real_perf_40k)

    # ========================================================================
    # Step 6: Leaf Alignment (if enabled)
    # ========================================================================
    leaf_results = {}
    if run_leaf_alignment:
        console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
        console.print(f"[bold cyan]{'Phase 4: Leaf Alignment - Harmful Detection':^70}[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

        # Import leaf alignment functions
        from .leaf_alignment import run_leaf_alignment

        # Run with 10k params
        console.print("[bold][1/2] With 10k-tuned params[/bold]")
        leaf_10k_file = output_dir / f"leaf_alignment_10k_params_{timestamp}.csv"
        leaf_10k_results = run_leaf_alignment(
            X_synth_train, y_synth_train,
            X_real_test, y_real_test,
            params_10k,
            output_file=leaf_10k_file,
            n_estimators=leaf_n_estimators,
            n_jobs=n_jobs,
            random_state=random_state
        )
        console.print(f"  [green]✓ Saved: {leaf_10k_file.name}[/green]")
        leaf_results['10k_params'] = leaf_10k_results

        # Run with 40k params
        console.print("\n[bold][2/2] With 40k-tuned params[/bold]")
        leaf_40k_file = output_dir / f"leaf_alignment_40k_params_{timestamp}.csv"
        leaf_40k_results = run_leaf_alignment(
            X_synth_train, y_synth_train,
            X_real_test, y_real_test,
            params_40k,
            output_file=leaf_40k_file,
            n_estimators=leaf_n_estimators,
            n_jobs=n_jobs,
            random_state=random_state
        )
        console.print(f"  [green]✓ Saved: {leaf_40k_file.name}[/green]")
        leaf_results['40k_params'] = leaf_40k_results

    # ========================================================================
    # Step 7: Summary
    # ========================================================================
    console.print(f"\n[bold green]{'═' * 70}[/bold green]")
    console.print(f"[bold green]{'Summary':^70}[/bold green]")
    console.print(f"[bold green]{'═' * 70}[/bold green]\n")

    console.print("[bold green]✓ Dual evaluation complete![/bold green]\n")

    # Summary statistics
    transfer_gap = real_perf_10k['auroc'] - real_perf_40k['auroc']
    synth_gap_10k = real_perf_10k['auroc'] - synth_perf_10k['auroc']
    synth_gap_40k = real_perf_40k['auroc'] - synth_perf_40k['auroc']

    console.print("[bold]Key Findings:[/bold]")
    console.print(f"  • Hyperparameter transfer gap: {transfer_gap:+.4f} AUROC")
    console.print(f"  • Synthetic quality gap (10k params): {synth_gap_10k:+.4f} AUROC")
    console.print(f"  • Synthetic quality gap (40k params): {synth_gap_40k:+.4f} AUROC")

    if run_leaf_alignment:
        console.print(f"  • Leaf alignment (10k params): "
                     f"{leaf_10k_results['n_hallucinated']:,} "
                     f"({leaf_10k_results['pct_hallucinated']:.1f}%) hallucinated")
        console.print(f"  • Leaf alignment (40k params): "
                     f"{leaf_40k_results['n_hallucinated']:,} "
                     f"({leaf_40k_results['pct_hallucinated']:.1f}%) hallucinated")

    console.print(f"\n[bold]Results saved to:[/bold] {output_dir}/")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'threshold_metric': threshold_metric,
        'transfer_gap': transfer_gap,
        'threshold_gap': threshold_gap,
        'scenario_1_optimal': {
            'params_source': 'real_train_10k',
            'threshold': threshold_10k,
            'real_auroc': real_perf_10k['auroc'],
            'synth_auroc': synth_perf_10k['auroc'],
            'gap': synth_gap_10k,
        },
        'scenario_2_deployment': {
            'params_source': 'tuning_data_40k',
            'threshold': threshold_40k,
            'real_auroc': real_perf_40k['auroc'],
            'synth_auroc': synth_perf_40k['auroc'],
            'gap': synth_gap_40k,
        },
        'leaf_alignment': leaf_results if run_leaf_alignment else None,
        'files': {
            'params_40k': str(params_40k_file),
            'params_10k': str(params_10k_file),
            'scenario_1': str(scenario1_file),
            'scenario_2': str(scenario2_file),
        }
    }

    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
