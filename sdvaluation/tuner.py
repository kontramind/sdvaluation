"""
Hyperparameter tuning module for LightGBM models.

This module provides utilities for:
- Auto-discovering files in dseed directories
- Hyperparameter optimization using Optuna
- Classification threshold optimization
- Dual-scenario tuning (deployment + optimal)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter tuning. "
        "Install with: pip install optuna"
    )

from .encoding import RDTDatasetEncoder, load_encoding_config

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

console = Console()


class DseedFileDiscovery:
    """
    Auto-discover files in a dseed directory.

    Expected structure:
        dseed<N>/
        ├── *_encoding.yaml
        ├── *_training.csv
        ├── *_test.csv
        ├── *_unsampled.csv
        └── *_metadata.json
    """

    def __init__(self, dseed_dir: Path):
        """
        Initialize file discovery for a dseed directory.

        Args:
            dseed_dir: Path to dseed directory
        """
        self.dseed_dir = Path(dseed_dir)
        if not self.dseed_dir.exists():
            raise FileNotFoundError(f"Directory not found: {dseed_dir}")

        self.files = self._discover_files()

    def _discover_files(self) -> Dict[str, Optional[Path]]:
        """
        Discover files in the dseed directory.

        Returns:
            Dictionary with keys: encoding, training, test, unsampled, metadata
        """
        discovered = {
            "encoding": None,
            "training": None,
            "test": None,
            "unsampled": None,
            "metadata": None,
        }

        # Find all files in directory
        all_files = list(self.dseed_dir.glob("*"))

        for file_path in all_files:
            if file_path.is_dir():
                continue

            filename = file_path.name.lower()

            # Match encoding config
            if file_path.suffix == ".yaml" and "encoding" in filename:
                discovered["encoding"] = file_path

            # Match CSV files
            elif file_path.suffix == ".csv":
                if "training" in filename:
                    discovered["training"] = file_path
                elif "test" in filename:
                    discovered["test"] = file_path
                elif "unsampled" in filename:
                    discovered["unsampled"] = file_path

            # Match metadata
            elif file_path.suffix == ".json" and "metadata" in filename:
                discovered["metadata"] = file_path

        # Validate required files exist
        required = ["encoding", "training", "unsampled"]
        missing = [k for k in required if discovered[k] is None]

        if missing:
            raise FileNotFoundError(
                f"Missing required files in {self.dseed_dir}: {missing}\n"
                f"Found files: {[f.name for f in all_files if f.is_file()]}"
            )

        return discovered

    def get_file(self, key: str) -> Path:
        """Get path to a discovered file."""
        if key not in self.files:
            raise KeyError(f"Unknown file key: {key}")

        path = self.files[key]
        if path is None:
            raise FileNotFoundError(f"File not found: {key}")

        return path

    def __repr__(self) -> str:
        """String representation of discovered files."""
        lines = [f"DseedFileDiscovery({self.dseed_dir})"]
        for key, path in self.files.items():
            status = "✓" if path else "✗"
            name = path.name if path else "Not found"
            lines.append(f"  {status} {key:12s}: {name}")
        return "\n".join(lines)


def load_and_encode_data(
    data_file: Path,
    encoding_config: Path,
    target_column: str = "IS_READMISSION_30D",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and encode data using RDT.

    Args:
        data_file: Path to CSV file
        encoding_config: Path to encoding YAML
        target_column: Name of target column

    Returns:
        Tuple of (X_encoded, y)
    """
    # Load data
    data = pd.read_csv(data_file)

    # Validate target column
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {data_file.name}. "
            f"Available: {list(data.columns)}"
        )

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Load encoding config
    config = load_encoding_config(encoding_config)

    # Filter config to only include feature columns
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

    # Encode features
    encoder = RDTDatasetEncoder(filtered_config)
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    return X_encoded, y


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    n_jobs: int = -1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X: Feature matrix
        y: Target labels
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        n_jobs: Number of parallel jobs
        seed: Random seed

    Returns:
        Dictionary with best_params and best_cv_score
    """
    console.print(f"[cyan]Optimizing hyperparameters ({n_trials} trials, {n_folds}-fold CV)...[/cyan]")

    # Compute class imbalance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    console.print(f"  Class distribution: {n_pos:,} positive, {n_neg:,} negative")
    console.print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    # Define objective function
    def objective(trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=4),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50, step=5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": seed,
            "verbose": -1,
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model
            model = LGBMClassifier(**params)
            model.fit(X_train_fold, y_train_fold)

            # Evaluate with ROC-AUC
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(score)

        return np.mean(cv_scores)

    # Create Optuna study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Extract best results
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = seed
    best_params["verbose"] = -1

    console.print(f"[green]✓ Best CV ROC-AUC: {study.best_value:.4f}[/green]")

    return {
        "best_params": best_params,
        "best_cv_score": study.best_value,
    }


def optimize_threshold(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "f1",
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Optimize classification threshold using cross-validation.

    Args:
        model: Trained LightGBM model
        X: Feature matrix
        y: Target labels
        metric: Metric to optimize (f1, recall, precision, youden)
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with optimal_threshold and metrics
    """
    console.print(f"[cyan]Optimizing threshold (metric: {metric})...[/cyan]")

    # Collect predictions from CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_y_true = []
    all_y_proba = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Clone model and train
        fold_model = LGBMClassifier(**model.get_params())
        fold_model.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]

        all_y_true.extend(y_val_fold)
        all_y_proba.extend(y_pred_proba)

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    # Find optimal threshold
    thresholds = np.linspace(0.1, 0.9, 81)  # Test 81 thresholds
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (all_y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(all_y_true, y_pred, zero_division=0.0)
        elif metric == "recall":
            score = recall_score(all_y_true, y_pred, zero_division=0.0)
        elif metric == "precision":
            score = precision_score(all_y_true, y_pred, zero_division=0.0)
        elif metric == "youden":
            # Youden's J statistic = Sensitivity + Specificity - 1
            tn = np.sum((y_pred == 0) & (all_y_true == 0))
            fp = np.sum((y_pred == 1) & (all_y_true == 0))
            fn = np.sum((y_pred == 0) & (all_y_true == 1))
            tp = np.sum((y_pred == 1) & (all_y_true == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    # Compute all metrics at optimal threshold
    y_pred_optimal = (all_y_proba >= best_threshold).astype(int)

    metrics = {
        "f1": f1_score(all_y_true, y_pred_optimal, zero_division=0.0),
        "precision": precision_score(all_y_true, y_pred_optimal, zero_division=0.0),
        "recall": recall_score(all_y_true, y_pred_optimal, zero_division=0.0),
    }

    # Compute Youden's J
    tn = np.sum((y_pred_optimal == 0) & (all_y_true == 0))
    fp = np.sum((y_pred_optimal == 1) & (all_y_true == 0))
    fn = np.sum((y_pred_optimal == 0) & (all_y_true == 1))
    tp = np.sum((y_pred_optimal == 1) & (all_y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["youden"] = sensitivity + specificity - 1

    console.print(f"[green]✓ Optimal threshold: {best_threshold:.3f}[/green]")
    console.print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
                 f"Recall: {metrics['recall']:.4f}")

    return {
        "optimal_threshold": float(best_threshold),
        "threshold_metrics": {k: float(v) for k, v in metrics.items()},
    }


def tune_dual_scenario(
    dseed_dir: Path,
    target_column: str = "IS_READMISSION_30D",
    n_trials: int = 100,
    n_folds: int = 5,
    threshold_metric: str = "f1",
    n_jobs: int = -1,
    seed: int = 42,
    output_name: str = "hyperparams.json",
) -> Dict[str, Any]:
    """
    Tune hyperparameters for both deployment and optimal scenarios.

    Args:
        dseed_dir: Path to dseed directory
        target_column: Name of target column
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        threshold_metric: Metric to optimize threshold (f1, recall, precision, youden)
        n_jobs: Number of parallel jobs
        seed: Random seed
        output_name: Name of output JSON file

    Returns:
        Dictionary with tuning results
    """
    dseed_dir = Path(dseed_dir)

    # Discover files
    console.print(f"\n[bold]Discovering files in {dseed_dir.name}...[/bold]")
    discovery = DseedFileDiscovery(dseed_dir)
    console.print(discovery)

    # Load and encode data
    console.print(f"\n[bold]Loading and encoding data...[/bold]")

    console.print("[cyan]Loading deployment data (unsampled)...[/cyan]")
    X_deployment, y_deployment = load_and_encode_data(
        discovery.get_file("unsampled"),
        discovery.get_file("encoding"),
        target_column,
    )
    console.print(f"  Samples: {len(X_deployment):,}")

    console.print("[cyan]Loading optimal data (training)...[/cyan]")
    X_optimal, y_optimal = load_and_encode_data(
        discovery.get_file("training"),
        discovery.get_file("encoding"),
        target_column,
    )
    console.print(f"  Samples: {len(X_optimal):,}")

    # Tune deployment scenario
    console.print(f"\n[bold yellow]Scenario 1: Deployment (unsampled data)[/bold yellow]")
    deployment_results = optimize_hyperparameters(
        X_deployment, y_deployment,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
        seed=seed,
    )

    # Optimize threshold for deployment
    deployment_model = LGBMClassifier(**deployment_results["best_params"])
    deployment_threshold = optimize_threshold(
        deployment_model,
        X_deployment, y_deployment,
        metric=threshold_metric,
        n_folds=n_folds,
        seed=seed,
    )

    # Tune optimal scenario
    console.print(f"\n[bold green]Scenario 2: Optimal (training data)[/bold green]")
    optimal_results = optimize_hyperparameters(
        X_optimal, y_optimal,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
        seed=seed,
    )

    # Optimize threshold for optimal
    optimal_model = LGBMClassifier(**optimal_results["best_params"])
    optimal_threshold = optimize_threshold(
        optimal_model,
        X_optimal, y_optimal,
        metric=threshold_metric,
        n_folds=n_folds,
        seed=seed,
    )

    # Compute parameter differences
    param_diff = {}
    for key in deployment_results["best_params"]:
        if key in ["random_state", "verbose"]:
            continue
        deploy_val = deployment_results["best_params"][key]
        optimal_val = optimal_results["best_params"][key]
        if isinstance(deploy_val, (int, float)) and isinstance(optimal_val, (int, float)):
            param_diff[key] = float(optimal_val - deploy_val)

    # Create output dictionary
    output = {
        "metadata": {
            "dseed": dseed_dir.name,
            "created_at": datetime.now().isoformat(),
            "sdvaluation_version": "0.1.0",
            "tuning_config": {
                "n_trials": n_trials,
                "n_folds": n_folds,
                "threshold_metric": threshold_metric,
                "target_column": target_column,
                "seed": seed,
            },
        },
        "deployment": {
            "description": "Hyperparameters tuned on unsampled (population) data",
            "tuning_data": discovery.get_file("unsampled").name,
            "tuning_samples": len(X_deployment),
            "best_cv_score": deployment_results["best_cv_score"],
            "lgbm_params": deployment_results["best_params"],
            **deployment_threshold,
        },
        "optimal": {
            "description": "Hyperparameters tuned on training (real) data",
            "tuning_data": discovery.get_file("training").name,
            "tuning_samples": len(X_optimal),
            "best_cv_score": optimal_results["best_cv_score"],
            "lgbm_params": optimal_results["best_params"],
            **optimal_threshold,
        },
        "comparison": {
            "param_differences": param_diff,
            "cv_score_gap": float(optimal_results["best_cv_score"] - deployment_results["best_cv_score"]),
            "threshold_gap": float(optimal_threshold["optimal_threshold"] - deployment_threshold["optimal_threshold"]),
        },
    }

    # Save to file
    output_path = dseed_dir / output_name
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[bold green]✓ Tuning complete![/bold green]")
    console.print(f"[green]Results saved to: {output_path}[/green]")

    return output
