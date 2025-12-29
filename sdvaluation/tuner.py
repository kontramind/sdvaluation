"""
Hyperparameter tuning module for LightGBM models.

This module provides utilities for:
- Auto-discovering files in dseed directories
- Hyperparameter optimization using Optuna with comprehensive search space
- Classification threshold optimization
- Dual-scenario tuning (deployment + optimal)

Combines the best features from both previous implementations:
- Comprehensive search space (boosting type, early stopping, feature sampling)
- Native lgb.train() API with early stopping
- Rich console output and progress tracking
- File auto-discovery
- Dual-scenario workflow
"""

import json
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.console import Console
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from rich.table import Table

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter tuning. "
        "Install with: pip install optuna"
    )

from .encoding import RDTDatasetEncoder, load_encoding_config
from .leaf_alignment import run_leaf_alignment

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
        required = ["encoding", "training"]
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


class LGBMTuner:
    """
    Enhanced LightGBM hyperparameter tuner with comprehensive search space.

    Features:
    - Boosting type selection (GBDT vs GOSS)
    - Early stopping rounds optimization
    - Feature and sample fraction tuning
    - Flexible class imbalance handling
    - Native lgb.train() API with callbacks
    - Wide regularization and learning rate ranges
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int = 5,
        n_trials: int = 100,
        n_jobs: int = -1,
        random_state: int = 42,
        optimize_metric: str = "auroc",
    ):
        """
        Initialize the tuner.

        Args:
            X_train: Training features
            y_train: Training labels (binary)
            n_folds: Number of cross-validation folds
            n_trials: Number of Bayesian optimization trials
            n_jobs: Number of parallel jobs for LGBM (1=sequential, -1=all CPUs)
            random_state: Random seed for reproducibility
            optimize_metric: Metric to optimize ('auroc', 'pr_auc', 'f1', 'precision', 'recall')
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.optimize_metric = optimize_metric.lower()
        self.best_params = None
        self.best_score = None

        # Validate optimize_metric
        valid_metrics = ['auroc', 'pr_auc', 'f1', 'precision', 'recall']
        if self.optimize_metric not in valid_metrics:
            raise ValueError(f"optimize_metric must be one of {valid_metrics}, got '{optimize_metric}'")
        self.study = None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization with comprehensive search space.

        Args:
            trial: A trial object from Optuna

        Returns:
            Mean AUROC across folds
        """
        # Scale search space based on dataset size to prevent overfitting
        n_samples = len(self.X_train)

        if n_samples < 15000:
            # MIMIC-III optimized ranges for small datasets (< 15k samples)
            # Based on medical data best practices: stable, less prone to overfitting
            # Tightened to reduce CV→Test gap (targeting ±2% instead of -7%+)
            max_leaves_upper = 31  # Force simpler trees (was 50)
            max_depth_upper = 5  # Shallower trees (was 7)
            min_reg_lambda = 1.0  # Stronger regularization (was 0.5)
            learning_rate_range = (0.01, 0.03)  # Slower, more stable learning (was 0.01-0.05)
            min_data_in_leaf_range = (100, 200)  # Larger leaves for stability (was 50-200)
        else:
            # Wider ranges for large datasets (>= 15k samples)
            max_leaves_upper = 100
            max_depth_upper = 12
            min_reg_lambda = 0.1
            learning_rate_range = (0.001, 0.1)  # Wider for large datasets
            min_data_in_leaf_range = (20, 100)  # Can afford smaller leaves

        # Core hyperparameters
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
            "num_leaves": trial.suggest_int("num_leaves", 10, max_leaves_upper),  # Min 10, adaptive max
            "max_depth": trial.suggest_int("max_depth", 3, max_depth_upper),  # Min 3, adaptive max
            "learning_rate": trial.suggest_float("learning_rate", learning_rate_range[0], learning_rate_range[1]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),  # Min 5
            "n_estimators": 1000,  # Large number, will use early stopping
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

        # Regularization with mandatory minimum for small datasets
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)  # L1
        params["reg_lambda"] = trial.suggest_float("reg_lambda", min_reg_lambda, 10.0)  # L2 with minimum

        # Feature and sample sampling
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        # Leaf constraints
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", min_data_in_leaf_range[0], min_data_in_leaf_range[1])

        # Class imbalance handling
        imbalance_method = trial.suggest_categorical(
            "imbalance_method", ["none", "scale_pos_weight", "is_unbalance"]
        )

        if imbalance_method == "scale_pos_weight":
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            params["scale_pos_weight"] = scale_pos_weight
        elif imbalance_method == "is_unbalance":
            params["is_unbalance"] = True

        # Early stopping
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 7, 30)

        # Perform k-fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        cv_scores = []

        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model with early stopping (suppress all output)
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[val_data],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                            lgb.log_evaluation(0),
                        ],
                    )

            # Predict on validation set
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

            # Calculate metric based on optimize_metric
            if self.optimize_metric == 'auroc':
                score = roc_auc_score(y_val, y_pred_proba)
            elif self.optimize_metric == 'pr_auc':
                from sklearn.metrics import average_precision_score
                score = average_precision_score(y_val, y_pred_proba)
            elif self.optimize_metric in ['f1', 'precision', 'recall']:
                from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
                # Find optimal threshold on this fold
                precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

                if self.optimize_metric == 'f1':
                    # Compute F1 for each threshold
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                    best_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                elif self.optimize_metric == 'precision':
                    # Use threshold that maximizes precision while maintaining reasonable recall
                    valid_indices = recalls >= 0.3  # Maintain at least 30% recall
                    if np.any(valid_indices):
                        best_idx = np.argmax(precisions[valid_indices])
                        best_threshold = thresholds[valid_indices][best_idx] if best_idx < len(thresholds[valid_indices]) else 0.5
                    else:
                        best_threshold = 0.5
                else:  # recall
                    # Use threshold that maximizes recall while maintaining reasonable precision
                    valid_indices = precisions >= 0.2  # Maintain at least 20% precision
                    if np.any(valid_indices):
                        best_idx = np.argmax(recalls[valid_indices])
                        best_threshold = thresholds[valid_indices][best_idx] if best_idx < len(thresholds[valid_indices]) else 0.5
                    else:
                        best_threshold = 0.5

                # Make predictions with optimal threshold
                y_pred = (y_pred_proba >= best_threshold).astype(int)

                # Calculate the metric
                if self.optimize_metric == 'f1':
                    score = f1_score(y_val, y_pred, zero_division=0)
                elif self.optimize_metric == 'precision':
                    score = precision_score(y_val, y_pred, zero_division=0)
                else:  # recall
                    score = recall_score(y_val, y_pred, zero_division=0)

            cv_scores.append(score)

        return np.mean(cv_scores)

    def tune(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run Bayesian Optimization to find best hyperparameters.

        Args:
            show_progress: Whether to show Optuna progress bar

        Returns:
            Dictionary with best_params and best_cv_score
        """
        # Compute class imbalance for reporting
        n_pos = np.sum(self.y_train == 1)
        n_neg = np.sum(self.y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        console.print(f"  Class distribution: {n_pos:,} positive, {n_neg:,} negative")
        console.print(f"  Scale pos weight: {scale_pos_weight:.2f}")

        # Create Optuna study with TPE sampler
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=1,  # Optuna parallelization (1 for sequential)
            show_progress_bar=show_progress,
        )

        # Store best results
        self.best_params = self.study.best_params.copy()
        self.best_score = self.study.best_value

        # Format parameters for LGBMClassifier
        lgbm_params = self._format_params_for_lgbm()

        return {
            "best_params": lgbm_params,
            "best_cv_score": self.best_score,
        }

    def _format_params_for_lgbm(self) -> Dict[str, Any]:
        """
        Format best parameters for LGBMClassifier.

        Returns:
            Dictionary of hyperparameters ready for LGBMClassifier
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before getting parameters")

        # Base parameters
        lgbm_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": self.best_params["boosting_type"],
            "num_leaves": self.best_params["num_leaves"],
            "max_depth": self.best_params["max_depth"],
            "learning_rate": self.best_params["learning_rate"],
            "min_child_samples": self.best_params["min_child_samples"],
            "n_estimators": self.best_params.get("n_estimators", 1000),
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": -1,
            "reg_alpha": self.best_params["reg_alpha"],
            "reg_lambda": self.best_params["reg_lambda"],
            "feature_fraction": self.best_params["feature_fraction"],
            "subsample": self.best_params["subsample"],
            "colsample_bytree": self.best_params["colsample_bytree"],
            "min_data_in_leaf": self.best_params["min_data_in_leaf"],
        }

        # Add class imbalance handling if selected
        imbalance_method = self.best_params.get("imbalance_method", "none")
        if imbalance_method == "scale_pos_weight":
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            lgbm_params["scale_pos_weight"] = scale_pos_weight
        elif imbalance_method == "is_unbalance":
            lgbm_params["is_unbalance"] = True

        return lgbm_params


def load_and_encode_data(
    data_file: Path,
    encoding_config: Path,
    target_column: str = "READMIT",
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
    optimize_metric: str = "auroc",
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using enhanced tuner.

    Args:
        X: Feature matrix
        y: Target labels
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        n_jobs: Number of parallel jobs
        seed: Random seed
        optimize_metric: Metric to optimize ('auroc', 'pr_auc', 'f1', 'precision', 'recall')

    Returns:
        Dictionary with best_params and best_cv_score
    """
    console.print(
        f"[cyan]Optimizing hyperparameters ({n_trials} trials, {n_folds}-fold CV)...[/cyan]"
    )

    # Create and run tuner
    tuner = LGBMTuner(
        X_train=X,
        y_train=y,
        n_folds=n_folds,
        n_trials=n_trials,
        n_jobs=n_jobs,
        random_state=seed,
        optimize_metric=optimize_metric,
    )

    results = tuner.tune(show_progress=True)

    # Display metric name based on optimize_metric
    metric_names = {
        'auroc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'f1': 'F1',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    metric_display = metric_names.get(optimize_metric, optimize_metric.upper())

    console.print(f"[green]✓ Best CV {metric_display}: {results['best_cv_score']:.4f}[/green]")

    return results


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
        model: Trained LightGBM model (or model with get_params())
        X: Feature matrix
        y: Target labels
        metric: Metric to optimize (f1, recall, precision, youden)
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with optimal_threshold and threshold_metrics
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
    # Expanded range for imbalanced data (class imbalance often needs thresholds < 0.1)
    thresholds = np.linspace(0.01, 0.9, 90)  # Test 90 thresholds from 0.01 to 0.9
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
    console.print(
        f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}"
    )

    return {
        "optimal_threshold": float(best_threshold),
        "threshold_metrics": {k: float(v) for k, v in metrics.items()},
    }


def evaluate_on_test(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train model with given params and evaluate on test data.

    Args:
        params: LightGBM hyperparameters
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        seed: Random seed

    Returns:
        Dictionary with test metrics and confusion matrix
    """
    # Train model on full training data
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Compute metrics
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

    metrics = {
        "auroc": float(roc_auc_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "logloss": float(log_loss(y_test, y_pred_proba_clipped)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
    }

    return metrics


def display_test_evaluation(
    scenario_name: str,
    cv_score: float,
    test_metrics: Dict[str, Any],
) -> None:
    """
    Display test evaluation results with confusion matrix.

    Args:
        scenario_name: Name of scenario (e.g., "Deployment", "Optimal")
        cv_score: Cross-validation ROC-AUC score
        test_metrics: Test evaluation metrics
    """
    cm = test_metrics["confusion_matrix"]

    console.print(f"\n[bold cyan]{scenario_name} - Test Evaluation[/bold cyan]")
    console.print(f"  Threshold: {test_metrics['threshold']:.3f}")

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

    # Metrics with CV comparison
    console.print(f"\n  [bold]Performance Metrics:[/bold]")
    auroc_gap = test_metrics["auroc"] - cv_score
    gap_color = "green" if auroc_gap >= 0 else "red"
    console.print(
        f"    AUROC:     {test_metrics['auroc']:.4f} "
        f"(CV: {cv_score:.4f}, gap: [{gap_color}]{auroc_gap:+.4f}[/{gap_color}])"
    )
    console.print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    console.print(f"    Log Loss:  {test_metrics['logloss']:.4f}")
    console.print(f"    F1 Score:  {test_metrics['f1']:.4f}")
    console.print(f"    Precision: {test_metrics['precision']:.4f}")
    console.print(f"    Recall:    {test_metrics['recall']:.4f}")
    console.print(f"    FPR:       {test_metrics['fpr']:.4f} (False Positive Rate)")
    console.print(f"    FNR:       {test_metrics['fnr']:.4f} (False Negative Rate)")


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    timeout: int = None,
    threshold_metric: Literal["f1", "precision", "recall", "youden"] = "f1",
    optimize_threshold: bool = True,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune LGBM hyperparameters and optionally find optimal threshold.

    This is a wrapper function for backwards compatibility with dual_evaluation.py.

    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        timeout: Timeout in seconds (unused, for API compatibility)
        threshold_metric: Metric to optimize threshold for
        optimize_threshold: Whether to find optimal threshold via CV
        n_jobs: Number of parallel jobs for LGBM
        random_state: Random seed

    Returns:
        Dictionary containing best parameters, CV score, and optional threshold
    """
    # Run hyperparameter optimization
    tuner = LGBMTuner(
        X_train=X_train,
        y_train=y_train,
        n_folds=n_folds,
        n_trials=n_trials,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    results = tuner.tune(show_progress=False)
    lgbm_params = results["best_params"]

    result = {
        "best_params": lgbm_params,
        "cv_score": results["best_cv_score"],
        "n_trials": n_trials,
        "n_folds": n_folds,
    }

    # Optimize threshold if requested
    if optimize_threshold:
        # Get CV predictions for threshold optimization
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        all_y_true = []
        all_y_proba = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train model
            model = LGBMClassifier(**lgbm_params)
            model.fit(X_tr, y_tr)

            # Predict
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            all_y_true.extend(y_val)
            all_y_proba.extend(y_pred_proba)

        all_y_true = np.array(all_y_true)
        all_y_proba = np.array(all_y_proba)

        # Find optimal threshold
        # Expanded range for imbalanced data (class imbalance often needs thresholds < 0.1)
        thresholds = np.arange(0.01, 0.9, 0.01)  # Step by 0.01 from 0.01 to 0.9
        best_score = -np.inf
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (all_y_proba >= threshold).astype(int)

            if threshold_metric == "f1":
                score = f1_score(all_y_true, y_pred, zero_division=0)
            elif threshold_metric == "precision":
                score = precision_score(all_y_true, y_pred, zero_division=0)
            elif threshold_metric == "recall":
                score = recall_score(all_y_true, y_pred, zero_division=0)
            elif threshold_metric == "youden":
                tn = np.sum((y_pred == 0) & (all_y_true == 0))
                fp = np.sum((y_pred == 1) & (all_y_true == 0))
                fn = np.sum((y_pred == 0) & (all_y_true == 1))
                tp = np.sum((y_pred == 1) & (all_y_true == 1))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {threshold_metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        result["threshold"] = float(best_threshold)
        result["threshold_metric"] = threshold_metric
    else:
        result["threshold"] = 0.5
        result["threshold_metric"] = "default"

    return result


def tune_dual_scenario(
    dseed_dir: Path,
    target_column: str = "READMIT",
    n_trials: int = 100,
    n_folds: int = 5,
    threshold_metric: str = "f1",
    n_jobs: int = -1,
    seed: int = 42,
    output_name: str = "hyperparams.json",
    optimize_metric: str = "auroc",
) -> Dict[str, Any]:
    """
    Tune hyperparameters on training data and evaluate on test data.

    Simplified tuning workflow:
    1. Tune hyperparameters on real training data
    2. Optimize classification threshold
    3. Evaluate on test data with confusion matrix

    Args:
        dseed_dir: Path to dseed directory
        target_column: Name of target column
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        threshold_metric: Metric to optimize threshold (f1, recall, precision, youden)
        n_jobs: Number of parallel jobs
        seed: Random seed
        output_name: Name of output JSON file
        optimize_metric: Metric to optimize during hyperparameter search ('auroc', 'pr_auc', 'f1', 'precision', 'recall')

    Returns:
        Dictionary with tuning results
    """
    dseed_dir = Path(dseed_dir)

    # Discover files
    console.print(f"\n[bold]Discovering files in {dseed_dir.name}...[/bold]")
    discovery = DseedFileDiscovery(dseed_dir)
    console.print(discovery)

    # Validate test data exists
    if discovery.files["test"] is None:
        raise FileNotFoundError(
            f"Test data not found in {dseed_dir}. "
            "Test data is required for hyperparameter tuning evaluation."
        )

    # Load and encode data
    console.print(f"\n[bold]Loading and encoding data...[/bold]")

    console.print("[cyan]Loading training data...[/cyan]")
    X_train, y_train = load_and_encode_data(
        discovery.get_file("training"),
        discovery.get_file("encoding"),
        target_column,
    )
    console.print(f"  Samples: {len(X_train):,}")
    console.print(f"  Class balance: {np.mean(y_train == 1):.1%} positive")

    console.print("[cyan]Loading test data...[/cyan]")
    X_test, y_test = load_and_encode_data(
        discovery.get_file("test"),
        discovery.get_file("encoding"),
        target_column,
    )
    console.print(f"  Samples: {len(X_test):,}")
    console.print(f"  Class balance: {np.mean(y_test == 1):.1%} positive")

    # Tune hyperparameters
    console.print(f"\n[bold green]Tuning hyperparameters on training data[/bold green]")
    tuning_results = optimize_hyperparameters(
        X_train,
        y_train,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
        seed=seed,
        optimize_metric=optimize_metric,
    )

    # Optimize threshold
    model = LGBMClassifier(**tuning_results["best_params"])
    threshold_results = optimize_threshold(
        model,
        X_train,
        y_train,
        metric=threshold_metric,
        n_folds=n_folds,
        seed=seed,
    )

    # Evaluate on test data
    console.print(f"\n[bold magenta]Evaluating on test data[/bold magenta]")
    test_metrics = evaluate_on_test(
        params=tuning_results["best_params"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold_results["optimal_threshold"],
        seed=seed,
    )

    # Display results
    display_test_evaluation(
        "Training → Test",
        tuning_results["best_cv_score"],
        test_metrics,
    )

    # Create output dictionary (simplified format)
    output = {
        "metadata": {
            "dseed": dseed_dir.name,
            "created_at": datetime.now().isoformat(),
            "sdvaluation_version": "0.2.0",
            "tuning_config": {
                "n_trials": n_trials,
                "n_folds": n_folds,
                "threshold_metric": threshold_metric,
                "target_column": target_column,
                "seed": seed,
            },
        },
        "hyperparams": {
            "description": "Hyperparameters tuned on real training data",
            "tuning_data": discovery.get_file("training").name,
            "tuning_samples": len(X_train),
            "best_cv_score": tuning_results["best_cv_score"],
            "lgbm_params": tuning_results["best_params"],
            **threshold_results,
            "test_evaluation": test_metrics,
        },
    }

    # Save to file
    output_path = dseed_dir / output_name
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[bold green]✓ Tuning complete![/bold green]")
    console.print(f"[green]Results saved to: {output_path}[/green]")

    return output


def evaluate_synthetic(
    dseed_dir: Path,
    synthetic_file: Path,
    target_column: str = "READMIT",
    n_estimators: int = 500,
    n_jobs: int = 1,
    seed: int = 42,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate synthetic data quality using leaf alignment.

    Workflow:
    1. Load hyperparameters from hyperparams.json (tuned on real training)
    2. Train model on synthetic data using those hyperparameters
    3. Evaluate against real test data via leaf co-occurrence
    4. Identify beneficial/harmful/hallucinated synthetic points

    Args:
        dseed_dir: Path to dseed directory with hyperparams.json
        synthetic_file: Path to synthetic training CSV file
        target_column: Name of target column
        n_estimators: Number of trees for leaf alignment (more = tighter CIs)
        n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs)
        seed: Random seed
        output_file: Optional custom output path for CSV (default: dseed_dir/synthetic_evaluation.csv)

    Returns:
        Dictionary with evaluation results
    """
    from .leaf_alignment import run_leaf_alignment

    dseed_dir = Path(dseed_dir)
    synthetic_file = Path(synthetic_file)

    # Discover files
    console.print(f"\n[bold]Discovering files in {dseed_dir.name}...[/bold]")
    discovery = DseedFileDiscovery(dseed_dir)
    console.print(discovery)

    # Validate test data exists
    if discovery.files["test"] is None:
        raise FileNotFoundError(
            f"Test data not found in {dseed_dir}. "
            "Test data is required for synthetic evaluation."
        )

    # Validate synthetic file exists
    if not synthetic_file.exists():
        raise FileNotFoundError(f"Synthetic file not found: {synthetic_file}")

    # Load hyperparameters
    hyperparams_path = dseed_dir / "hyperparams.json"
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"hyperparams.json not found in {dseed_dir}. "
            "Please run 'sdvaluation tune' first to generate hyperparameters."
        )

    console.print(f"\n[bold]Loading hyperparameters...[/bold]")
    with open(hyperparams_path, "r") as f:
        hyperparams_data = json.load(f)

    # Extract LGBM params (handle both old and new format)
    if "hyperparams" in hyperparams_data:
        # New simplified format
        lgbm_params = hyperparams_data["hyperparams"]["lgbm_params"]
        optimal_threshold = hyperparams_data["hyperparams"]["optimal_threshold"]
        best_cv_score = hyperparams_data["hyperparams"]["best_cv_score"]
    elif "optimal" in hyperparams_data:
        # Old dual scenario format
        lgbm_params = hyperparams_data["optimal"]["lgbm_params"]
        optimal_threshold = hyperparams_data["optimal"]["optimal_threshold"]
        best_cv_score = hyperparams_data["optimal"]["best_cv_score"]
    else:
        raise ValueError(
            f"Unrecognized hyperparams.json format. "
            f"Expected 'hyperparams' or 'optimal' key."
        )

    console.print(f"  ✓ Loaded hyperparameters")
    console.print(f"  CV ROC-AUC: {best_cv_score:.4f}")
    console.print(f"  Optimal threshold: {optimal_threshold:.3f}")

    # Load and encode data (match dual-eval approach: fit on real training, transform others)
    console.print(f"\n[bold]Loading and encoding data...[/bold]")

    # Step 1: Load real training data and fit encoder
    console.print("[cyan]Loading real training data (to fit encoder)...[/cyan]")
    from .encoding import RDTDatasetEncoder, load_encoding_config

    train_data = pd.read_csv(discovery.get_file("training"))
    X_train_raw = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    # Create and fit encoder on real training data
    config = load_encoding_config(discovery.get_file("encoding"))
    feature_columns = set(X_train_raw.columns)
    filtered_config = {
        "sdtypes": {col: dtype for col, dtype in config["sdtypes"].items() if col in feature_columns},
        "transformers": {col: transformer for col, transformer in config["transformers"].items() if col in feature_columns},
    }
    encoder = RDTDatasetEncoder(filtered_config)
    encoder.fit(X_train_raw)
    X_train = encoder.transform(X_train_raw).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    console.print(f"  Samples: {len(X_train):,}")
    console.print(f"  Class balance: {np.mean(y_train == 1):.1%} positive")

    # Step 2: Load synthetic data and transform with same encoder
    console.print("[cyan]Loading synthetic training data (using fitted encoder)...[/cyan]")
    synth_data = pd.read_csv(synthetic_file)
    X_synthetic_raw = synth_data.drop(columns=[target_column])
    y_synthetic = synth_data[target_column]
    X_synthetic = encoder.transform(X_synthetic_raw).reset_index(drop=True)
    y_synthetic = y_synthetic.reset_index(drop=True)
    console.print(f"  Samples: {len(X_synthetic):,}")
    console.print(f"  Class balance: {np.mean(y_synthetic == 1):.1%} positive")

    # Step 3: Load real test data and transform with same encoder
    console.print("[cyan]Loading real test data (using fitted encoder)...[/cyan]")
    test_data = pd.read_csv(discovery.get_file("test"))
    X_test_raw = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    X_test = encoder.transform(X_test_raw).reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    console.print(f"  Samples: {len(X_test):,}")
    console.print(f"  Class balance: {np.mean(y_test == 1):.1%} positive")

    # Display class imbalance comparison
    real_pos_pct = np.mean(y_train == 1)
    synth_pos_pct = np.mean(y_synthetic == 1)
    imbalance_diff = synth_pos_pct - real_pos_pct
    imbalance_ratio = synth_pos_pct / real_pos_pct if real_pos_pct > 0 else 0.0

    console.print(f"\n[bold]Class Imbalance Comparison:[/bold]")
    console.print(f"  Real training:      {real_pos_pct:.1%} positive")
    console.print(f"  Synthetic training: {synth_pos_pct:.1%} positive")

    # Color code based on severity
    if abs(imbalance_diff) < 0.02:  # Within 2pp
        color = "green"
        status = "✓ Good match"
    elif abs(imbalance_diff) < 0.05:  # Within 5pp
        color = "yellow"
        status = "⚠ Moderate difference"
    else:  # >5pp difference
        color = "red"
        status = "✗ Large difference"

    console.print(f"  Difference:         [{color}]{imbalance_diff:+.1%} ({status})[/{color}]")
    console.print(f"  Ratio:              {imbalance_ratio:.2f}x")

    # Evaluate model performance (like dual-eval)
    console.print(f"\n[bold magenta]{'═' * 70}[/bold magenta]")
    console.print(f"[bold magenta]{'Model Performance Evaluation':^70}[/bold magenta]")
    console.print(f"[bold magenta]{'═' * 70}[/bold magenta]\n")

    # Baseline: Real training → Real test
    console.print("[bold]Baseline: Real Training → Real Test[/bold]")
    real_metrics = evaluate_on_test(
        params=lgbm_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold,
        seed=seed,
    )
    display_test_evaluation("Real → Test", best_cv_score, real_metrics)

    # Synthetic: Synthetic training → Real test
    console.print("\n[bold]Synthetic: Synthetic Training → Real Test[/bold]")
    synth_metrics = evaluate_on_test(
        params=lgbm_params,
        X_train=X_synthetic,
        y_train=y_synthetic,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold,
        seed=seed,
    )
    display_test_evaluation("Synthetic → Test", best_cv_score, synth_metrics)

    # Display comparison
    console.print(f"\n[bold cyan]Performance Gap (Real - Synthetic)[/bold cyan]")
    console.print(f"  AUROC Gap:     {real_metrics['auroc'] - synth_metrics['auroc']:+.4f}")
    console.print(f"  F1 Gap:        {real_metrics['f1'] - synth_metrics['f1']:+.4f}")
    console.print(f"  Precision Gap: {real_metrics['precision'] - synth_metrics['precision']:+.4f}")
    console.print(f"  Recall Gap:    {real_metrics['recall'] - synth_metrics['recall']:+.4f}")

    # Set output file path
    if output_file is None:
        output_file = dseed_dir / "synthetic_evaluation.csv"
    else:
        output_file = Path(output_file)

    # Run leaf alignment
    console.print(f"\n[bold green]{'═' * 70}[/bold green]")
    console.print(f"[bold green]{'Leaf Alignment Analysis':^70}[/bold green]")
    console.print(f"[bold green]{'═' * 70}[/bold green]\n")
    console.print(f"  Training on: Synthetic data ({len(X_synthetic):,} samples)")
    console.print(f"  Testing on: Real test data ({len(X_test):,} samples)")
    console.print(f"  Using: Hyperparameters tuned on real training data")

    leaf_results = run_leaf_alignment(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        X_real_test=X_test,
        y_real_test=y_test,
        lgbm_params=lgbm_params,
        output_file=output_file,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=seed,
    )

    # Display summary
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print(f"[bold cyan]{'Synthetic Data Evaluation Summary':^70}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", justify="right", width=20)
    table.add_column("Percentage", justify="right", width=15)

    total_points = leaf_results["n_total"]
    table.add_row("Total synthetic points", f"{total_points:,}", "100.0%")
    table.add_row(
        "[green]Beneficial points[/green]",
        f"{leaf_results['n_beneficial']:,}",
        f"{leaf_results['pct_beneficial']:.1f}%",
    )
    table.add_row(
        "[yellow]Hallucinated points[/yellow]",
        f"{leaf_results['n_hallucinated']:,}",
        f"{leaf_results['pct_hallucinated']:.1f}%",
    )
    table.add_row(
        "[cyan]Uncertain points[/cyan]",
        f"{leaf_results['n_uncertain']:,}",
        f"{leaf_results['pct_uncertain']:.1f}%",
    )
    table.add_row("", "", "")
    table.add_row(
        "Mean utility",
        f"{leaf_results['mean_utility']:.4f}",
        "",
    )
    table.add_row(
        "Median utility",
        f"{leaf_results['median_utility']:.4f}",
        "",
    )

    console.print(table)

    # Save JSON summary
    summary_path = output_file.parent / f"{output_file.stem}_summary.json"
    summary_output = {
        "metadata": {
            "dseed": dseed_dir.name,
            "synthetic_file": synthetic_file.name,
            "created_at": datetime.now().isoformat(),
            "sdvaluation_version": "0.2.0",
        },
        "evaluation": {
            "n_estimators": n_estimators,
            "target_column": target_column,
            "seed": seed,
        },
        "model_performance": {
            "real_to_test": real_metrics,
            "synthetic_to_test": synth_metrics,
            "performance_gap": {
                "auroc": float(real_metrics['auroc'] - synth_metrics['auroc']),
                "f1": float(real_metrics['f1'] - synth_metrics['f1']),
                "precision": float(real_metrics['precision'] - synth_metrics['precision']),
                "recall": float(real_metrics['recall'] - synth_metrics['recall']),
            },
        },
        "leaf_alignment": leaf_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2)

    console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
    console.print(f"[green]Per-point utilities saved to: {output_file}[/green]")
    console.print(f"[green]Summary statistics saved to: {summary_path}[/green]")

    return summary_output


def run_leaf_alignment_baseline(
    dseed_dir: Path,
    target_column: str = "READMIT",
    n_estimators: int = 500,
    n_jobs: int = 1,
    random_state: int = 42,
    cross_test: bool = False,
) -> Dict[str, Any]:
    """
    Run leaf alignment baseline analysis on real training data.

    Evaluates how well real training data represents real test data using
    leaf co-occurrence analysis. Runs both deployment and optimal scenarios.
    Optionally runs cross-test scenarios for decomposition analysis.

    Args:
        dseed_dir: Path to dseed directory
        target_column: Name of target column
        n_estimators: Number of trees for leaf alignment
        n_jobs: Number of parallel jobs
        random_state: Random seed
        cross_test: If True, also run cross-test scenarios (unsampled+optimal, training+deployment)

    Returns:
        Dictionary with baseline results
    """
    dseed_dir = Path(dseed_dir)

    # Discover files
    console.print(f"\n[bold]Discovering files in {dseed_dir.name}...[/bold]")
    discovery = DseedFileDiscovery(dseed_dir)
    console.print(discovery)

    # Check for required files
    if discovery.files["test"] is None:
        raise FileNotFoundError(
            f"Test data not found in {dseed_dir}. "
            "Leaf alignment requires test data for evaluation."
        )

    # Load hyperparameters
    hyperparams_path = dseed_dir / "hyperparams.json"
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"hyperparams.json not found in {dseed_dir}. "
            "Please run 'sdvaluation tune' first to generate hyperparameters."
        )

    console.print(f"\n[bold]Loading hyperparameters...[/bold]")
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    deployment_params = hyperparams["deployment"]["lgbm_params"]
    deployment_threshold = hyperparams["deployment"]["optimal_threshold"]
    optimal_params = hyperparams["optimal"]["lgbm_params"]
    optimal_threshold = hyperparams["optimal"]["optimal_threshold"]

    console.print(f"  ✓ Deployment params loaded (threshold: {deployment_threshold:.3f})")
    console.print(f"  ✓ Optimal params loaded (threshold: {optimal_threshold:.3f})")

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

    console.print("[cyan]Loading test data...[/cyan]")
    X_test, y_test = load_and_encode_data(
        discovery.get_file("test"),
        discovery.get_file("encoding"),
        target_column,
    )
    console.print(f"  Samples: {len(X_test):,}")

    # Run deployment baseline
    console.print(f"\n[bold yellow]Scenario 1: Deployment Baseline (Unsampled → Test)[/bold yellow]")
    console.print(f"  Training {n_estimators} trees on {len(X_deployment):,} unsampled points...")

    deployment_output = dseed_dir / "leaf_alignment_deployment_baseline.csv"
    deployment_results = run_leaf_alignment(
        X_synthetic=X_deployment,
        y_synthetic=y_deployment,
        X_real_test=X_test,
        y_real_test=y_test,
        lgbm_params=deployment_params,
        output_file=deployment_output,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    console.print(f"\n[bold]Deployment Baseline Results:[/bold]")
    console.print(f"  Median Utility:    {deployment_results['median_utility']:.4f}")
    console.print(f"  Mean Utility:      {deployment_results['mean_utility']:.4f}")
    console.print(f"  Hallucinated Points: {deployment_results['n_hallucinated']:,}/{deployment_results['n_total']:,} "
                 f"({deployment_results['pct_hallucinated']:.1f}%)")
    console.print(f"  Beneficial Points:   {deployment_results['n_beneficial']:,}/{deployment_results['n_total']:,} "
                 f"({deployment_results['pct_beneficial']:.1f}%)")
    console.print(f"  Results saved to:  {deployment_output.name}")

    # Evaluate on test data
    console.print(f"\n[cyan]Evaluating deployment model on test data...[/cyan]")
    deployment_test_metrics = evaluate_on_test(
        params=deployment_params,
        X_train=X_deployment,
        y_train=y_deployment,
        X_test=X_test,
        y_test=y_test,
        threshold=deployment_threshold,
        seed=random_state,
    )

    # Run optimal baseline
    console.print(f"\n[bold green]Scenario 2: Optimal Baseline (Training → Test)[/bold green]")
    console.print(f"  Training {n_estimators} trees on {len(X_optimal):,} training points...")

    optimal_output = dseed_dir / "leaf_alignment_optimal_baseline.csv"
    optimal_results = run_leaf_alignment(
        X_synthetic=X_optimal,
        y_synthetic=y_optimal,
        X_real_test=X_test,
        y_real_test=y_test,
        lgbm_params=optimal_params,
        output_file=optimal_output,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    console.print(f"\n[bold]Optimal Baseline Results:[/bold]")
    console.print(f"  Median Utility:    {optimal_results['median_utility']:.4f}")
    console.print(f"  Mean Utility:      {optimal_results['mean_utility']:.4f}")
    console.print(f"  Hallucinated Points: {optimal_results['n_hallucinated']:,}/{optimal_results['n_total']:,} "
                 f"({optimal_results['pct_hallucinated']:.1f}%)")
    console.print(f"  Beneficial Points:   {optimal_results['n_beneficial']:,}/{optimal_results['n_total']:,} "
                 f"({optimal_results['pct_beneficial']:.1f}%)")
    console.print(f"  Results saved to:  {optimal_output.name}")

    # Evaluate on test data
    console.print(f"\n[cyan]Evaluating optimal model on test data...[/cyan]")
    optimal_test_metrics = evaluate_on_test(
        params=optimal_params,
        X_train=X_optimal,
        y_train=y_optimal,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold,
        seed=random_state,
    )

    # Run cross-test scenarios if requested
    cross_a_results = None
    cross_a_test_metrics = None
    cross_b_results = None
    cross_b_test_metrics = None

    if cross_test:
        # Scenario 3: Unsampled data + Optimal hyperparams
        console.print(f"\n[bold blue]Scenario 3: Cross-test A (Unsampled → Test with Optimal Params)[/bold blue]")
        console.print(f"  Training {n_estimators} trees on {len(X_deployment):,} unsampled points...")
        console.print(f"  [cyan]Using optimal hyperparameters (tuned on training data)[/cyan]")

        cross_a_output = dseed_dir / "leaf_alignment_cross_a.csv"
        cross_a_results = run_leaf_alignment(
            X_synthetic=X_deployment,
            y_synthetic=y_deployment,
            X_real_test=X_test,
            y_real_test=y_test,
            lgbm_params=optimal_params,  # Using optimal params
            output_file=cross_a_output,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        console.print(f"\n[bold]Cross-test A Results:[/bold]")
        console.print(f"  Median Utility:    {cross_a_results['median_utility']:.4f}")
        console.print(f"  Mean Utility:      {cross_a_results['mean_utility']:.4f}")
        console.print(f"  Hallucinated Points: {cross_a_results['n_hallucinated']:,}/{cross_a_results['n_total']:,} "
                     f"({cross_a_results['pct_hallucinated']:.1f}%)")
        console.print(f"  Beneficial Points:   {cross_a_results['n_beneficial']:,}/{cross_a_results['n_total']:,} "
                     f"({cross_a_results['pct_beneficial']:.1f}%)")
        console.print(f"  Results saved to:  {cross_a_output.name}")

        console.print(f"\n[cyan]Evaluating cross-test A model on test data...[/cyan]")
        cross_a_test_metrics = evaluate_on_test(
            params=optimal_params,
            X_train=X_deployment,
            y_train=y_deployment,
            X_test=X_test,
            y_test=y_test,
            threshold=optimal_threshold,
            seed=random_state,
        )

        # Scenario 4: Training data + Deployment hyperparams
        console.print(f"\n[bold purple]Scenario 4: Cross-test B (Training → Test with Deployment Params)[/bold purple]")
        console.print(f"  Training {n_estimators} trees on {len(X_optimal):,} training points...")
        console.print(f"  [cyan]Using deployment hyperparameters (tuned on unsampled data)[/cyan]")

        cross_b_output = dseed_dir / "leaf_alignment_cross_b.csv"
        cross_b_results = run_leaf_alignment(
            X_synthetic=X_optimal,
            y_synthetic=y_optimal,
            X_real_test=X_test,
            y_real_test=y_test,
            lgbm_params=deployment_params,  # Using deployment params
            output_file=cross_b_output,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        console.print(f"\n[bold]Cross-test B Results:[/bold]")
        console.print(f"  Median Utility:    {cross_b_results['median_utility']:.4f}")
        console.print(f"  Mean Utility:      {cross_b_results['mean_utility']:.4f}")
        console.print(f"  Hallucinated Points: {cross_b_results['n_hallucinated']:,}/{cross_b_results['n_total']:,} "
                     f"({cross_b_results['pct_hallucinated']:.1f}%)")
        console.print(f"  Beneficial Points:   {cross_b_results['n_beneficial']:,}/{cross_b_results['n_total']:,} "
                     f"({cross_b_results['pct_beneficial']:.1f}%)")
        console.print(f"  Results saved to:  {cross_b_output.name}")

        console.print(f"\n[cyan]Evaluating cross-test B model on test data...[/cyan]")
        cross_b_test_metrics = evaluate_on_test(
            params=deployment_params,
            X_train=X_optimal,
            y_train=y_optimal,
            X_test=X_test,
            y_test=y_test,
            threshold=deployment_threshold,
            seed=random_state,
        )

    # Display test evaluation results
    console.print(f"\n{'='*80}")
    console.print(f"[bold magenta]Test Data Evaluation Results[/bold magenta]")
    console.print(f"{'='*80}")

    # Display deployment test evaluation
    display_test_evaluation(
        "Deployment Baseline (Unsampled → Test)",
        deployment_results["median_utility"],  # Use median utility as proxy for CV score
        deployment_test_metrics,
    )

    # Display optimal test evaluation
    display_test_evaluation(
        "Optimal Baseline (Training → Test)",
        optimal_results["median_utility"],  # Use median utility as proxy for CV score
        optimal_test_metrics,
    )

    # Display cross-test evaluations if available
    if cross_test:
        display_test_evaluation(
            "Cross-test A (Unsampled → Test with Optimal Params)",
            cross_a_results["median_utility"],
            cross_a_test_metrics,
        )

        display_test_evaluation(
            "Cross-test B (Training → Test with Deployment Params)",
            cross_b_results["median_utility"],
            cross_b_test_metrics,
        )

    # Enhanced comparison section
    console.print(f"\n{'='*80}")
    console.print(f"[bold magenta]Scenario Comparison[/bold magenta]")
    console.print(f"{'='*80}")

    # Utility comparison
    utility_gap = optimal_results["median_utility"] - deployment_results["median_utility"]
    gap_color = "green" if utility_gap > 0 else "red"
    console.print(f"\n[bold]Utility Scores:[/bold]")
    console.print(f"  Deployment Median: {deployment_results['median_utility']:.4f}")
    console.print(f"  Optimal Median:    {optimal_results['median_utility']:.4f}")
    console.print(f"  Gap (Opt - Dep):   [{gap_color}]{utility_gap:+.4f}[/{gap_color}]")

    # Hallucination comparison
    hallucinated_reduction = deployment_results["n_hallucinated"] - optimal_results["n_hallucinated"]
    console.print(f"\n[bold]Hallucinated Points:[/bold]")
    console.print(f"  Deployment: {deployment_results['n_hallucinated']:,}/{deployment_results['n_total']:,} ({deployment_results['pct_hallucinated']:.1f}%)")
    console.print(f"  Optimal:    {optimal_results['n_hallucinated']:,}/{optimal_results['n_total']:,} ({optimal_results['pct_hallucinated']:.1f}%)")

    if hallucinated_reduction > 0:
        reduction_pct = (hallucinated_reduction / deployment_results["n_hallucinated"] * 100) if deployment_results["n_hallucinated"] > 0 else 0
        console.print(f"  Reduction:  [green]{hallucinated_reduction:,} fewer points ({reduction_pct:.1f}% improvement)[/green]")
    elif hallucinated_reduction < 0:
        console.print(f"  Increase:   [red]{abs(hallucinated_reduction):,} more points[/red]")
    else:
        console.print(f"  Change:     Same in both scenarios")

    # Test metrics comparison
    console.print(f"\n[bold]Test Performance Metrics:[/bold]")

    # Create comparison table
    comparison_table = Table(show_header=True, title="Deployment vs Optimal")
    comparison_table.add_column("Metric", style="bold")
    comparison_table.add_column("Deployment", justify="right")
    comparison_table.add_column("Optimal", justify="right")
    comparison_table.add_column("Δ (Opt - Dep)", justify="right")

    # Add metrics rows
    metrics_to_compare = [
        ("AUROC", "auroc", True),
        ("F1 Score", "f1", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("FPR", "fpr", False),
        ("FNR", "fnr", False),
    ]

    for metric_name, metric_key, higher_is_better in metrics_to_compare:
        deploy_val = deployment_test_metrics[metric_key]
        optimal_val = optimal_test_metrics[metric_key]
        delta = optimal_val - deploy_val

        # Color the delta based on whether it's an improvement
        if higher_is_better:
            delta_color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
        else:
            delta_color = "green" if delta < 0 else "red" if delta > 0 else "yellow"

        comparison_table.add_row(
            metric_name,
            f"{deploy_val:.4f}",
            f"{optimal_val:.4f}",
            f"[{delta_color}]{delta:+.4f}[/{delta_color}]",
        )

    console.print(comparison_table)

    # Add decomposition analysis if cross-test is enabled
    if cross_test:
        console.print(f"\n{'='*80}")
        console.print(f"[bold cyan]Decomposition Analysis (2x2 Matrix)[/bold cyan]")
        console.print(f"{'='*80}")

        # Create 2x2 matrix table for utility scores
        matrix_table = Table(show_header=True, title="Utility Score Matrix")
        matrix_table.add_column("Data \\ Params", style="bold")
        matrix_table.add_column("Deployment Params", justify="right")
        matrix_table.add_column("Optimal Params", justify="right")

        matrix_table.add_row(
            "Unsampled Data",
            f"{deployment_results['median_utility']:.4f}",
            f"{cross_a_results['median_utility']:.4f}",
        )
        matrix_table.add_row(
            "Training Data",
            f"{cross_b_results['median_utility']:.4f}",
            f"{optimal_results['median_utility']:.4f}",
        )

        console.print(matrix_table)

        # Decomposition
        console.print(f"\n[bold]Effect Decomposition:[/bold]")

        # Pure data effect (holding params constant)
        data_effect_deployment_params = cross_b_results["median_utility"] - deployment_results["median_utility"]
        data_effect_optimal_params = optimal_results["median_utility"] - cross_a_results["median_utility"]
        avg_data_effect = (data_effect_deployment_params + data_effect_optimal_params) / 2

        # Pure hyperparameter effect (holding data constant)
        param_effect_unsampled_data = cross_a_results["median_utility"] - deployment_results["median_utility"]
        param_effect_training_data = optimal_results["median_utility"] - cross_b_results["median_utility"]
        avg_param_effect = (param_effect_unsampled_data + param_effect_training_data) / 2

        # Interaction effect
        total_gap = optimal_results["median_utility"] - deployment_results["median_utility"]
        interaction_effect = total_gap - (avg_data_effect + avg_param_effect)

        data_color = "green" if avg_data_effect > 0 else "red"
        param_color = "green" if avg_param_effect > 0 else "red"
        interaction_color = "green" if interaction_effect > 0 else "red" if interaction_effect < 0 else "yellow"

        console.print(f"\n  [bold]Pure Data Effect:[/bold] [{data_color}]{avg_data_effect:+.4f}[/{data_color}]")
        console.print(f"    (Training data vs Unsampled, averaged across both param sets)")
        console.print(f"    - With deployment params: {data_effect_deployment_params:+.4f}")
        console.print(f"    - With optimal params:    {data_effect_optimal_params:+.4f}")

        console.print(f"\n  [bold]Pure Hyperparameter Effect:[/bold] [{param_color}]{avg_param_effect:+.4f}[/{param_color}]")
        console.print(f"    (Optimal params vs Deployment params, averaged across both datasets)")
        console.print(f"    - On unsampled data: {param_effect_unsampled_data:+.4f}")
        console.print(f"    - On training data:  {param_effect_training_data:+.4f}")

        console.print(f"\n  [bold]Interaction Effect:[/bold] [{interaction_color}]{interaction_effect:+.4f}[/{interaction_color}]")
        console.print(f"    (Non-additive interaction between data and params)")

        console.print(f"\n  [bold]Total Gap:[/bold] {total_gap:+.4f}")
        console.print(f"    = Data Effect ({avg_data_effect:+.4f}) + Param Effect ({avg_param_effect:+.4f}) + Interaction ({interaction_effect:+.4f})")

        # Show percentage contributions
        if total_gap != 0:
            data_pct = (avg_data_effect / total_gap * 100)
            param_pct = (avg_param_effect / total_gap * 100)
            interaction_pct = (interaction_effect / total_gap * 100)

            console.print(f"\n  [bold]Contribution Breakdown:[/bold]")
            console.print(f"    Data quality:    {data_pct:>6.1f}%")
            console.print(f"    Hyperparameters: {param_pct:>6.1f}%")
            console.print(f"    Interaction:     {interaction_pct:>6.1f}%")

    # Save summary
    summary = {
        "metadata": {
            "dseed": dseed_dir.name,
            "created_at": datetime.now().isoformat(),
            "config": {
                "n_estimators": n_estimators,
                "target_column": target_column,
                "random_state": random_state,
                "cross_test": cross_test,
            },
        },
        "deployment_baseline": {
            "description": "Unsampled (40k) → Test (10k) with deployment hyperparameters",
            "training_samples": len(X_deployment),
            "test_samples": len(X_test),
            **deployment_results,
            "test_evaluation": deployment_test_metrics,
        },
        "optimal_baseline": {
            "description": "Training (10k) → Test (10k) with optimal hyperparameters",
            "training_samples": len(X_optimal),
            "test_samples": len(X_test),
            **optimal_results,
            "test_evaluation": optimal_test_metrics,
        },
        "comparison": {
            "utility_gap": float(utility_gap),
            "hallucinated_reduction": int(hallucinated_reduction),
            "hallucinated_reduction_pct": float(reduction_pct) if hallucinated_reduction > 0 else 0.0,
            "test_metrics_delta": {
                "auroc": float(optimal_test_metrics["auroc"] - deployment_test_metrics["auroc"]),
                "f1": float(optimal_test_metrics["f1"] - deployment_test_metrics["f1"]),
                "precision": float(optimal_test_metrics["precision"] - deployment_test_metrics["precision"]),
                "recall": float(optimal_test_metrics["recall"] - deployment_test_metrics["recall"]),
                "fpr": float(optimal_test_metrics["fpr"] - deployment_test_metrics["fpr"]),
                "fnr": float(optimal_test_metrics["fnr"] - deployment_test_metrics["fnr"]),
            },
        },
    }

    # Add cross-test results to summary if available
    if cross_test:
        summary["cross_test_a"] = {
            "description": "Unsampled (40k) → Test (10k) with optimal hyperparameters",
            "training_samples": len(X_deployment),
            "test_samples": len(X_test),
            **cross_a_results,
            "test_evaluation": cross_a_test_metrics,
        }
        summary["cross_test_b"] = {
            "description": "Training (10k) → Test (10k) with deployment hyperparameters",
            "training_samples": len(X_optimal),
            "test_samples": len(X_test),
            **cross_b_results,
            "test_evaluation": cross_b_test_metrics,
        }
        summary["decomposition"] = {
            "data_effect": {
                "average": float(avg_data_effect),
                "with_deployment_params": float(data_effect_deployment_params),
                "with_optimal_params": float(data_effect_optimal_params),
            },
            "hyperparameter_effect": {
                "average": float(avg_param_effect),
                "on_unsampled_data": float(param_effect_unsampled_data),
                "on_training_data": float(param_effect_training_data),
            },
            "interaction_effect": float(interaction_effect),
            "total_gap": float(total_gap),
            "contribution_percentages": {
                "data_quality_pct": float(data_pct) if total_gap != 0 else 0.0,
                "hyperparameters_pct": float(param_pct) if total_gap != 0 else 0.0,
                "interaction_pct": float(interaction_pct) if total_gap != 0 else 0.0,
            },
        }

    summary_path = dseed_dir / "leaf_alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold]Summary saved to: {summary_path.name}[/bold]")

    return summary
