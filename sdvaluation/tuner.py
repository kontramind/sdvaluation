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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.console import Console
from sklearn.metrics import (
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
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization with comprehensive search space.

        Args:
            trial: A trial object from Optuna

        Returns:
            Mean AUROC across folds
        """
        # Core hyperparameters
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
            "num_leaves": trial.suggest_int("num_leaves", 4, 100),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 2**(-10), 2**0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 60),
            "n_estimators": 1000,  # Large number, will use early stopping
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

        # Regularization (wider ranges for better exploration)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 10.0)  # L1
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 10.0)  # L2

        # Feature and sample sampling
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        # Leaf constraints
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 10, 50)

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

            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=0),  # Suppress output
                ],
            )

            # Predict on validation set
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            # Calculate AUROC
            auroc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auroc)

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
    )

    results = tuner.tune(show_progress=True)

    console.print(f"[green]✓ Best CV ROC-AUC: {results['best_cv_score']:.4f}[/green]")

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
        thresholds = np.arange(0.1, 0.9, 0.02)
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
        X_deployment,
        y_deployment,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
        seed=seed,
    )

    # Optimize threshold for deployment
    deployment_model = LGBMClassifier(**deployment_results["best_params"])
    deployment_threshold = optimize_threshold(
        deployment_model,
        X_deployment,
        y_deployment,
        metric=threshold_metric,
        n_folds=n_folds,
        seed=seed,
    )

    # Tune optimal scenario
    console.print(f"\n[bold green]Scenario 2: Optimal (training data)[/bold green]")
    optimal_results = optimize_hyperparameters(
        X_optimal,
        y_optimal,
        n_trials=n_trials,
        n_folds=n_folds,
        n_jobs=n_jobs,
        seed=seed,
    )

    # Optimize threshold for optimal
    optimal_model = LGBMClassifier(**optimal_results["best_params"])
    optimal_threshold = optimize_threshold(
        optimal_model,
        X_optimal,
        y_optimal,
        metric=threshold_metric,
        n_folds=n_folds,
        seed=seed,
    )

    # Evaluate on test data if available
    deployment_test_metrics = None
    optimal_test_metrics = None

    if discovery.files["test"] is not None:
        console.print(f"\n[bold magenta]Evaluating on Test Data[/bold magenta]")

        # Load test data
        console.print("[cyan]Loading test data...[/cyan]")
        X_test, y_test = load_and_encode_data(
            discovery.get_file("test"),
            discovery.get_file("encoding"),
            target_column,
        )
        console.print(f"  Samples: {len(X_test):,}")

        # Evaluate deployment model on test data
        deployment_test_metrics = evaluate_on_test(
            params=deployment_results["best_params"],
            X_train=X_deployment,
            y_train=y_deployment,
            X_test=X_test,
            y_test=y_test,
            threshold=deployment_threshold["optimal_threshold"],
            seed=seed,
        )

        # Evaluate optimal model on test data
        optimal_test_metrics = evaluate_on_test(
            params=optimal_results["best_params"],
            X_train=X_optimal,
            y_train=y_optimal,
            X_test=X_test,
            y_test=y_test,
            threshold=optimal_threshold["optimal_threshold"],
            seed=seed,
        )

        # Display results
        display_test_evaluation(
            "Deployment (Unsampled Params)",
            deployment_results["best_cv_score"],
            deployment_test_metrics,
        )

        display_test_evaluation(
            "Optimal (Training Params)",
            optimal_results["best_cv_score"],
            optimal_test_metrics,
        )

    # Compute parameter differences
    param_diff = {}
    for key in deployment_results["best_params"]:
        if key in ["random_state", "verbose"]:
            continue
        # Skip if parameter not in both sets (e.g., is_unbalance vs scale_pos_weight)
        if key not in optimal_results["best_params"]:
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
            **({"test_evaluation": deployment_test_metrics} if deployment_test_metrics else {}),
        },
        "optimal": {
            "description": "Hyperparameters tuned on training (real) data",
            "tuning_data": discovery.get_file("training").name,
            "tuning_samples": len(X_optimal),
            "best_cv_score": optimal_results["best_cv_score"],
            "lgbm_params": optimal_results["best_params"],
            **optimal_threshold,
            **({"test_evaluation": optimal_test_metrics} if optimal_test_metrics else {}),
        },
        "comparison": {
            "param_differences": param_diff,
            "cv_score_gap": float(
                optimal_results["best_cv_score"] - deployment_results["best_cv_score"]
            ),
            "threshold_gap": float(
                optimal_threshold["optimal_threshold"]
                - deployment_threshold["optimal_threshold"]
            ),
        },
    }

    # Save to file
    output_path = dseed_dir / output_name
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[bold green]✓ Tuning complete![/bold green]")
    console.print(f"[green]Results saved to: {output_path}[/green]")

    return output


def run_leaf_alignment_baseline(
    dseed_dir: Path,
    target_column: str = "READMIT",
    n_estimators: int = 500,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run leaf alignment baseline analysis on real training data.

    Evaluates how well real training data represents real test data using
    leaf co-occurrence analysis. Runs both deployment and optimal scenarios.

    Args:
        dseed_dir: Path to dseed directory
        target_column: Name of target column
        n_estimators: Number of trees for leaf alignment
        n_jobs: Number of parallel jobs
        random_state: Random seed

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
    optimal_params = hyperparams["optimal"]["lgbm_params"]

    console.print(f"  ✓ Deployment params loaded")
    console.print(f"  ✓ Optimal params loaded")

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
    console.print(f"  Harmful Points:    {deployment_results['harmful_count']:,}/{len(X_test):,} "
                 f"({deployment_results['harmful_pct']*100:.1f}%)")
    console.print(f"  Results saved to:  {deployment_output.name}")

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
    console.print(f"  Harmful Points:    {optimal_results['harmful_count']:,}/{len(X_test):,} "
                 f"({optimal_results['harmful_pct']*100:.1f}%)")
    console.print(f"  Results saved to:  {optimal_output.name}")

    # Compare scenarios
    console.print(f"\n[bold magenta]Comparison:[/bold magenta]")
    utility_gap = optimal_results["median_utility"] - deployment_results["median_utility"]
    harmful_reduction = deployment_results["harmful_count"] - optimal_results["harmful_count"]

    gap_color = "green" if utility_gap > 0 else "red"
    console.print(f"  Utility Gap (Optimal - Deployment): [{gap_color}]{utility_gap:+.4f}[/{gap_color}]")

    if harmful_reduction > 0:
        reduction_pct = (harmful_reduction / deployment_results["harmful_count"] * 100) if deployment_results["harmful_count"] > 0 else 0
        console.print(f"  Harmful Reduction: [green]{harmful_reduction:,} fewer points ({reduction_pct:.1f}% improvement)[/green]")
    elif harmful_reduction < 0:
        console.print(f"  Harmful Increase: [red]{abs(harmful_reduction):,} more points[/red]")
    else:
        console.print(f"  Harmful Count: Same in both scenarios")

    # Save summary
    summary = {
        "metadata": {
            "dseed": dseed_dir.name,
            "created_at": datetime.now().isoformat(),
            "config": {
                "n_estimators": n_estimators,
                "target_column": target_column,
                "random_state": random_state,
            },
        },
        "deployment_baseline": {
            "description": "Unsampled (40k) → Test (10k) with deployment hyperparameters",
            "training_samples": len(X_deployment),
            "test_samples": len(X_test),
            **deployment_results,
        },
        "optimal_baseline": {
            "description": "Training (10k) → Test (10k) with optimal hyperparameters",
            "training_samples": len(X_optimal),
            "test_samples": len(X_test),
            **optimal_results,
        },
        "comparison": {
            "utility_gap": float(utility_gap),
            "harmful_reduction": int(harmful_reduction),
            "harmful_reduction_pct": float(reduction_pct) if harmful_reduction > 0 else 0.0,
        },
    }

    summary_path = dseed_dir / "leaf_alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold]Summary saved to: {summary_path.name}[/bold]")

    return summary
