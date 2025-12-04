"""
Core workflow for Data Shapley valuation.

This module provides the main entry point for running data valuation
on training and test datasets.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from scipy import stats

from .encoding import RDTDatasetEncoder, load_encoding_config
from .valuator import LGBMDataValuator

console = Console()


def _compute_proportion_ci(
    count: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    The Wilson score interval is more accurate than the normal approximation,
    especially for small sample sizes or proportions near 0 or 1.

    Args:
        count: Number of successes (e.g., number of harmful points)
        total: Total number of trials (e.g., total number of points)
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    if total == 0:
        return 0.0, 0.0

    p = count / total
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    lower = max(0.0, center - margin) * 100
    upper = min(1.0, center + margin) * 100

    return lower, upper


def run_data_valuation(
    train_file: Path,
    test_file: Path,
    target_column: str = "IS_READMISSION_30D",
    num_samples: int = 100,
    max_coalition_size: Optional[int] = None,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
    lgbm_params: Optional[Dict[str, Any]] = None,
    encoding_config: Optional[Path] = None,
    include_features: bool = True,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Run complete Data Shapley valuation workflow.

    This function orchestrates the entire workflow:
    1. Load training and test data
    2. Separate features from target
    3. Encode features (using RDT if config provided, else LabelEncoder)
    4. Create and run LGBMDataValuator
    5. Save results with timestamp
    6. Compute and display summary statistics

    Args:
        train_file: Path to training data CSV
        test_file: Path to test data CSV
        target_column: Name of target column (default: "IS_READMISSION_30D")
        num_samples: Number of random permutations for Shapley estimation
        max_coalition_size: Maximum coalition size (None = use all training data)
        random_state: Random seed for reproducibility
        output_dir: Directory to save results (default: ./output)
        lgbm_params: LightGBM hyperparameters
        encoding_config: Path to RDT encoding config YAML (None = use LabelEncoder)
        include_features: Whether to include original features in output CSV
        n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs)

    Returns:
        Dictionary containing:
            - valuator: The fitted LGBMDataValuator instance
            - results_file: Path to saved results CSV
            - shapley_values: Computed Shapley values
            - summary_stats: Dictionary of summary statistics

    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If target column is missing
    """
    # Convert paths
    train_file = Path(train_file)
    test_file = Path(test_file)
    if output_dir is None:
        output_dir = Path("output")
    output_dir = Path(output_dir)

    # Validate files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    console.print()
    console.print(Panel(
        "[bold cyan]Data Shapley Valuation for Synthetic Data[/bold cyan]",
        expand=False
    ))

    # ====================================================================
    # Step 1: Load Data
    # ====================================================================
    console.print("[bold]Step 1: Loading Data[/bold]")
    console.print(f"  Training file: {train_file}")
    console.print(f"  Test file:     {test_file}")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    console.print(f"  Training samples: {len(train_data):,}")
    console.print(f"  Test samples:     {len(test_data):,}")
    console.print(f"  Features:         {len(train_data.columns) - 1}")

    # Validate target column
    if target_column not in train_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data. "
            f"Available: {list(train_data.columns)}"
        )
    if target_column not in test_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in test data. "
            f"Available: {list(test_data.columns)}"
        )

    # ====================================================================
    # Step 2: Separate Features and Target
    # ====================================================================
    console.print("\n[bold]Step 2: Separating Features and Target[/bold]")

    X_train_original = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test_original = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Print class distribution
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)

    console.print(f"  Training class distribution:")
    console.print(
        f"    Positive (1): {train_pos:,} ({100 * train_pos / len(y_train):.1f}%)"
    )
    console.print(
        f"    Negative (0): {train_neg:,} ({100 * train_neg / len(y_train):.1f}%)"
    )
    console.print(f"  Test class distribution:")
    console.print(
        f"    Positive (1): {test_pos:,} ({100 * test_pos / len(y_test):.1f}%)"
    )
    console.print(
        f"    Negative (0): {test_neg:,} ({100 * test_neg / len(y_test):.1f}%)"
    )

    # ====================================================================
    # Step 3: Encode Features
    # ====================================================================
    console.print("\n[bold]Step 3: Encoding Features[/bold]")

    if encoding_config is not None:
        console.print(f"  Using RDT encoding: {encoding_config}")
        config = load_encoding_config(encoding_config)

        # Filter config to exclude target column
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

        console.print(
            f"  Filtered encoding config: {len(filtered_config['sdtypes'])} feature columns "
            f"(excluded target: {target_column})"
        )

        encoder = RDTDatasetEncoder(filtered_config)
        encoder.fit(X_train_original)
        X_train = encoder.transform(X_train_original)
        X_test = encoder.transform(X_test_original)

        # Reset indexes to ensure sequential integer indexing
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        X_train_original = X_train_original.reset_index(drop=True)
        X_test_original = X_test_original.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        console.print(
            f"  Encoded features: {len(X_train.columns)} "
            f"(from {len(X_train_original.columns)} original)"
        )
    else:
        console.print("  Using simple LabelEncoder for categorical features")
        from sklearn.preprocessing import LabelEncoder

        X_train = X_train_original.copy()
        X_test = X_test_original.copy()

        # Encode categorical columns
        for col in X_train.columns:
            if X_train[col].dtype == "object":
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

        # Reset indexes to ensure sequential integer indexing
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        X_train_original = X_train_original.reset_index(drop=True)
        X_test_original = X_test_original.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        console.print(f"  Features: {len(X_train.columns)}")

    # ====================================================================
    # Step 4: Create Valuator
    # ====================================================================
    console.print("\n[bold]Step 4: Creating Data Valuator[/bold]")

    if lgbm_params:
        console.print(f"  LightGBM parameters: {lgbm_params}")

    valuator = LGBMDataValuator(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lgbm_params=lgbm_params,
        random_state=random_state,
        X_train_original=X_train_original if include_features else None,
    )

    console.print(f"  Valuator created with {valuator.n_train:,} training points")

    # ====================================================================
    # Step 5: Compute Shapley Values
    # ====================================================================
    console.print("\n[bold]Step 5: Computing Shapley Values[/bold]")
    console.print(f"  Number of permutations: {num_samples}")
    console.print(
        f"  Max coalition size: "
        f"{max_coalition_size if max_coalition_size else valuator.n_train}"
    )

    shapley_values = valuator.compute_shapley_values(
        num_samples=num_samples,
        max_coalition_size=max_coalition_size,
        show_progress=True,
        n_jobs=n_jobs,
    )

    # ====================================================================
    # Step 6: Save Results
    # ====================================================================
    console.print("\n[bold]Step 6: Saving Results[/bold]")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"shapley_values_{timestamp}.csv"

    valuator.save_results(results_file, include_features=include_features)

    # ====================================================================
    # Step 7: Compute Additional Statistics
    # ====================================================================
    console.print("\n[bold]Step 7: Computing Summary Statistics[/bold]")

    # Count negative (harmful) points
    n_negative = np.sum(shapley_values < 0)
    n_positive = np.sum(shapley_values > 0)
    n_zero = np.sum(shapley_values == 0)

    # Compute confidence intervals for proportions
    neg_ci_lower, neg_ci_upper = _compute_proportion_ci(n_negative, valuator.n_train)
    pos_ci_lower, pos_ci_upper = _compute_proportion_ci(n_positive, valuator.n_train)

    console.print(f"\n[bold]Value Distribution:[/bold]")
    console.print(
        f"  Harmful (SV < 0):    {n_negative:,} "
        f"({100 * n_negative / valuator.n_train:.2f}%) "
        f"[95% CI: {neg_ci_lower:.2f}%-{neg_ci_upper:.2f}%]"
    )
    console.print(
        f"  Beneficial (SV > 0): {n_positive:,} "
        f"({100 * n_positive / valuator.n_train:.2f}%) "
        f"[95% CI: {pos_ci_lower:.2f}%-{pos_ci_upper:.2f}%]"
    )
    console.print(
        f"  Neutral (SV = 0):    {n_zero:,} "
        f"({100 * n_zero / valuator.n_train:.2f}%)"
    )

    # Count RELIABLY negative/positive (with statistical confidence)
    # Reliably harmful: entire 95% CI is below zero (CI upper bound < 0)
    # Reliably beneficial: entire 95% CI is above zero (CI lower bound > 0)
    n_reliable_negative = np.sum(valuator.shapley_ci_upper < 0)
    n_reliable_positive = np.sum(valuator.shapley_ci_lower > 0)
    n_uncertain = valuator.n_train - n_reliable_negative - n_reliable_positive

    # Compute confidence intervals for reliable proportions
    reliable_neg_ci_lower, reliable_neg_ci_upper = _compute_proportion_ci(
        n_reliable_negative, valuator.n_train
    )
    reliable_pos_ci_lower, reliable_pos_ci_upper = _compute_proportion_ci(
        n_reliable_positive, valuator.n_train
    )

    console.print(f"\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably harmful (CI upper < 0):    {n_reliable_negative:,} "
        f"({100 * n_reliable_negative / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_neg_ci_lower:.2f}%-{reliable_neg_ci_upper:.2f}%]"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0): {n_reliable_positive:,} "
        f"({100 * n_reliable_positive / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_pos_ci_lower:.2f}%-{reliable_pos_ci_upper:.2f}%]"
    )
    console.print(
        f"  Uncertain (CI spans 0):             {n_uncertain:,} "
        f"({100 * n_uncertain / valuator.n_train:.2f}%)"
    )

    # ====================================================================
    # Log-Loss Metric Statistics
    # ====================================================================
    console.print(f"\n[bold cyan]Log-Loss Metric Analysis:[/bold cyan]")

    # Count negative (harmful) points for Log-Loss
    n_negative_ll = np.sum(valuator.shapley_values_logloss < 0)
    n_positive_ll = np.sum(valuator.shapley_values_logloss > 0)
    n_zero_ll = np.sum(valuator.shapley_values_logloss == 0)

    # Compute confidence intervals for proportions
    neg_ci_lower_ll, neg_ci_upper_ll = _compute_proportion_ci(n_negative_ll, valuator.n_train)
    pos_ci_lower_ll, pos_ci_upper_ll = _compute_proportion_ci(n_positive_ll, valuator.n_train)

    console.print(f"\n[bold]Value Distribution:[/bold]")
    console.print(
        f"  Harmful (SV < 0):    {n_negative_ll:,} "
        f"({100 * n_negative_ll / valuator.n_train:.2f}%) "
        f"[95% CI: {neg_ci_lower_ll:.2f}%-{neg_ci_upper_ll:.2f}%]"
    )
    console.print(
        f"  Beneficial (SV > 0): {n_positive_ll:,} "
        f"({100 * n_positive_ll / valuator.n_train:.2f}%) "
        f"[95% CI: {pos_ci_lower_ll:.2f}%-{pos_ci_upper_ll:.2f}%]"
    )
    console.print(
        f"  Neutral (SV = 0):    {n_zero_ll:,} "
        f"({100 * n_zero_ll / valuator.n_train:.2f}%)"
    )

    # Count RELIABLY negative/positive for Log-Loss
    n_reliable_negative_ll = np.sum(valuator.shapley_ci_upper_logloss < 0)
    n_reliable_positive_ll = np.sum(valuator.shapley_ci_lower_logloss > 0)
    n_uncertain_ll = valuator.n_train - n_reliable_negative_ll - n_reliable_positive_ll

    # Compute confidence intervals for reliable proportions
    reliable_neg_ci_lower_ll, reliable_neg_ci_upper_ll = _compute_proportion_ci(
        n_reliable_negative_ll, valuator.n_train
    )
    reliable_pos_ci_lower_ll, reliable_pos_ci_upper_ll = _compute_proportion_ci(
        n_reliable_positive_ll, valuator.n_train
    )

    console.print(f"\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably harmful (CI upper < 0):    {n_reliable_negative_ll:,} "
        f"({100 * n_reliable_negative_ll / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_neg_ci_lower_ll:.2f}%-{reliable_neg_ci_upper_ll:.2f}%]"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0): {n_reliable_positive_ll:,} "
        f"({100 * n_reliable_positive_ll / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_pos_ci_lower_ll:.2f}%-{reliable_pos_ci_upper_ll:.2f}%]"
    )
    console.print(
        f"  Uncertain (CI spans 0):             {n_uncertain_ll:,} "
        f"({100 * n_uncertain_ll / valuator.n_train:.2f}%)"
    )

    # ====================================================================
    # F1 Score Metric Statistics
    # ====================================================================
    console.print(f"\n[bold yellow]F1 Score Metric Analysis:[/bold yellow]")

    # Count negative (harmful) points for F1 Score
    n_negative_f1 = np.sum(valuator.shapley_values_f1 < 0)
    n_positive_f1 = np.sum(valuator.shapley_values_f1 > 0)
    n_zero_f1 = np.sum(valuator.shapley_values_f1 == 0)

    # Compute confidence intervals for proportions
    neg_ci_lower_f1, neg_ci_upper_f1 = _compute_proportion_ci(n_negative_f1, valuator.n_train)
    pos_ci_lower_f1, pos_ci_upper_f1 = _compute_proportion_ci(n_positive_f1, valuator.n_train)

    console.print(f"\n[bold]Value Distribution:[/bold]")
    console.print(
        f"  Harmful (SV < 0):    {n_negative_f1:,} "
        f"({100 * n_negative_f1 / valuator.n_train:.2f}%) "
        f"[95% CI: {neg_ci_lower_f1:.2f}%-{neg_ci_upper_f1:.2f}%]"
    )
    console.print(
        f"  Beneficial (SV > 0): {n_positive_f1:,} "
        f"({100 * n_positive_f1 / valuator.n_train:.2f}%) "
        f"[95% CI: {pos_ci_lower_f1:.2f}%-{pos_ci_upper_f1:.2f}%]"
    )
    console.print(
        f"  Neutral (SV = 0):    {n_zero_f1:,} "
        f"({100 * n_zero_f1 / valuator.n_train:.2f}%)"
    )

    # Count RELIABLY negative/positive for F1 Score
    n_reliable_negative_f1 = np.sum(valuator.shapley_ci_upper_f1 < 0)
    n_reliable_positive_f1 = np.sum(valuator.shapley_ci_lower_f1 > 0)
    n_uncertain_f1 = valuator.n_train - n_reliable_negative_f1 - n_reliable_positive_f1

    # Compute confidence intervals for reliable proportions
    reliable_neg_ci_lower_f1, reliable_neg_ci_upper_f1 = _compute_proportion_ci(
        n_reliable_negative_f1, valuator.n_train
    )
    reliable_pos_ci_lower_f1, reliable_pos_ci_upper_f1 = _compute_proportion_ci(
        n_reliable_positive_f1, valuator.n_train
    )

    console.print(f"\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably harmful (CI upper < 0):    {n_reliable_negative_f1:,} "
        f"({100 * n_reliable_negative_f1 / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_neg_ci_lower_f1:.2f}%-{reliable_neg_ci_upper_f1:.2f}%]"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0): {n_reliable_positive_f1:,} "
        f"({100 * n_reliable_positive_f1 / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_pos_ci_lower_f1:.2f}%-{reliable_pos_ci_upper_f1:.2f}%]"
    )
    console.print(
        f"  Uncertain (CI spans 0):             {n_uncertain_f1:,} "
        f"({100 * n_uncertain_f1 / valuator.n_train:.2f}%)"
    )

    # ====================================================================
    # Precision Metric Statistics
    # ====================================================================
    console.print(f"\n[bold green]Precision Metric Analysis:[/bold green]")

    # Count negative (harmful) points for Precision
    n_negative_prec = np.sum(valuator.shapley_values_precision < 0)
    n_positive_prec = np.sum(valuator.shapley_values_precision > 0)
    n_zero_prec = np.sum(valuator.shapley_values_precision == 0)

    # Compute confidence intervals for proportions
    neg_ci_lower_prec, neg_ci_upper_prec = _compute_proportion_ci(n_negative_prec, valuator.n_train)
    pos_ci_lower_prec, pos_ci_upper_prec = _compute_proportion_ci(n_positive_prec, valuator.n_train)

    console.print(f"\n[bold]Value Distribution:[/bold]")
    console.print(
        f"  Harmful (SV < 0):    {n_negative_prec:,} "
        f"({100 * n_negative_prec / valuator.n_train:.2f}%) "
        f"[95% CI: {neg_ci_lower_prec:.2f}%-{neg_ci_upper_prec:.2f}%]"
    )
    console.print(
        f"  Beneficial (SV > 0): {n_positive_prec:,} "
        f"({100 * n_positive_prec / valuator.n_train:.2f}%) "
        f"[95% CI: {pos_ci_lower_prec:.2f}%-{pos_ci_upper_prec:.2f}%]"
    )
    console.print(
        f"  Neutral (SV = 0):    {n_zero_prec:,} "
        f"({100 * n_zero_prec / valuator.n_train:.2f}%)"
    )

    # Count RELIABLY negative/positive for Precision
    n_reliable_negative_prec = np.sum(valuator.shapley_ci_upper_precision < 0)
    n_reliable_positive_prec = np.sum(valuator.shapley_ci_lower_precision > 0)
    n_uncertain_prec = valuator.n_train - n_reliable_negative_prec - n_reliable_positive_prec

    # Compute confidence intervals for reliable proportions
    reliable_neg_ci_lower_prec, reliable_neg_ci_upper_prec = _compute_proportion_ci(
        n_reliable_negative_prec, valuator.n_train
    )
    reliable_pos_ci_lower_prec, reliable_pos_ci_upper_prec = _compute_proportion_ci(
        n_reliable_positive_prec, valuator.n_train
    )

    console.print(f"\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably harmful (CI upper < 0):    {n_reliable_negative_prec:,} "
        f"({100 * n_reliable_negative_prec / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_neg_ci_lower_prec:.2f}%-{reliable_neg_ci_upper_prec:.2f}%]"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0): {n_reliable_positive_prec:,} "
        f"({100 * n_reliable_positive_prec / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_pos_ci_lower_prec:.2f}%-{reliable_pos_ci_upper_prec:.2f}%]"
    )
    console.print(
        f"  Uncertain (CI spans 0):             {n_uncertain_prec:,} "
        f"({100 * n_uncertain_prec / valuator.n_train:.2f}%)"
    )

    # ====================================================================
    # Recall Metric Statistics
    # ====================================================================
    console.print(f"\n[bold blue]Recall Metric Analysis:[/bold blue]")

    # Count negative (harmful) points for Recall
    n_negative_rec = np.sum(valuator.shapley_values_recall < 0)
    n_positive_rec = np.sum(valuator.shapley_values_recall > 0)
    n_zero_rec = np.sum(valuator.shapley_values_recall == 0)

    # Compute confidence intervals for proportions
    neg_ci_lower_rec, neg_ci_upper_rec = _compute_proportion_ci(n_negative_rec, valuator.n_train)
    pos_ci_lower_rec, pos_ci_upper_rec = _compute_proportion_ci(n_positive_rec, valuator.n_train)

    console.print(f"\n[bold]Value Distribution:[/bold]")
    console.print(
        f"  Harmful (SV < 0):    {n_negative_rec:,} "
        f"({100 * n_negative_rec / valuator.n_train:.2f}%) "
        f"[95% CI: {neg_ci_lower_rec:.2f}%-{neg_ci_upper_rec:.2f}%]"
    )
    console.print(
        f"  Beneficial (SV > 0): {n_positive_rec:,} "
        f"({100 * n_positive_rec / valuator.n_train:.2f}%) "
        f"[95% CI: {pos_ci_lower_rec:.2f}%-{pos_ci_upper_rec:.2f}%]"
    )
    console.print(
        f"  Neutral (SV = 0):    {n_zero_rec:,} "
        f"({100 * n_zero_rec / valuator.n_train:.2f}%)"
    )

    # Count RELIABLY negative/positive for Recall
    n_reliable_negative_rec = np.sum(valuator.shapley_ci_upper_recall < 0)
    n_reliable_positive_rec = np.sum(valuator.shapley_ci_lower_recall > 0)
    n_uncertain_rec = valuator.n_train - n_reliable_negative_rec - n_reliable_positive_rec

    # Compute confidence intervals for reliable proportions
    reliable_neg_ci_lower_rec, reliable_neg_ci_upper_rec = _compute_proportion_ci(
        n_reliable_negative_rec, valuator.n_train
    )
    reliable_pos_ci_lower_rec, reliable_pos_ci_upper_rec = _compute_proportion_ci(
        n_reliable_positive_rec, valuator.n_train
    )

    console.print(f"\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably harmful (CI upper < 0):    {n_reliable_negative_rec:,} "
        f"({100 * n_reliable_negative_rec / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_neg_ci_lower_rec:.2f}%-{reliable_neg_ci_upper_rec:.2f}%]"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0): {n_reliable_positive_rec:,} "
        f"({100 * n_reliable_positive_rec / valuator.n_train:.2f}%) "
        f"[95% CI: {reliable_pos_ci_lower_rec:.2f}%-{reliable_pos_ci_upper_rec:.2f}%]"
    )
    console.print(
        f"  Uncertain (CI spans 0):             {n_uncertain_rec:,} "
        f"({100 * n_uncertain_rec / valuator.n_train:.2f}%)"
    )

    # Summary statistics
    summary_stats = {
        "n_train": valuator.n_train,
        # ROC-AUC metrics
        "n_harmful": n_negative,
        "n_beneficial": n_positive,
        "n_neutral": n_zero,
        "pct_harmful": 100 * n_negative / valuator.n_train,
        "pct_beneficial": 100 * n_positive / valuator.n_train,
        "harmful_ci_lower": neg_ci_lower,
        "harmful_ci_upper": neg_ci_upper,
        "beneficial_ci_lower": pos_ci_lower,
        "beneficial_ci_upper": pos_ci_upper,
        "n_reliable_harmful": n_reliable_negative,
        "n_reliable_beneficial": n_reliable_positive,
        "n_uncertain": n_uncertain,
        "pct_reliable_harmful": 100 * n_reliable_negative / valuator.n_train,
        "pct_reliable_beneficial": 100 * n_reliable_positive / valuator.n_train,
        "pct_uncertain": 100 * n_uncertain / valuator.n_train,
        "reliable_harmful_ci_lower": reliable_neg_ci_lower,
        "reliable_harmful_ci_upper": reliable_neg_ci_upper,
        "reliable_beneficial_ci_lower": reliable_pos_ci_lower,
        "reliable_beneficial_ci_upper": reliable_pos_ci_upper,
        "mean_shapley": np.mean(shapley_values),
        "std_shapley": np.std(shapley_values),
        "min_shapley": np.min(shapley_values),
        "max_shapley": np.max(shapley_values),
        "mean_uncertainty": np.mean(valuator.shapley_se),
        # Log-Loss metrics
        "n_harmful_logloss": n_negative_ll,
        "n_beneficial_logloss": n_positive_ll,
        "n_neutral_logloss": n_zero_ll,
        "pct_harmful_logloss": 100 * n_negative_ll / valuator.n_train,
        "pct_beneficial_logloss": 100 * n_positive_ll / valuator.n_train,
        "harmful_ci_lower_logloss": neg_ci_lower_ll,
        "harmful_ci_upper_logloss": neg_ci_upper_ll,
        "beneficial_ci_lower_logloss": pos_ci_lower_ll,
        "beneficial_ci_upper_logloss": pos_ci_upper_ll,
        "n_reliable_harmful_logloss": n_reliable_negative_ll,
        "n_reliable_beneficial_logloss": n_reliable_positive_ll,
        "n_uncertain_logloss": n_uncertain_ll,
        "pct_reliable_harmful_logloss": 100 * n_reliable_negative_ll / valuator.n_train,
        "pct_reliable_beneficial_logloss": 100 * n_reliable_positive_ll / valuator.n_train,
        "pct_uncertain_logloss": 100 * n_uncertain_ll / valuator.n_train,
        "reliable_harmful_ci_lower_logloss": reliable_neg_ci_lower_ll,
        "reliable_harmful_ci_upper_logloss": reliable_neg_ci_upper_ll,
        "reliable_beneficial_ci_lower_logloss": reliable_pos_ci_lower_ll,
        "reliable_beneficial_ci_upper_logloss": reliable_pos_ci_upper_ll,
        "mean_shapley_logloss": np.mean(valuator.shapley_values_logloss),
        "std_shapley_logloss": np.std(valuator.shapley_values_logloss),
        "min_shapley_logloss": np.min(valuator.shapley_values_logloss),
        "max_shapley_logloss": np.max(valuator.shapley_values_logloss),
        "mean_uncertainty_logloss": np.mean(valuator.shapley_se_logloss),
        # F1 Score metrics
        "n_harmful_f1": n_negative_f1,
        "n_beneficial_f1": n_positive_f1,
        "n_neutral_f1": n_zero_f1,
        "pct_harmful_f1": 100 * n_negative_f1 / valuator.n_train,
        "pct_beneficial_f1": 100 * n_positive_f1 / valuator.n_train,
        "harmful_ci_lower_f1": neg_ci_lower_f1,
        "harmful_ci_upper_f1": neg_ci_upper_f1,
        "beneficial_ci_lower_f1": pos_ci_lower_f1,
        "beneficial_ci_upper_f1": pos_ci_upper_f1,
        "n_reliable_harmful_f1": n_reliable_negative_f1,
        "n_reliable_beneficial_f1": n_reliable_positive_f1,
        "n_uncertain_f1": n_uncertain_f1,
        "pct_reliable_harmful_f1": 100 * n_reliable_negative_f1 / valuator.n_train,
        "pct_reliable_beneficial_f1": 100 * n_reliable_positive_f1 / valuator.n_train,
        "pct_uncertain_f1": 100 * n_uncertain_f1 / valuator.n_train,
        "reliable_harmful_ci_lower_f1": reliable_neg_ci_lower_f1,
        "reliable_harmful_ci_upper_f1": reliable_neg_ci_upper_f1,
        "reliable_beneficial_ci_lower_f1": reliable_pos_ci_lower_f1,
        "reliable_beneficial_ci_upper_f1": reliable_pos_ci_upper_f1,
        "mean_shapley_f1": np.mean(valuator.shapley_values_f1),
        "std_shapley_f1": np.std(valuator.shapley_values_f1),
        "min_shapley_f1": np.min(valuator.shapley_values_f1),
        "max_shapley_f1": np.max(valuator.shapley_values_f1),
        "mean_uncertainty_f1": np.mean(valuator.shapley_se_f1),
        # Precision metrics
        "n_harmful_precision": n_negative_prec,
        "n_beneficial_precision": n_positive_prec,
        "n_neutral_precision": n_zero_prec,
        "pct_harmful_precision": 100 * n_negative_prec / valuator.n_train,
        "pct_beneficial_precision": 100 * n_positive_prec / valuator.n_train,
        "harmful_ci_lower_precision": neg_ci_lower_prec,
        "harmful_ci_upper_precision": neg_ci_upper_prec,
        "beneficial_ci_lower_precision": pos_ci_lower_prec,
        "beneficial_ci_upper_precision": pos_ci_upper_prec,
        "n_reliable_harmful_precision": n_reliable_negative_prec,
        "n_reliable_beneficial_precision": n_reliable_positive_prec,
        "n_uncertain_precision": n_uncertain_prec,
        "pct_reliable_harmful_precision": 100 * n_reliable_negative_prec / valuator.n_train,
        "pct_reliable_beneficial_precision": 100 * n_reliable_positive_prec / valuator.n_train,
        "pct_uncertain_precision": 100 * n_uncertain_prec / valuator.n_train,
        "reliable_harmful_ci_lower_precision": reliable_neg_ci_lower_prec,
        "reliable_harmful_ci_upper_precision": reliable_neg_ci_upper_prec,
        "reliable_beneficial_ci_lower_precision": reliable_pos_ci_lower_prec,
        "reliable_beneficial_ci_upper_precision": reliable_pos_ci_upper_prec,
        "mean_shapley_precision": np.mean(valuator.shapley_values_precision),
        "std_shapley_precision": np.std(valuator.shapley_values_precision),
        "min_shapley_precision": np.min(valuator.shapley_values_precision),
        "max_shapley_precision": np.max(valuator.shapley_values_precision),
        "mean_uncertainty_precision": np.mean(valuator.shapley_se_precision),
        # Recall metrics
        "n_harmful_recall": n_negative_rec,
        "n_beneficial_recall": n_positive_rec,
        "n_neutral_recall": n_zero_rec,
        "pct_harmful_recall": 100 * n_negative_rec / valuator.n_train,
        "pct_beneficial_recall": 100 * n_positive_rec / valuator.n_train,
        "harmful_ci_lower_recall": neg_ci_lower_rec,
        "harmful_ci_upper_recall": neg_ci_upper_rec,
        "beneficial_ci_lower_recall": pos_ci_lower_rec,
        "beneficial_ci_upper_recall": pos_ci_upper_rec,
        "n_reliable_harmful_recall": n_reliable_negative_rec,
        "n_reliable_beneficial_recall": n_reliable_positive_rec,
        "n_uncertain_recall": n_uncertain_rec,
        "pct_reliable_harmful_recall": 100 * n_reliable_negative_rec / valuator.n_train,
        "pct_reliable_beneficial_recall": 100 * n_reliable_positive_rec / valuator.n_train,
        "pct_uncertain_recall": 100 * n_uncertain_rec / valuator.n_train,
        "reliable_harmful_ci_lower_recall": reliable_neg_ci_lower_rec,
        "reliable_harmful_ci_upper_recall": reliable_neg_ci_upper_rec,
        "reliable_beneficial_ci_lower_recall": reliable_pos_ci_lower_rec,
        "reliable_beneficial_ci_upper_recall": reliable_pos_ci_upper_rec,
        "mean_shapley_recall": np.mean(valuator.shapley_values_recall),
        "std_shapley_recall": np.std(valuator.shapley_values_recall),
        "min_shapley_recall": np.min(valuator.shapley_values_recall),
        "max_shapley_recall": np.max(valuator.shapley_values_recall),
        "mean_uncertainty_recall": np.mean(valuator.shapley_se_recall),
    }

    console.print()
    console.print(Panel(
        "[bold green]✓ Data Valuation Complete![/bold green]",
        expand=False,
        style="green"
    ))
    console.print()

    return {
        "valuator": valuator,
        "results_file": results_file,
        "shapley_values": shapley_values,
        "summary_stats": summary_stats,
    }
