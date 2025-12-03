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

    console.print("\n[bold cyan]PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP[/bold cyan]")
    console.print("[bold cyan]   Data Shapley Valuation for Synthetic Data[/bold cyan]")
    console.print("[bold cyan]PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP[/bold cyan]\n")

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

    # Summary statistics
    summary_stats = {
        "n_train": valuator.n_train,
        "n_harmful": n_negative,
        "n_beneficial": n_positive,
        "n_neutral": n_zero,
        "pct_harmful": 100 * n_negative / valuator.n_train,
        "pct_beneficial": 100 * n_positive / valuator.n_train,
        "harmful_ci_lower": neg_ci_lower,
        "harmful_ci_upper": neg_ci_upper,
        "beneficial_ci_lower": pos_ci_lower,
        "beneficial_ci_upper": pos_ci_upper,
        "mean_shapley": np.mean(shapley_values),
        "std_shapley": np.std(shapley_values),
        "min_shapley": np.min(shapley_values),
        "max_shapley": np.max(shapley_values),
        "mean_uncertainty": np.mean(valuator.shapley_se),
    }

    console.print("\n[bold green]PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP[/bold green]")
    console.print("[bold green]   Data Valuation Complete![/bold green]")
    console.print("[bold green]PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP[/bold green]\n")

    return {
        "valuator": valuator,
        "results_file": results_file,
        "shapley_values": shapley_values,
        "summary_stats": summary_stats,
    }
