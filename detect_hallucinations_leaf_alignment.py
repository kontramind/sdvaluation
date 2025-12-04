"""
Detect hallucinated synthetic data points using Leaf Co-occurrence Analysis.

Adaptation of "In-Run Shapley" for LightGBM: Identifies synthetic points that
create decision boundaries misaligned with real test data patterns.

Algorithm:
1. Train LGBM once on synthetic training data
2. Pass synthetic training + real test data through the model
3. For each leaf in each tree:
   - Calculate how well it classifies real test data (leaf utility)
   - Assign utility to synthetic points that fell into that leaf
4. Points with low utility = hallucinated (created bad decision boundaries)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.console import Console
from rich.table import Table
from scipy import stats

from sdvaluation.encoding import RDTDatasetEncoder, load_encoding_config

console = Console()


def load_lgbm_params(params_file: Path) -> dict:
    """Load LightGBM parameters from JSON file."""
    with open(params_file, "r") as f:
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


def calculate_leaf_utility(
    y_true: np.ndarray,
    leaf_value: float,
    is_binary: bool = True,
) -> float:
    """
    Calculate utility of a leaf based on how well it predicts real data.

    Args:
        y_true: True labels of real points in this leaf
        leaf_value: The prediction value from the tree leaf
        is_binary: Whether this is binary classification

    Returns:
        Utility score: positive = good prediction, negative = bad prediction
    """
    if len(y_true) == 0:
        return 0.0

    if is_binary:
        # For binary classification:
        # leaf_value > 0 → predicts positive (class 1)
        # leaf_value < 0 → predicts negative (class 0)
        predicted_class = 1 if leaf_value > 0 else 0
        accuracy = np.mean(y_true == predicted_class)

        # Convert to utility: 0.5 = random, 1.0 = perfect, 0.0 = terrible
        # Range: -0.5 (bad) to +0.5 (good)
        utility = accuracy - 0.5

        return utility

    return 0.0


def compute_utility_scores(
    model: LGBMClassifier,
    X_synthetic: np.ndarray,
    X_real_test: np.ndarray,
    y_real_test: np.ndarray,
    empty_leaf_penalty: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute utility scores for synthetic points based on leaf co-occurrence.

    Args:
        model: Trained LightGBM model
        X_synthetic: Synthetic training data features
        X_real_test: Real test data features
        y_real_test: Real test data labels
        empty_leaf_penalty: Penalty for synthetic points in leaves with no real data

    Returns:
        - Mean utility scores per synthetic point (shape: [n_synthetic])
        - Utility scores per tree per synthetic point (shape: [n_synthetic, n_trees])
    """
    n_synthetic = len(X_synthetic)

    # Get leaf assignments
    console.print("\n[bold]Getting leaf assignments...[/bold]")
    synthetic_leaves = model.predict(X_synthetic, pred_leaf=True)  # [n_synthetic, n_trees]
    real_leaves = model.predict(X_real_test, pred_leaf=True)  # [n_real, n_trees]

    n_trees = synthetic_leaves.shape[1]
    console.print(f"  Trees: {n_trees}")
    console.print(f"  Synthetic points: {n_synthetic:,}")
    console.print(f"  Real test points: {len(X_real_test):,}")

    # Initialize utility tracking
    # Track per-tree scores so we can compute confidence intervals
    utility_per_tree = np.zeros((n_synthetic, n_trees))

    # Get the underlying Booster to access leaf values
    booster = model.booster_

    console.print("\n[bold]Computing leaf utilities...[/bold]")

    for tree_k in range(n_trees):
        if tree_k % 20 == 0:
            console.print(f"  Processing tree {tree_k}/{n_trees}...")

        # Get leaf assignments for this tree
        real_leaves_k = real_leaves[:, tree_k]
        synth_leaves_k = synthetic_leaves[:, tree_k]

        # Get unique leaves that contain real data
        unique_leaves = np.unique(real_leaves_k)

        # Get tree structure to access leaf values
        tree_dump = booster.dump_model()["tree_info"][tree_k]

        for leaf_id in unique_leaves:
            # Find real points in this leaf
            real_mask = real_leaves_k == leaf_id
            real_indices = np.where(real_mask)[0]

            if len(real_indices) == 0:
                continue

            # Get leaf value (prediction contribution) from tree structure
            # Navigate tree structure to find this leaf
            leaf_value = get_leaf_value_from_tree(tree_dump["tree_structure"], leaf_id)

            # Calculate utility: how well does this leaf classify real data?
            y_true_in_leaf = y_real_test[real_indices]
            leaf_utility = calculate_leaf_utility(y_true_in_leaf, leaf_value)

            # Weight by importance: leaves handling more real data are more important
            weight = len(real_indices) / len(X_real_test)
            weighted_utility = leaf_utility * weight

            # Find synthetic points in this leaf
            synth_mask = synth_leaves_k == leaf_id
            synth_indices = np.where(synth_mask)[0]

            if len(synth_indices) > 0:
                # Distribute utility among synthetic points in this leaf
                score_per_point = weighted_utility / len(synth_indices)
                utility_per_tree[synth_indices, tree_k] += score_per_point

        # Handle empty leaves (synthetic points in leaves with NO real data)
        # These are potential hallucinations - regions where no real data exists
        synth_unique_leaves = np.unique(synth_leaves_k)
        empty_leaves = np.setdiff1d(synth_unique_leaves, unique_leaves)

        for leaf_id in empty_leaves:
            synth_mask = synth_leaves_k == leaf_id
            synth_indices = np.where(synth_mask)[0]

            if len(synth_indices) > 0:
                # Penalize: these synthetic points created regions with no real data
                utility_per_tree[synth_indices, tree_k] += empty_leaf_penalty / len(
                    synth_indices
                )

    # Compute mean utility across trees
    mean_utility = np.mean(utility_per_tree, axis=1)

    console.print("[green]✓ Leaf utility computation complete[/green]")

    return mean_utility, utility_per_tree


def get_leaf_value_from_tree(tree_node: dict, target_leaf_id: int) -> float:
    """
    Recursively find the leaf value for a given leaf_id in tree structure.

    Args:
        tree_node: Tree structure node from LightGBM dump_model
        target_leaf_id: The leaf_id we're looking for

    Returns:
        Leaf value (prediction contribution)
    """
    if "leaf_value" in tree_node:
        # This is a leaf node
        if tree_node.get("leaf_index") == target_leaf_id:
            return tree_node["leaf_value"]
        return 0.0  # Not the leaf we're looking for

    # Internal node - recurse
    left_val = get_leaf_value_from_tree(tree_node.get("left_child", {}), target_leaf_id)
    if left_val != 0.0:
        return left_val

    right_val = get_leaf_value_from_tree(
        tree_node.get("right_child", {}), target_leaf_id
    )
    return right_val


def compute_confidence_intervals(
    utility_per_tree: np.ndarray, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for utility scores.

    Args:
        utility_per_tree: Utility scores per tree (shape: [n_points, n_trees])
        confidence: Confidence level (default: 0.95)

    Returns:
        - Mean utility per point
        - Standard error per point
        - Lower confidence bound per point
        - Upper confidence bound per point
    """
    n_points, n_trees = utility_per_tree.shape

    mean = np.mean(utility_per_tree, axis=1)
    std = np.std(utility_per_tree, axis=1, ddof=1)
    se = std / np.sqrt(n_trees)

    # 95% confidence interval using t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, n_trees - 1)
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se

    return mean, se, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(
        description="Detect hallucinated synthetic data using leaf co-occurrence analysis"
    )
    parser.add_argument(
        "--synthetic-train",
        type=Path,
        required=True,
        help="Path to synthetic training data CSV",
    )
    parser.add_argument(
        "--real-test",
        type=Path,
        required=True,
        help="Path to real test data CSV (held-out)",
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
        "--output",
        type=Path,
        default=Path("hallucination_scores.csv"),
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="IS_READMISSION_30D",
        help="Name of target column",
    )
    parser.add_argument(
        "--empty-leaf-penalty",
        type=float,
        default=-1.0,
        help="Penalty for synthetic points in leaves with no real data",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Leaf Co-Occurrence Hallucination Detection      [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

    # Load data
    console.print("[bold]Step 1: Loading Data[/bold]")
    synthetic_data = pd.read_csv(args.synthetic_train)
    real_test_data = pd.read_csv(args.real_test)

    console.print(f"  Synthetic train: {len(synthetic_data):,} samples")
    console.print(f"  Real test:       {len(real_test_data):,} samples")

    # Separate features and target
    X_synthetic = synthetic_data.drop(columns=[args.target_column])
    y_synthetic = synthetic_data[args.target_column]
    X_real_test = real_test_data.drop(columns=[args.target_column])
    y_real_test = real_test_data[args.target_column]

    # Encode features
    console.print("\n[bold]Step 2: Encoding Features[/bold]")
    config = load_encoding_config(args.encoding_config)
    feature_columns = set(X_synthetic.columns)
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
    encoder.fit(X_synthetic)
    X_synthetic_encoded = encoder.transform(X_synthetic)
    X_real_test_encoded = encoder.transform(X_real_test)

    console.print(f"  Encoded features: {X_synthetic_encoded.shape[1]}")

    # Load LightGBM parameters
    lgbm_params = load_lgbm_params(args.lgbm_params)

    # Handle class imbalance
    n_pos = np.sum(y_synthetic == 1)
    n_neg = np.sum(y_synthetic == 0)
    lgbm_params["scale_pos_weight"] = n_neg / n_pos

    console.print("\n[bold]Step 3: Training LightGBM on Synthetic Data[/bold]")
    console.print(f"  Class distribution: {n_pos:,} positive, {n_neg:,} negative")
    console.print(f"  Scale pos weight: {lgbm_params['scale_pos_weight']:.2f}")

    model = LGBMClassifier(**lgbm_params, random_state=args.random_state, verbose=-1)
    model.fit(X_synthetic_encoded, y_synthetic)

    console.print("[green]✓ Model trained[/green]")

    # Compute utility scores
    console.print("\n[bold]Step 4: Computing Leaf Co-Occurrence Utilities[/bold]")
    mean_utility, utility_per_tree = compute_utility_scores(
        model,
        X_synthetic_encoded,
        X_real_test_encoded,
        y_real_test.values,
        args.empty_leaf_penalty,
    )

    # Compute confidence intervals
    console.print("\n[bold]Step 5: Computing Confidence Intervals[/bold]")
    mean, se, ci_lower, ci_upper = compute_confidence_intervals(utility_per_tree)

    # Identify reliably hallucinated points (CI upper < 0)
    reliably_hallucinated = ci_upper < 0
    n_hallucinated = np.sum(reliably_hallucinated)

    console.print(f"  Mean utility: {np.mean(mean):.6f}")
    console.print(f"  Std utility:  {np.std(mean):.6f}")
    console.print(f"  Min utility:  {np.min(mean):.6f}")
    console.print(f"  Max utility:  {np.max(mean):.6f}")

    # Create results dataframe
    results = pd.DataFrame(
        {
            "synthetic_index": range(len(synthetic_data)),
            "utility_score": mean,
            "utility_se": se,
            "utility_ci_lower": ci_lower,
            "utility_ci_upper": ci_upper,
            "reliably_hallucinated": reliably_hallucinated,
        }
    )

    # Save results
    console.print(f"\n[bold]Step 6: Saving Results[/bold]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    console.print(f"  Results saved to: {args.output}")

    # Display summary statistics
    console.print("\n[bold yellow]═══════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]                 Summary Statistics                 [/bold yellow]")
    console.print("[bold yellow]═══════════════════════════════════════════════════[/bold yellow]\n")

    # Value distribution
    n_negative = np.sum(mean < 0)
    n_positive = np.sum(mean > 0)
    n_zero = np.sum(mean == 0)

    console.print("[bold]Utility Score Distribution:[/bold]")
    console.print(
        f"  Negative utility (< 0): {n_negative:,} ({100*n_negative/len(mean):.2f}%)"
    )
    console.print(
        f"  Positive utility (> 0): {n_positive:,} ({100*n_positive/len(mean):.2f}%)"
    )
    console.print(f"  Zero utility (= 0):     {n_zero:,} ({100*n_zero/len(mean):.2f}%)")

    # Statistical confidence
    reliably_positive = ci_lower > 0
    n_reliable_positive = np.sum(reliably_positive)
    uncertain = ~reliably_hallucinated & ~reliably_positive
    n_uncertain = np.sum(uncertain)

    console.print("\n[bold]Statistical Confidence (95% CI-based):[/bold]")
    console.print(
        f"  Reliably hallucinated (CI upper < 0): [red]{n_hallucinated:,}[/red] "
        f"({100*n_hallucinated/len(mean):.2f}%)"
    )
    console.print(
        f"  Reliably beneficial (CI lower > 0):   [green]{n_reliable_positive:,}[/green] "
        f"({100*n_reliable_positive/len(mean):.2f}%)"
    )
    console.print(
        f"  Uncertain (CI spans 0):                {n_uncertain:,} "
        f"({100*n_uncertain/len(mean):.2f}%)"
    )

    # Show top hallucinations
    if n_hallucinated > 0:
        console.print("\n[bold red]Top 10 Reliably Hallucinated Points:[/bold red]")
        top_hallucinations = results[reliably_hallucinated].nsmallest(
            10, "utility_score"
        )

        table = Table(show_header=True)
        table.add_column("Index", justify="right")
        table.add_column("Utility Score", justify="right")
        table.add_column("Std Error", justify="right")
        table.add_column("95% CI", justify="center")

        for _, row in top_hallucinations.iterrows():
            table.add_row(
                f"{int(row['synthetic_index'])}",
                f"{row['utility_score']:.6f}",
                f"{row['utility_se']:.6f}",
                f"[{row['utility_ci_lower']:.6f}, {row['utility_ci_upper']:.6f}]",
            )

        console.print(table)
    else:
        console.print("\n[green]No reliably hallucinated points detected![/green]")

    console.print("\n[bold green]✓ Analysis Complete![/bold green]\n")


if __name__ == "__main__":
    main()
