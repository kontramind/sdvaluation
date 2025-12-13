"""
Leaf Co-Occurrence Analysis for Hallucination Detection.

Adaptation of "In-Run Shapley" for LightGBM: Identifies synthetic points that
create decision boundaries misaligned with real test data patterns.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from rich.console import Console
from scipy import stats

console = Console()


def calculate_leaf_utility(
    y_true: np.ndarray,
    leaf_value: float,
) -> float:
    """
    Calculate utility of a leaf based on how well it predicts real data.

    Args:
        y_true: True labels of real points in this leaf
        leaf_value: The prediction value from the tree leaf

    Returns:
        Utility score: positive = good prediction, negative = bad prediction
    """
    if len(y_true) == 0:
        return 0.0

    # For binary classification:
    # leaf_value > 0 → predicts positive (class 1)
    # leaf_value < 0 → predicts negative (class 0)
    predicted_class = 1 if leaf_value > 0 else 0
    accuracy = np.mean(y_true == predicted_class)

    # Convert to utility: 0.5 = random, 1.0 = perfect, 0.0 = terrible
    # Range: -0.5 (bad) to +0.5 (good)
    utility = accuracy - 0.5

    return utility


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

    right_val = get_leaf_value_from_tree(tree_node.get("right_child", {}), target_leaf_id)
    return right_val


def process_single_tree(
    tree_k: int,
    tree_dump: dict,
    synthetic_leaves_k: np.ndarray,
    real_leaves_k: np.ndarray,
    y_real_test: np.ndarray,
    n_synthetic: int,
    n_real_test: int,
    empty_leaf_penalty: float,
) -> np.ndarray:
    """
    Process a single tree and compute utility scores for synthetic points.

    Args:
        tree_k: Tree index
        tree_dump: Tree structure from booster dump
        synthetic_leaves_k: Leaf assignments for synthetic points in this tree
        real_leaves_k: Leaf assignments for real test points in this tree
        y_real_test: Real test labels
        n_synthetic: Number of synthetic points
        n_real_test: Number of real test points
        empty_leaf_penalty: Penalty for leaves with no real data

    Returns:
        Utility scores for this tree (shape: [n_synthetic])
    """
    utility_scores = np.zeros(n_synthetic)

    # Get unique leaves that contain real data
    unique_leaves = np.unique(real_leaves_k)

    for leaf_id in unique_leaves:
        # Find real points in this leaf
        real_mask = real_leaves_k == leaf_id
        real_indices = np.where(real_mask)[0]

        if len(real_indices) == 0:
            continue

        # Get leaf value (prediction contribution) from tree structure
        leaf_value = get_leaf_value_from_tree(tree_dump["tree_structure"], leaf_id)

        # Calculate utility: how well does this leaf classify real data?
        y_true_in_leaf = y_real_test[real_indices]
        leaf_utility = calculate_leaf_utility(y_true_in_leaf, leaf_value)

        # Weight by importance: leaves handling more real data are more important
        weight = len(real_indices) / n_real_test
        weighted_utility = leaf_utility * weight

        # Find synthetic points in this leaf
        synth_mask = synthetic_leaves_k == leaf_id
        synth_indices = np.where(synth_mask)[0]

        if len(synth_indices) > 0:
            # Distribute utility among synthetic points in this leaf
            score_per_point = weighted_utility / len(synth_indices)
            utility_scores[synth_indices] += score_per_point

    # Handle empty leaves (synthetic points in leaves with NO real data)
    synth_unique_leaves = np.unique(synthetic_leaves_k)
    empty_leaves = np.setdiff1d(synth_unique_leaves, unique_leaves)

    for leaf_id in empty_leaves:
        synth_mask = synthetic_leaves_k == leaf_id
        synth_indices = np.where(synth_mask)[0]

        if len(synth_indices) > 0:
            # Penalize: these synthetic points created regions with no real data
            utility_scores[synth_indices] += empty_leaf_penalty / len(synth_indices)

    return utility_scores


def compute_utility_scores(
    model: LGBMClassifier,
    X_synthetic: np.ndarray,
    X_real_test: np.ndarray,
    y_real_test: np.ndarray,
    empty_leaf_penalty: float = -1.0,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute utility scores for synthetic points based on leaf co-occurrence.

    Args:
        model: Trained LightGBM model
        X_synthetic: Synthetic training data features
        X_real_test: Real test data features
        y_real_test: Real test data labels
        empty_leaf_penalty: Penalty for synthetic points in leaves with no real data
        n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs)

    Returns:
        - Mean utility scores per synthetic point (shape: [n_synthetic])
        - Utility scores per tree per synthetic point (shape: [n_synthetic, n_trees])
    """
    n_synthetic = len(X_synthetic)
    n_real_test = len(X_real_test)

    # Get leaf assignments
    synthetic_leaves = model.predict(X_synthetic, pred_leaf=True)  # [n_synthetic, n_trees]
    real_leaves = model.predict(X_real_test, pred_leaf=True)  # [n_real, n_trees]

    n_trees = synthetic_leaves.shape[1]

    # Get the underlying Booster to access leaf values
    booster = model.booster_
    tree_dump_all = booster.dump_model()["tree_info"]

    # Process trees either sequentially or in parallel
    console.print("\n[bold]Computing leaf utilities...[/bold]")
    console.print(f"  Trees: {n_trees}")
    console.print(f"  Synthetic points: {n_synthetic:,}")
    console.print(f"  Real test points: {n_real_test:,}")

    if n_jobs == 1:
        # Sequential execution
        utility_per_tree_list = []
        for tree_k in range(n_trees):
            if tree_k % 20 == 0:
                console.print(f"  Processing tree {tree_k}/{n_trees}...")

            utility_scores = process_single_tree(
                tree_k,
                tree_dump_all[tree_k],
                synthetic_leaves[:, tree_k],
                real_leaves[:, tree_k],
                y_real_test,
                n_synthetic,
                n_real_test,
                empty_leaf_penalty,
            )
            utility_per_tree_list.append(utility_scores)
    else:
        # Parallel execution
        console.print(f"  Using parallel execution with n_jobs={n_jobs}")

        utility_per_tree_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single_tree)(
                tree_k,
                tree_dump_all[tree_k],
                synthetic_leaves[:, tree_k],
                real_leaves[:, tree_k],
                y_real_test,
                n_synthetic,
                n_real_test,
                empty_leaf_penalty,
            )
            for tree_k in range(n_trees)
        )

    # Convert list to array [n_synthetic, n_trees]
    utility_per_tree = np.column_stack(utility_per_tree_list)

    # Compute mean utility across trees
    mean_utility = np.mean(utility_per_tree, axis=1)

    return mean_utility, utility_per_tree


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


def run_leaf_alignment(
    X_synthetic: pd.DataFrame,
    y_synthetic: pd.Series,
    X_real_test: pd.DataFrame,
    y_real_test: pd.Series,
    lgbm_params: Dict,
    output_file: Optional[Path] = None,
    n_estimators: int = 500,
    empty_leaf_penalty: float = -1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict:
    """
    Run leaf alignment analysis to detect hallucinated synthetic points.

    Args:
        X_synthetic: Synthetic training features
        y_synthetic: Synthetic training labels
        X_real_test: Real test features
        y_real_test: Real test labels
        lgbm_params: LightGBM hyperparameters
        output_file: Path to save results CSV (optional)
        n_estimators: Number of trees (more = tighter CIs)
        empty_leaf_penalty: Penalty for empty leaves
        n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs)
        random_state: Random seed

    Returns:
        Dictionary with summary statistics
    """
    # Prepare parameters
    params = lgbm_params.copy()
    params.pop('imbalance_method', None)
    params.pop('early_stopping_rounds', None)
    params['n_estimators'] = n_estimators
    params['verbose'] = -1  # Suppress output

    # Handle class imbalance
    n_pos = np.sum(y_synthetic == 1)
    n_neg = np.sum(y_synthetic == 0)
    params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1.0

    # Train model
    console.print(f"  Training LGBM with {n_estimators} trees...")
    model = LGBMClassifier(**params)
    model.fit(X_synthetic, y_synthetic)
    console.print("  [green]✓ Model trained[/green]")

    # Compute utility scores
    mean_utility, utility_per_tree = compute_utility_scores(
        model,
        X_synthetic,
        X_real_test,
        y_real_test.values,
        empty_leaf_penalty,
        n_jobs
    )

    # Compute confidence intervals
    mean, se, ci_lower, ci_upper = compute_confidence_intervals(utility_per_tree)

    # Identify reliably hallucinated points
    reliably_hallucinated = ci_upper < 0
    n_hallucinated = np.sum(reliably_hallucinated)
    reliably_beneficial = ci_lower > 0
    n_beneficial = np.sum(reliably_beneficial)
    n_uncertain = len(mean) - n_hallucinated - n_beneficial

    console.print(f"  [green]✓ Hallucinated: {n_hallucinated:,} "
                 f"({100*n_hallucinated/len(mean):.1f}%)[/green]")
    console.print(f"    Beneficial: {n_beneficial:,} "
                 f"({100*n_beneficial/len(mean):.1f}%)")
    console.print(f"    Uncertain: {n_uncertain:,} "
                 f"({100*n_uncertain/len(mean):.1f}%)")

    # Create results dataframe
    results = pd.DataFrame({
        "synthetic_index": range(len(mean)),
        "utility_score": mean,
        "utility_se": se,
        "utility_ci_lower": ci_lower,
        "utility_ci_upper": ci_upper,
        "reliably_hallucinated": reliably_hallucinated,
    })

    # Save if output file specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)

    # Return summary stats
    return {
        'n_total': len(mean),
        'n_hallucinated': int(n_hallucinated),
        'n_beneficial': int(n_beneficial),
        'n_uncertain': int(n_uncertain),
        'pct_hallucinated': 100 * n_hallucinated / len(mean),
        'pct_beneficial': 100 * n_beneficial / len(mean),
        'pct_uncertain': 100 * n_uncertain / len(mean),
        'mean_utility': float(np.mean(mean)),
        'median_utility': float(np.median(mean)),
    }
