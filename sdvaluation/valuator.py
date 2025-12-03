"""
Data Shapley Valuator using LightGBM for MIMIC-III synthetic data evaluation.

This module implements the Truncated Monte Carlo Shapley algorithm for computing
data point values based on their marginal contribution to model performance.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from sklearn.metrics import roc_auc_score

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

console = Console()


class LGBMDataValuator:
    """
    Data Shapley valuator using LightGBM classifier.

    Computes the Shapley value for each training data point based on its
    marginal contribution to the model's ROC-AUC score on a test set.

    Attributes:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        lgbm_params: LightGBM hyperparameters
        random_state: Random seed for reproducibility
        X_train_original: Original training features (before encoding) for output
        n_train: Number of training samples
        shapley_values: Computed Shapley values for each training point
        shapley_std: Standard deviation of Shapley estimates
        shapley_se: Standard error of Shapley estimates
        shapley_ci_lower: Lower bound of 95% confidence interval
        shapley_ci_upper: Upper bound of 95% confidence interval
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        lgbm_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        X_train_original: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the LGBMDataValuator.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_test: Test feature matrix
            y_test: Test labels
            lgbm_params: Dictionary of LightGBM parameters
            random_state: Random seed for reproducibility
            X_train_original: Original training features before encoding (for output)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lgbm_params = self._process_lgbm_params(lgbm_params or {})
        self.random_state = random_state
        self.X_train_original = X_train_original
        self.n_train = len(X_train)

        # Results storage
        self.shapley_values: Optional[np.ndarray] = None
        self.shapley_std: Optional[np.ndarray] = None
        self.shapley_se: Optional[np.ndarray] = None
        self.shapley_ci_lower: Optional[np.ndarray] = None
        self.shapley_ci_upper: Optional[np.ndarray] = None

        np.random.seed(random_state)

    def _process_lgbm_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LightGBM parameters, handling special cases.

        Args:
            params: Raw parameter dictionary

        Returns:
            Processed parameter dictionary ready for LGBMClassifier
        """
        processed = params.copy()

        # Handle imbalance_method parameter
        if "imbalance_method" in processed:
            method = processed.pop("imbalance_method")
            if method == "scale_pos_weight":
                # Will be computed per-subset during training
                pass
            elif method == "class_weight":
                processed["class_weight"] = "balanced"

        # Remove parameters that shouldn't be passed to LGBMClassifier
        processed.pop("early_stopping_rounds", None)
        processed.pop("optimal_threshold", None)

        return processed

    def _train_and_evaluate(self, indices: np.ndarray, num_threads: int = 0) -> float:
        """
        Train LightGBM on a subset of training data and evaluate on test set.

        Args:
            indices: Array of indices to use from training data
            num_threads: Number of threads for LightGBM (0 = use default)

        Returns:
            ROC-AUC score on test set. Returns 0.0 if training fails or
            if the subset contains only one class.
        """
        if len(indices) == 0:
            return 0.0

        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]

        # Check if we have both classes
        if len(np.unique(y_subset)) < 2:
            return 0.0

        try:
            # Prepare parameters for this subset
            params = self.lgbm_params.copy()

            # Set num_threads if specified (for parallel coordination)
            if num_threads > 0:
                params["num_threads"] = num_threads

            # Compute scale_pos_weight if imbalance method was set
            if "class_weight" not in params:
                # Compute scale_pos_weight for this subset
                n_pos = np.sum(y_subset == 1)
                n_neg = np.sum(y_subset == 0)
                if n_pos > 0 and n_neg > 0:
                    params["scale_pos_weight"] = n_neg / n_pos

            # Train model
            model = LGBMClassifier(**params, random_state=self.random_state, verbose=-1)
            model.fit(X_subset, y_subset)

            # Evaluate on test set
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            auroc = roc_auc_score(self.y_test, y_pred_proba)

            return auroc

        except Exception as e:
            # Silently return 0.0 for failed training
            return 0.0

    def _compute_single_permutation(
        self,
        perm_idx: int,
        random_seed: int,
        max_coalition_size: int,
        num_threads: int = 0,
    ) -> List[List[float]]:
        """
        Compute contributions for a single permutation.

        This method is designed to be run in parallel - each permutation
        is completely independent.

        Args:
            perm_idx: Index of this permutation (for logging/debugging)
            random_seed: Random seed for this permutation
            max_coalition_size: Maximum coalition size to evaluate
            num_threads: Number of threads for LightGBM (0 = use default)

        Returns:
            List of contribution lists, one per training point.
            contributions[i] contains all marginal contributions for point i.
        """
        # Set seed for reproducibility
        np.random.seed(random_seed)

        # Generate random permutation
        permutation = np.random.permutation(self.n_train)

        # Track contributions for each point
        contributions = [[] for _ in range(self.n_train)]

        # Previous coalition performance
        prev_performance = 0.0

        # Evaluate contributions along the permutation
        for coalition_size in range(1, max_coalition_size + 1):
            # Current coalition
            coalition = permutation[:coalition_size]

            # Evaluate coalition performance
            current_performance = self._train_and_evaluate(coalition, num_threads)

            # Marginal contribution of the last added point
            added_point = permutation[coalition_size - 1]
            marginal_contribution = current_performance - prev_performance

            # Record contribution
            contributions[added_point].append(marginal_contribution)

            # Update for next iteration
            prev_performance = current_performance

        return contributions

    def compute_shapley_values(
        self,
        num_samples: int = 100,
        max_coalition_size: Optional[int] = None,
        show_progress: bool = True,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Compute Data Shapley values using Truncated Monte Carlo algorithm.

        The algorithm:
        1. For each permutation (num_samples times):
           - Generate random permutation of training indices
           - For each position in permutation:
             - Compute marginal contribution: V(S * {i}) - V(S)
             where S is the coalition before adding point i
        2. Average contributions across all permutations

        Args:
            num_samples: Number of random permutations to sample
            max_coalition_size: Maximum coalition size to evaluate (None = all)
            show_progress: Whether to show progress bar
            n_jobs: Number of parallel jobs. 1 = sequential (default),
                   -1 = use all CPUs, >1 = specific number of CPUs.
                   Falls back to sequential if parallelization fails.

        Returns:
            Array of Shapley values for each training point
        """
        if max_coalition_size is None:
            max_coalition_size = self.n_train

        max_coalition_size = min(max_coalition_size, self.n_train)

        # Track all contributions for each point for uncertainty quantification
        contributions_per_point = [[] for _ in range(self.n_train)]

        total_iterations = num_samples * max_coalition_size

        # Determine whether to use parallel execution
        use_parallel = n_jobs != 1 and num_samples > 1

        # Calculate optimal num_threads for LightGBM based on n_jobs
        import os
        total_cpus = os.cpu_count() or 1

        if use_parallel:
            # Divide threads among parallel jobs to avoid oversubscription
            effective_n_jobs = total_cpus if n_jobs == -1 else min(n_jobs, total_cpus)
            lgbm_threads = max(1, total_cpus // effective_n_jobs)
        else:
            # Sequential: let LightGBM use all cores
            lgbm_threads = total_cpus

        if use_parallel:
            try:
                if show_progress:
                    console.print(
                        f"[bold cyan]Using parallel execution with n_jobs={n_jobs}[/bold cyan]"
                    )
                    console.print(
                        f"[cyan]LightGBM threads per job: {lgbm_threads} "
                        f"(total CPUs: {total_cpus})[/cyan]"
                    )
                    console.print(
                        f"Computing Shapley values (N={self.n_train}, "
                        f"samples={num_samples}, max_coalition={max_coalition_size})..."
                    )
                    console.print(
                        f"[dim]Training {num_samples * max_coalition_size:,} models total. "
                        f"This may take several minutes...[/dim]"
                    )

                # Parallel execution: each permutation runs independently
                # Generate random seeds for reproducibility
                random_seeds = [self.random_state + i for i in range(num_samples)]

                # Use multiprocessing backend for true multi-core parallelism
                # Uses standard library multiprocessing (stable and fast)
                import time
                start_time = time.time()

                results = Parallel(
                    n_jobs=n_jobs,
                    backend="multiprocessing",
                    verbose=10,  # Print progress updates
                )(
                    delayed(self._compute_single_permutation)(
                        i, random_seeds[i], max_coalition_size, lgbm_threads
                    )
                    for i in range(num_samples)
                )

                elapsed = time.time() - start_time

                # Aggregate results from all permutations
                for perm_contributions in results:
                    for point_idx in range(self.n_train):
                        contributions_per_point[point_idx].extend(
                            perm_contributions[point_idx]
                        )

                if show_progress:
                    console.print(
                        f"[bold green]✓[/bold green] Completed {total_iterations:,} model trainings "
                        f"in {elapsed:.1f}s"
                    )

            except Exception as e:
                # Fallback to sequential execution
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Parallel execution failed: {e}"
                )
                console.print("[bold yellow]Falling back to sequential execution...[/bold yellow]")
                use_parallel = False

        if not use_parallel:
            # Sequential execution with progress bar
            if show_progress:
                console.print(
                    f"[cyan]LightGBM using {lgbm_threads} threads[/cyan]"
                )
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                )
                task = progress.add_task(
                    f"Computing Shapley values (N={self.n_train}, "
                    f"samples={num_samples}, max_coalition={max_coalition_size})...",
                    total=total_iterations,
                )
                progress.start()

            # Monte Carlo sampling
            for sample_idx in range(num_samples):
                # Generate random permutation
                permutation = np.random.permutation(self.n_train)

                # Previous coalition performance
                prev_performance = 0.0

                # Evaluate contributions along the permutation
                for coalition_size in range(1, max_coalition_size + 1):
                    # Current coalition
                    coalition = permutation[:coalition_size]

                    # Evaluate coalition performance
                    current_performance = self._train_and_evaluate(coalition, lgbm_threads)

                    # Marginal contribution of the last added point
                    added_point = permutation[coalition_size - 1]
                    marginal_contribution = current_performance - prev_performance

                    # Record contribution
                    contributions_per_point[added_point].append(marginal_contribution)

                    # Update for next iteration
                    prev_performance = current_performance

                    if show_progress:
                        progress.update(task, advance=1)

            if show_progress:
                progress.stop()

        # Compute Shapley values and uncertainty metrics
        self.shapley_values = np.array(
            [np.mean(contribs) if contribs else 0.0
             for contribs in contributions_per_point]
        )

        self.shapley_std = np.array(
            [np.std(contribs) if len(contribs) > 1 else 0.0
             for contribs in contributions_per_point]
        )

        self.shapley_se = np.array(
            [np.std(contribs) / np.sqrt(len(contribs)) if len(contribs) > 1 else 0.0
             for contribs in contributions_per_point]
        )

        # 95% confidence interval (z=1.96 for 95% CI)
        self.shapley_ci_lower = self.shapley_values - 1.96 * self.shapley_se
        self.shapley_ci_upper = self.shapley_values + 1.96 * self.shapley_se

        # Print summary statistics
        if show_progress:
            console.print("\n[bold green]Shapley Value Computation Complete[/bold green]")
            console.print(f"Total models trained: {total_iterations:,}")
            console.print(f"\n[bold]Summary Statistics:[/bold]")
            console.print(f"  Mean Shapley value: {np.mean(self.shapley_values):.6f}")
            console.print(f"  Std Shapley value:  {np.std(self.shapley_values):.6f}")
            console.print(f"  Min Shapley value:  {np.min(self.shapley_values):.6f}")
            console.print(f"  Max Shapley value:  {np.max(self.shapley_values):.6f}")
            console.print(
                f"  Mean uncertainty (SE): {np.mean(self.shapley_se):.6f}"
            )

            # Count harmful vs beneficial points
            n_harmful = np.sum(self.shapley_values < 0)
            n_beneficial = np.sum(self.shapley_values > 0)
            console.print(f"\n[bold]Value Distribution:[/bold]")
            console.print(f"  Harmful points (SV < 0):    {n_harmful:,} "
                         f"({100 * n_harmful / self.n_train:.1f}%)")
            console.print(f"  Beneficial points (SV > 0): {n_beneficial:,} "
                         f"({100 * n_beneficial / self.n_train:.1f}%)")

        return self.shapley_values

    def save_results(
        self,
        output_path: Path,
        include_features: bool = True,
    ) -> None:
        """
        Save Shapley values and uncertainty metrics to CSV.

        Results are sorted by Shapley value (most harmful first).
        Output format matches sdpype convention:
        - Original feature columns first (if include_features=True)
        - Target column
        - Shapley metrics
        - Sample index

        Args:
            output_path: Path to save CSV file
            include_features: Whether to include original feature values

        Raises:
            ValueError: If Shapley values haven't been computed yet
        """
        if self.shapley_values is None:
            raise ValueError(
                "Shapley values not computed. Run compute_shapley_values() first."
            )

        # Start with original features if available (sdpype format)
        if include_features and self.X_train_original is not None:
            results = self.X_train_original.copy()
        else:
            results = pd.DataFrame()

        # Add target column
        results["target"] = self.y_train.values

        # Add Shapley metrics
        results["shapley_value"] = self.shapley_values
        results["shapley_std"] = self.shapley_std
        results["shapley_se"] = self.shapley_se
        results["shapley_ci_lower"] = self.shapley_ci_lower
        results["shapley_ci_upper"] = self.shapley_ci_upper

        # Add sample index (matching sdpype)
        results["sample_index"] = range(self.n_train)

        # Sort by Shapley value (most harmful first)
        results = results.sort_values("shapley_value", ascending=True)

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

        console.print(f"\n[bold green]Results saved to:[/bold green] {output_path}")
        console.print(f"Total points: {len(results):,}")

    def get_top_k_harmful(self, k: int = 10) -> pd.DataFrame:
        """
        Get the k most harmful data points.

        Args:
            k: Number of points to return

        Returns:
            DataFrame with top k harmful points and their statistics

        Raises:
            ValueError: If Shapley values haven't been computed yet
        """
        if self.shapley_values is None:
            raise ValueError(
                "Shapley values not computed. Run compute_shapley_values() first."
            )

        # Get indices of k smallest Shapley values
        harmful_indices = np.argsort(self.shapley_values)[:k]

        results = pd.DataFrame(
            {
                "sample_index": harmful_indices,
                "shapley_value": self.shapley_values[harmful_indices],
                "shapley_std": self.shapley_std[harmful_indices],
                "shapley_se": self.shapley_se[harmful_indices],
                "shapley_ci_lower": self.shapley_ci_lower[harmful_indices],
                "shapley_ci_upper": self.shapley_ci_upper[harmful_indices],
                "target": self.y_train.iloc[harmful_indices].values,
            }
        )

        return results

    def get_top_k_beneficial(self, k: int = 10) -> pd.DataFrame:
        """
        Get the k most beneficial data points.

        Args:
            k: Number of points to return

        Returns:
            DataFrame with top k beneficial points and their statistics

        Raises:
            ValueError: If Shapley values haven't been computed yet
        """
        if self.shapley_values is None:
            raise ValueError(
                "Shapley values not computed. Run compute_shapley_values() first."
            )

        # Get indices of k largest Shapley values
        beneficial_indices = np.argsort(self.shapley_values)[-k:][::-1]

        results = pd.DataFrame(
            {
                "sample_index": beneficial_indices,
                "shapley_value": self.shapley_values[beneficial_indices],
                "shapley_std": self.shapley_std[beneficial_indices],
                "shapley_se": self.shapley_se[beneficial_indices],
                "shapley_ci_lower": self.shapley_ci_lower[beneficial_indices],
                "shapley_ci_upper": self.shapley_ci_upper[beneficial_indices],
                "target": self.y_train.iloc[beneficial_indices].values,
            }
        )

        return results
