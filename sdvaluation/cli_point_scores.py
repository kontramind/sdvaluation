"""
Point-level scoring CLI for synthetic data quality assessment.

This module provides modular commands for computing per-point quality metrics
on synthetic data. Each metric can be run independently and outputs both
CSV (per-point scores) and JSON (summary statistics).

Commands:
- leaf-alignment: Detect classification-harmful synthetic points
- (future) alpha-precision: Detect geometrically implausible points
- (future) prdc-density: Measure real neighbor density
- (future) combined: Multi-metric combined scoring
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="point-scores",
    help="Compute per-point quality scores for synthetic data",
)
console = Console()


@app.command(name="leaf-alignment")
def leaf_alignment_scoring(
    dseed_dir: Path = typer.Option(
        ...,
        "-d",
        "--dseed-dir",
        help="Path to dseed directory containing hyperparams.json, real test data, and encoding config",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    synthetic_file: Path = typer.Option(
        ...,
        "-s",
        "--synthetic-file",
        help="Path to synthetic data CSV file to evaluate",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target_column: str = typer.Option(
        "READMIT",
        "-c",
        "--target-column",
        help="Name of the target column",
    ),
    n_estimators: int = typer.Option(
        500,
        "--n-estimators",
        help="Number of trees for leaf alignment (more = tighter confidence intervals)",
        min=100,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output CSV file path (default: dseed_dir/point_scores/leaf_alignment_scores.csv)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """
    Compute leaf alignment scores for synthetic data points.

    This command evaluates how well each synthetic data point aligns with real
    test data in the trained model's decision space. Points that create
    misleading decision boundaries are flagged as harmful.

    Prerequisites:
    - Run 'sdvaluation tune --dseed-dir <dir>' first to generate hyperparams.json

    Workflow:
    1. Auto-discover files in dseed directory (hyperparams.json, test, encoding)
    2. Load pre-tuned LightGBM hyperparameters from hyperparams.json
    3. Evaluate model performance (Real vs Synthetic baseline comparison)
    4. Compute leaf co-occurrence utility scores
    5. Output per-point scores (CSV) and summary statistics (JSON)

    Output files:
    - <output>.csv: Per-point utility scores with confidence intervals
    - <output>.json: Aggregate statistics and metadata

    Interpretation:
    - Reliably beneficial (CI lower > 0): Points that help the classifier
    - Reliably harmful (CI upper < 0): Points that hurt the classifier (hallucinations)
    - Uncertain (CI spans 0): Ambiguous contribution

    Example usage:

        # First, tune hyperparameters on real data
        sdvaluation tune --dseed-dir dseed55/

        # Then compute leaf alignment scores
        sdvaluation point-scores leaf-alignment \\
            --dseed-dir dseed55/ \\
            --synthetic-file synth_10k.csv

        # Custom output path and more trees
        sdvaluation point-scores leaf-alignment \\
            --dseed-dir dseed55/ \\
            --synthetic-file synth_10k.csv \\
            --output results/my_evaluation.csv \\
            --n-estimators 10000
    """
    from .tuner import (
        run_leaf_alignment_workflow,
        display_test_evaluation,
        DseedFileDiscovery,
    )

    start_time = time.time()

    try:
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Point-Level Scoring: Leaf Alignment[/bold cyan]")
        console.print("=" * 60)

        # =====================================================================
        # Step 1: File Discovery (display only)
        # =====================================================================
        console.print("\n[bold]Step 1: File Discovery[/bold]")

        discovery = DseedFileDiscovery(dseed_dir)
        console.print(f"  [green]✓[/green] Found encoding config: {discovery.files['encoding'].name}")
        console.print(f"  [green]✓[/green] Found training data: {discovery.files['training'].name}")

        if discovery.files['test'] is None:
            console.print("[bold red]Error:[/bold red] Test data not found in dseed directory")
            raise typer.Exit(code=1)
        console.print(f"  [green]✓[/green] Found test data: {discovery.files['test'].name}")

        hyperparams_path = dseed_dir / "hyperparams.json"
        if not hyperparams_path.exists():
            console.print(f"[bold red]Error:[/bold red] hyperparams.json not found in {dseed_dir}")
            console.print("  Please run 'sdvaluation tune --dseed-dir <dir>' first.")
            raise typer.Exit(code=1)
        console.print(f"  [green]✓[/green] Found hyperparams.json")

        # =====================================================================
        # Step 2: Set up output path
        # =====================================================================
        if output is None:
            output_dir = dseed_dir / "point_scores"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / "leaf_alignment_scores.csv"
        else:
            output_csv = Path(output)
            output_csv.parent.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # Step 3: Run shared workflow (identical to eval command)
        # =====================================================================
        console.print("\n[bold]Step 2: Running Leaf Alignment Workflow[/bold]")
        console.print("  This uses the same code path as 'sdvaluation eval' for consistency.")

        results = run_leaf_alignment_workflow(
            dseed_dir=dseed_dir,
            synthetic_file=synthetic_file,
            target_column=target_column,
            n_estimators=n_estimators,
            seed=seed,
            output_file=output_csv,
        )

        # =====================================================================
        # Step 4: Display Results
        # =====================================================================
        hyperparams = results["hyperparams"]
        perf = results["performance"]
        leaf = results["leaf_alignment"]

        console.print(f"\n[bold magenta]{'═' * 60}[/bold magenta]")
        console.print(f"[bold magenta]{'Model Performance Evaluation':^60}[/bold magenta]")
        console.print(f"[bold magenta]{'═' * 60}[/bold magenta]")

        display_test_evaluation("Real → Test", hyperparams["best_cv_score"], perf["real"])
        display_test_evaluation("Synthetic → Test", hyperparams["best_cv_score"], perf["synthetic"])

        # Performance gap
        console.print("\n[bold]Performance Gap (Real - Synthetic)[/bold]")
        gap_color = lambda g: "green" if g >= 0 else "red"
        console.print(f"    AUROC Gap:     [{gap_color(perf['auroc_gap'])}]{perf['auroc_gap']:+.4f}[/{gap_color(perf['auroc_gap'])}]")
        console.print(f"    F1 Gap:        [{gap_color(perf['f1_gap'])}]{perf['f1_gap']:+.4f}[/{gap_color(perf['f1_gap'])}]")
        console.print(f"    Precision Gap: [{gap_color(perf['precision_gap'])}]{perf['precision_gap']:+.4f}[/{gap_color(perf['precision_gap'])}]")
        console.print(f"    Recall Gap:    [{gap_color(perf['recall_gap'])}]{perf['recall_gap']:+.4f}[/{gap_color(perf['recall_gap'])}]")

        # =====================================================================
        # Step 5: Save Summary JSON
        # =====================================================================
        console.print("\n[bold]Step 3: Saving Results[/bold]")

        elapsed_time = time.time() - start_time

        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dseed_dir": str(dseed_dir),
                "synthetic_file": str(synthetic_file),
                "target_column": target_column,
                "seed": seed,
                "execution_time_seconds": round(elapsed_time, 2),
            },
            "hyperparams": {
                "source": str(hyperparams_path),
                "best_cv_auroc": hyperparams["best_cv_score"],
                "optimal_threshold": hyperparams["optimal_threshold"],
                "lgbm_params": hyperparams["lgbm_params"],
            },
            "performance": {
                "real_auroc": perf["real"]["auroc"],
                "real_f1": perf["real"]["f1"],
                "real_precision": perf["real"]["precision"],
                "real_recall": perf["real"]["recall"],
                "synthetic_auroc": perf["synthetic"]["auroc"],
                "synthetic_f1": perf["synthetic"]["f1"],
                "synthetic_precision": perf["synthetic"]["precision"],
                "synthetic_recall": perf["synthetic"]["recall"],
                "auroc_gap": perf["auroc_gap"],
                "f1_gap": perf["f1_gap"],
            },
            "leaf_alignment": {
                "n_estimators": n_estimators,
                **leaf,
            },
            "data_shapes": results["data_shapes"],
        }

        output_json = output_csv.with_suffix(".json")
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)

        console.print(f"  [green]✓[/green] Saved scores: {output_csv}")
        console.print(f"  [green]✓[/green] Saved summary: {output_json}")

        # =====================================================================
        # Final Summary
        # =====================================================================
        console.print("\n" + "=" * 60)
        console.print("[bold]Results Summary[/bold]")
        console.print("=" * 60)
        console.print(f"  AUROC Gap (Real - Synth):   {perf['auroc_gap']:+.4f}")
        console.print(f"  Total synthetic points:     {leaf['n_total']:,}")
        console.print(f"  Reliably beneficial:        {leaf['n_beneficial']:,} ({leaf['pct_beneficial']:.2f}%)")
        console.print(f"  Reliably harmful:           {leaf['n_hallucinated']:,} ({leaf['pct_hallucinated']:.2f}%)")
        console.print(f"  Uncertain:                  {leaf['n_uncertain']:,} ({leaf['pct_uncertain']:.2f}%)")
        console.print(f"  Mean utility:               {leaf['mean_utility']:.6f}")
        console.print(f"\n  Execution time:             {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Leaf alignment scoring completed successfully![/bold green]\n")

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Error during leaf alignment scoring:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


# Placeholder for future commands
# @app.command(name="alpha-precision")
# def alpha_precision_scoring(...):
#     """Compute AlphaPrecision per-point authenticity scores."""
#     pass

# @app.command(name="prdc-density")
# def prdc_density_scoring(...):
#     """Compute PRDC density per-point scores."""
#     pass

# @app.command(name="combined")
# def combined_scoring(...):
#     """Compute combined multi-metric scores with risk categories."""
#     pass
