#!/usr/bin/env python3
"""
Visualize utility score distributions from leaf alignment analysis.

Creates interactive histogram plots showing the distribution of per-point utility
scores from Level 1/2/3 evaluation results.

Usage:
    uv run python visualize_utility.py <csv_path> [--output <output_path>] [--title <title>]

Example:
    uv run python visualize_utility.py ../rd-lake/dseed5/synth_eval_level1_dropin.csv --output level1_dist.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

console = Console()


def plot_utility_distribution(
    csv_path: str,
    output_path: str = None,
    title: str = "Utility Score Distribution",
) -> None:
    """
    Create stacked histograms showing utility distribution by classification and class.

    Args:
        csv_path: Path to leaf alignment CSV output
        output_path: Where to save figure (HTML). If not provided, opens in browser.
        title: Plot title
    """
    # Load data
    console.print(f"[bold]Loading data from:[/bold] {csv_path}")
    df = pd.read_csv(csv_path)

    # Extract utility scores
    utility_scores = df["utility_score"].values

    # Calculate statistics
    n_total = len(utility_scores)
    mean_utility = np.mean(utility_scores)
    median_utility = np.median(utility_scores)
    std_utility = np.std(utility_scores)

    # Classification based on CI bounds
    reliably_harmful = df["utility_ci_upper"] < 0
    reliably_beneficial = df["utility_ci_lower"] > 0
    uncertain = ~reliably_harmful & ~reliably_beneficial

    n_harmful = reliably_harmful.sum()
    n_beneficial = reliably_beneficial.sum()
    n_uncertain = uncertain.sum()

    pct_harmful = 100 * n_harmful / n_total
    pct_beneficial = 100 * n_beneficial / n_total
    pct_uncertain = 100 * n_uncertain / n_total

    # Class breakdown
    class_col = df["class"]
    if class_col.dtype == bool:
        majority_mask = ~class_col
        minority_mask = class_col
    else:
        majority_mask = class_col == 0
        minority_mask = class_col == 1

    n_majority = majority_mask.sum()
    n_minority = minority_mask.sum()

    # Print summary to console
    console.print(f"\n[bold cyan]Summary Statistics:[/bold cyan]")
    console.print(f"  Total points: {n_total:,}")
    console.print(f"  Mean utility: {mean_utility:.2e}")
    console.print(f"  Median utility: {median_utility:.2e}")
    console.print(f"  Std dev: {std_utility:.2e}")
    console.print(f"\n[bold cyan]Classification (95% CI):[/bold cyan]")
    console.print(f"  [red]Harmful[/red]: {n_harmful:,} ({pct_harmful:.1f}%)")
    console.print(f"  [green]Beneficial[/green]: {n_beneficial:,} ({pct_beneficial:.1f}%)")
    console.print(f"  [yellow]Uncertain[/yellow]: {n_uncertain:,} ({pct_uncertain:.1f}%)")
    console.print(f"\n[bold cyan]By Class:[/bold cyan]")
    console.print(f"  Majority (label=0): {n_majority:,}")
    console.print(f"  Minority (label=1): {n_minority:,}")

    # Compute shared x-axis range
    x_min, x_max = utility_scores.min(), utility_scores.max()
    x_padding = (x_max - x_min) * 0.05
    x_range = [x_min - x_padding, x_max + x_padding]

    # Helper function to compute KDE
    def compute_kde(data, n_points=500):
        """Compute KDE using scipy."""
        from scipy import stats
        if len(data) < 2:
            return np.array([]), np.array([])
        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(x_min - x_padding, x_max + x_padding, n_points)
        y_vals = kde(x_vals)
        return x_vals, y_vals

    # Create 2-row subplot: classification on top, class on bottom
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "By Classification (95% CI)",
            "By Class",
        ],
    )

    # Row 1: Classification KDEs (Harmful, Uncertain, Beneficial)
    # Harmful
    x_kde, y_kde = compute_kde(df.loc[reliably_harmful, "utility_score"].values)
    if len(x_kde) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"Harmful (N={n_harmful:,}, {pct_harmful:.1f}%)",
                line=dict(color="red", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.15)",
                legendgroup="classification",
            ),
            row=1, col=1,
        )

    # Uncertain
    x_kde, y_kde = compute_kde(df.loc[uncertain, "utility_score"].values)
    if len(x_kde) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"Uncertain (N={n_uncertain:,}, {pct_uncertain:.1f}%)",
                line=dict(color="orange", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.15)",
                legendgroup="classification",
            ),
            row=1, col=1,
        )

    # Beneficial
    x_kde, y_kde = compute_kde(df.loc[reliably_beneficial, "utility_score"].values)
    if len(x_kde) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"Beneficial (N={n_beneficial:,}, {pct_beneficial:.1f}%)",
                line=dict(color="green", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 128, 0, 0.15)",
                legendgroup="classification",
            ),
            row=1, col=1,
        )

    # Row 2: Class KDEs (Majority, Minority)
    # Majority
    x_kde, y_kde = compute_kde(df.loc[majority_mask, "utility_score"].values)
    if len(x_kde) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"Majority - label=0 (N={n_majority:,})",
                line=dict(color="steelblue", width=2),
                fill="tozeroy",
                fillcolor="rgba(70, 130, 180, 0.15)",
                legendgroup="class",
            ),
            row=2, col=1,
        )

    # Minority
    x_kde, y_kde = compute_kde(df.loc[minority_mask, "utility_score"].values)
    if len(x_kde) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"Minority - label=1 (N={n_minority:,})",
                line=dict(color="coral", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 127, 80, 0.15)",
                legendgroup="class",
            ),
            row=2, col=1,
        )

    # Add vertical line at x=0 for both subplots
    for row in range(1, 3):
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            line_width=1.5,
            row=row, col=1,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        template="plotly_white",
        height=700,
        width=1000,
    )

    # Set shared x-axis range for all subplots
    fig.update_xaxes(range=x_range)
    fig.update_xaxes(title_text="Utility Score", row=2, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        console.print(f"\n[green]Saved interactive figure to:[/green] {output_path}")
    else:
        console.print("\n[yellow]Opening in browser...[/yellow]")
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize utility score distributions from leaf alignment analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Open interactive plot in browser
  uv run python visualize_utility.py ../rd-lake/dseed5/synth_eval_level1_dropin.csv

  # Save to HTML file
  uv run python visualize_utility.py ../rd-lake/dseed5/synth_eval_level1_dropin.csv --output level1_dist.html

  # Custom title
  uv run python visualize_utility.py data.csv --output fig.html --title "Level 1: Drop-in Test"
        """
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to leaf alignment CSV output (from eval command)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for figure (HTML). If not provided, opens in browser."
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="Utility Score Distribution",
        help="Plot title"
    )

    args = parser.parse_args()

    plot_utility_distribution(
        csv_path=args.csv_path,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
