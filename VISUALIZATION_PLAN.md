# Visualization Plan for Harmful Synthetic Data Points Paper

## Single Dataset Visualizations (Essential)

### Figure 1: Three-Level Utility Distribution Comparison ⭐⭐⭐ (CRITICAL)

**Type:** Three-panel horizontal layout with density plots

**Panel A: Level 1 (Drop-in)**
- X-axis: Utility score (log scale: -0.00001 to +0.00001)
- Y-axis: Density
- Histogram with KDE overlay
- Color: Gray (artifact, don't emphasize)
- Annotations:
  - "94.2% beneficial" with arrow
  - "Artifact: utilities ≈ 0"
  - Mean: 0.000001 displayed

**Panel B: Level 2 (Adjusted)**
- Same as Panel A
- Color: Gray (identical to L1)
- Annotation: "Identical to L1 → imbalance not the cause"

**Panel C: Level 3 (Retuned)**
- Color: Two-tone (harmful=red, beneficial=green, uncertain=yellow)
- Annotations:
  - "8.5% harmful"
  - "74.9% beneficial"
  - "16.6% uncertain"
  - "Wider distribution = realistic uncertainty"

**Impact:** Visually demonstrates artifact vs realistic results

**Code sketch:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (level, data) in enumerate(levels.items()):
    sns.histplot(data['utility_score'], kde=True, ax=axes[i])
    axes[i].axvline(0, color='black', linestyle='--')
    axes[i].set_title(f'Level {i+1}: {level}')
```

---

### Figure 2: Class-Specific Harmful Rates (Level 3) ⭐⭐⭐ (CRITICAL)

**Type:** Grouped bar chart with error bars (95% CIs)

**X-axis categories:**
- Majority class (No readmission)
- Minority class (Readmission)

**Y-axis:** Percentage (0-100%)

**Bars per category:**
1. Beneficial (green)
2. Harmful (red)
3. Uncertain (yellow)

**Key annotations:**
- Arrow showing 4.0× differential between harmful rates
- N = 9,327 (majority), N = 673 (minority)
- P-value from chi-square test of independence

**Impact:** Starkly shows minority class failure

**Code sketch:**
```python
data = {
    'Majority': {'Beneficial': 77.65, 'Harmful': 7.07, 'Uncertain': 15.29},
    'Minority': {'Beneficial': 36.70, 'Harmful': 28.38, 'Uncertain': 34.92}
}
df = pd.DataFrame(data).T
df.plot(kind='bar', stacked=False, color=['green', 'red', 'yellow'])
plt.ylabel('Percentage (%)')
plt.axhline(y=28.38/7.07, linestyle='--', label='4.0× differential')
```

---

### Figure 3: Hyperparameter Drift Radar Plot ⭐⭐

**Type:** Radar/spider chart

**Axes (normalized 0-1):**
- num_leaves (22 → 16)
- reg_lambda (2.95 → 9.98)
- subsample (0.64 → 0.95)
- learning_rate (0.0251 → 0.0287)
- CV score (0.6439 → 0.5746)

**Two overlaid polygons:**
- Real data (blue, solid)
- Synthetic data (red, dashed)

**Impact:** Visually shows "shape change" of optimal hyperparameters

---

### Figure 4: Threshold Shift Visualization ⭐⭐

**Type:** Probability distribution with threshold markers

**X-axis:** Predicted probability (0 to 1)
**Y-axis:** Density

**Elements:**
- Distribution of predicted probabilities from real-trained model (blue)
- Distribution from synthetic-trained model (red)
- Vertical line at threshold=0.390 (real, solid)
- Vertical line at threshold=0.050 (synthetic, dashed)
- Shaded region showing "predicted positive" for each

**Annotation:** "-87% threshold shift = severe miscalibration"

**Impact:** Shows how synthetic model "doesn't trust" its predictions

---

### Figure 5: Utility Score Scatter with Confidence Intervals ⭐⭐

**Type:** Scatter plot with error bars (sample of 1000 points)

**X-axis:** Point index (sample of 1000)
**Y-axis:** Utility score with 95% CI

**Points colored by classification:**
- Green: Beneficial (CI lower > 0)
- Red: Harmful (CI upper < 0)
- Yellow: Uncertain (CI spans 0)

**Horizontal line at y=0**

**Impact:** Shows uncertainty quantification and CI-based classification

---

### Figure 6: Performance Gap vs Tree Count ⭐

**Type:** Line plot showing convergence

**X-axis:** Number of trees (1000, 2000, ..., 10000)
**Y-axis:** AUROC gap (Real - Synthetic)

**Three lines:**
- Level 1 (stays ~8.3%, flat)
- Level 2 (stays ~8.3%, flat)
- Level 3 (improves slightly as trees increase, plateaus at 7.4%)

**Vertical line at 7626:** "Early stopping point (Level 3)"

**Impact:** Shows why L3 stopped early (no improvement)

---

## Multi-Generation Visualizations (For 20 Generations)

### Figure 7: Harmful Rate Evolution Over Generations ⭐⭐⭐ (CRITICAL)

**Type:** Time series line plot with confidence bands

**X-axis:** Generation number (1-20)
**Y-axis:** Harmful point percentage (0-100%)

**Three lines:**
1. Overall harmful rate (solid black, thick)
2. Minority class harmful rate (red, dashed)
3. Majority class harmful rate (blue, dotted)

**95% confidence band around each line**

**Potential insights:**
- Does quality improve over generations?
- Does gap between classes narrow or widen?
- Is there convergence?

**Code sketch:**
```python
generations = range(1, 21)
harmful_overall = [results[g]['harmful_pct'] for g in generations]
harmful_minority = [results[g]['minority_harmful_pct'] for g in generations]
harmful_majority = [results[g]['majority_harmful_pct'] for g in generations]

plt.plot(generations, harmful_overall, 'k-', linewidth=2, label='Overall')
plt.plot(generations, harmful_minority, 'r--', label='Minority class')
plt.plot(generations, harmful_majority, 'b:', label='Majority class')
plt.fill_between(generations, lower_ci, upper_ci, alpha=0.2)
```

---

### Figure 8: Quality Stability Across Generations (Heatmap) ⭐⭐⭐

**Type:** Heatmap

**Rows:** Individual synthetic points (e.g., sample 100 points consistently across generations)
**Columns:** Generations (1-20)
**Cell color:**
- Green: Beneficial in that generation
- Red: Harmful in that generation
- Yellow: Uncertain in that generation
- White: Point doesn't exist in that generation

**Clustering:** Hierarchical clustering on rows to group similar temporal patterns

**Potential insights:**
- Are some points consistently harmful across generations?
- Is there random variation or systematic patterns?
- Do certain "problem points" persist?

---

### Figure 9: Performance Metrics Trajectory ⭐⭐

**Type:** Multi-line time series

**X-axis:** Generation (1-20)
**Y-axis:** Metric value

**Lines:**
1. AUROC gap (Real - Synthetic)
2. Threshold shift magnitude (|Real - Synthetic|)
3. Harmful point percentage
4. Early stopping point (trees built / trees requested)

**All normalized to 0-1 scale for comparison**

**Impact:** Shows correlation between different quality signals

---

### Figure 10: Class Differential Evolution ⭐⭐⭐

**Type:** Stacked area chart

**X-axis:** Generation (1-20)
**Y-axis:** Percentage (0-100%)

**Areas (stacked):**
- Bottom: Majority class harmful rate (blue)
- Top: Minority class harmful rate (red)
- Gap between = class differential

**Secondary Y-axis:** Differential ratio (minority/majority)

**Potential insights:**
- Does generator learn to balance classes over time?
- Is minority class consistently worse?

---

### Figure 11: Convergence Analysis (Box Plots) ⭐

**Type:** Box plot grid

**Rows:** Metrics (harmful %, AUROC gap, threshold shift, CV score gap)
**Columns:** Generation bins (1-5, 6-10, 11-15, 16-20)

**Each box shows distribution across generations in that bin**

**Impact:** Shows variance reduction over time (convergence) or persistent instability

---

### Figure 12: Point-Level Consistency Matrix ⭐⭐

**Type:** Confusion matrix style

**Axes:** Classification in Generation 1 vs Classification in Generation 20

**Cells:**
- Beneficial → Beneficial: Stable quality (green)
- Beneficial → Harmful: Quality degraded (yellow)
- Harmful → Beneficial: Quality improved (light green)
- Harmful → Harmful: Consistently bad (red)

**Percentage in each cell**

**Impact:** Quantifies generator stability

---

## Supplementary Visualizations

### Supplementary Figure S1: Utility Distribution by Confidence Interval Width ⭐

**Type:** Scatter plot

**X-axis:** CI width (upper - lower)
**Y-axis:** Utility score
**Color:** Classification (beneficial/harmful/uncertain)

**Impact:** Shows relationship between uncertainty and classification

---

### Supplementary Figure S2: Calibration Curves ⭐⭐

**Type:** Traditional calibration plot

**X-axis:** Predicted probability (binned)
**Y-axis:** Observed frequency

**Two curves:**
- Real-trained model (blue, well-calibrated)
- Synthetic-trained model (red, miscalibrated)

**Diagonal line:** Perfect calibration

**Impact:** Formal calibration analysis supporting threshold shift findings

---

### Supplementary Figure S3: Tree-Level Utility Variance ⭐

**Type:** Box plot per point

**X-axis:** Top 20 harmful points (by mean utility)
**Y-axis:** Utility score

**Each box:** Distribution of utilities across 7,626 trees for that point

**Impact:** Shows consistency of harmful classification across trees

---

### Supplementary Figure S4: Pairwise Generation Correlation ⭐ (Multi-gen only)

**Type:** Correlation matrix heatmap

**Axes:** Generations (1-20) × Generations (1-20)
**Cell value:** Spearman correlation of utility scores between generation pairs

**Impact:**
- High correlation = stable harmful point identification
- Low correlation = random variation (bad sign for generator)

---

## Interactive Visualizations (For Online Supplement)

### Interactive 1: Per-Point Explorer

**Tool:** Plotly/Dash

**Features:**
- Scatter plot of all 10,000 points (x=index, y=utility, color=class)
- Click point → show details:
  - Utility score, CI, classification
  - Original features (if privacy allows)
  - Trajectory across generations (if multi-gen)
- Filter by: Class, classification, generation

---

### Interactive 2: Generation Comparison Slider

**Tool:** Plotly

**Features:**
- Slider to select generation (1-20)
- Updates all plots in real-time:
  - Utility distribution
  - Class-specific harmful rates
  - Hyperparameter radar
  - Performance metrics

---

## Recommended Figure Priorities for Main Paper

**Must-have (main text):**
1. ⭐⭐⭐ Figure 1: Three-level utility distributions (shows artifact)
2. ⭐⭐⭐ Figure 2: Class-specific harmful rates (shows 4× differential)
3. ⭐⭐⭐ Figure 7: Harmful rate evolution (if multi-gen) OR Figure 3: Hyperparameter drift

**Nice-to-have (main text):**
4. Figure 4: Threshold shift visualization
5. Figure 8: Quality stability heatmap (if multi-gen)

**Supplementary:**
- All others

---

## Multi-Generation Specific Analyses

### Analysis 1: Temporal Autocorrelation

**Question:** Are consecutive generations more similar than distant ones?

**Method:** Compute correlation(generation_i, generation_i+k) for k=1,2,...,19

**Visualization:** Line plot of correlation vs lag k

**Interpretation:**
- High autocorrelation → generator is stable (good)
- Low autocorrelation → random variation (bad)

---

### Analysis 2: Harmful Point Overlap Across Generations

**Question:** Are the same points consistently harmful?

**Method:**
- Identify harmful points in each generation
- Compute Jaccard similarity: |Harmful_i ∩ Harmful_j| / |Harmful_i ∪ Harmful_j|

**Visualization:** Heatmap of pairwise Jaccard similarities

**Interpretation:**
- High overlap → systematic failure (generator can't fix certain patterns)
- Low overlap → random noise (different story)

---

### Analysis 3: Learning Curve

**Question:** Does synthetic data quality improve with more generations?

**Method:**
- Fit regression: Harmful_rate ~ Generation
- Test for significant trend (p-value)

**Visualization:** Scatter with regression line + confidence band

**Interpretation:**
- Negative slope → improvement over time
- Flat → no learning
- Positive slope → degradation (bad!)

---

### Analysis 4: Convergence Testing

**Question:** Has quality converged by generation 20?

**Method:**
- Compare variance in early gens (1-10) vs late gens (11-20)
- Levene's test for equality of variances

**Visualization:** Box plot of harmful rates, early vs late

**Interpretation:**
- Lower variance in late gens → convergence
- Similar variance → still unstable

---

## Code Template for Multi-Generation Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load all generation results
generations = {}
for gen in range(1, 21):
    csv_path = f'results/gen_{gen:02d}_evaluation.csv'
    summary_path = f'results/gen_{gen:02d}_summary.json'

    generations[gen] = {
        'data': pd.read_csv(csv_path),
        'summary': json.load(open(summary_path))
    }

# Extract metrics
metrics = pd.DataFrame({
    'generation': range(1, 21),
    'harmful_overall': [g['summary']['leaf_alignment']['pct_hallucinated'] for g in generations.values()],
    'harmful_minority': [...],  # Need to compute from CSV
    'harmful_majority': [...],
    'auroc_gap': [...],
    'threshold_real': [...],
    'threshold_synth': [...],
    'trees_built': [...],
})

# Figure 7: Evolution over generations
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(metrics['generation'], metrics['harmful_overall'], 'k-', linewidth=2, label='Overall')
ax.plot(metrics['generation'], metrics['harmful_minority'], 'r--', label='Minority')
ax.plot(metrics['generation'], metrics['harmful_majority'], 'b:', label='Majority')
ax.set_xlabel('Generation')
ax.set_ylabel('Harmful Point Rate (%)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig7_evolution.pdf', dpi=300)

# Figure 8: Stability heatmap
# Sample 100 points that exist across all generations
sample_points = np.random.choice(10000, 100, replace=False)

stability_matrix = np.zeros((100, 20))
for i, pt_idx in enumerate(sample_points):
    for gen in range(1, 21):
        df = generations[gen]['data']
        if pt_idx < len(df):
            if df.iloc[pt_idx]['reliably_hallucinated']:
                stability_matrix[i, gen-1] = -1  # Red
            elif df.iloc[pt_idx]['reliably_beneficial']:
                stability_matrix[i, gen-1] = 1   # Green
            else:
                stability_matrix[i, gen-1] = 0   # Yellow

sns.heatmap(stability_matrix, cmap='RdYlGn', center=0,
            xticklabels=range(1,21), yticklabels=False,
            cbar_kws={'label': 'Classification'})
plt.xlabel('Generation')
plt.ylabel('Synthetic Points (sample of 100)')
plt.title('Point-Level Quality Stability Across Generations')
plt.savefig('figures/fig8_stability_heatmap.pdf', dpi=300)

# Analysis: Point overlap
from sklearn.metrics import jaccard_score

overlap_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        harmful_i = generations[i+1]['data']['reliably_hallucinated'].values
        harmful_j = generations[j+1]['data']['reliably_hallucinated'].values

        # Jaccard similarity
        overlap_matrix[i, j] = np.sum(harmful_i & harmful_j) / np.sum(harmful_i | harmful_j)

sns.heatmap(overlap_matrix, annot=False, cmap='viridis',
            xticklabels=range(1,21), yticklabels=range(1,21))
plt.xlabel('Generation')
plt.ylabel('Generation')
plt.title('Harmful Point Overlap (Jaccard Similarity)')
plt.savefig('figures/supplementary_overlap.pdf', dpi=300)

# Statistical test: Is there improvement over time?
from scipy.stats import spearmanr

corr, pval = spearmanr(metrics['generation'], metrics['harmful_overall'])
print(f"Spearman correlation: {corr:.3f} (p={pval:.4f})")

if pval < 0.05:
    if corr < 0:
        print("✓ Significant improvement over generations")
    else:
        print("⚠ Significant degradation over generations")
else:
    print("No significant trend detected")
```

---

## Publication-Ready Formatting Tips

**For all figures:**
- DPI: 300+ for publication
- Format: PDF (vector) for plots, PNG for heatmaps
- Font size: 10-12pt for labels, 8-10pt for tick labels
- Line width: 1.5-2pt for main lines
- Color palette: Colorblind-friendly (use seaborn 'colorblind' palette)
- Legends: Outside plot area or semi-transparent overlay
- Annotations: 8pt font, minimal, high contrast

**Figure dimensions:**
- Single column: 3.5 inches wide
- Double column: 7 inches wide
- Height: Maintain aspect ratio, typically 2.5-4 inches

**Captions:**
- Start with one-sentence summary
- Define all abbreviations
- Describe what each panel shows
- Include sample sizes (N=...)
- Note statistical tests if applicable

**Example caption:**
> **Figure 7. Harmful point detection rates decrease over generator training.** Evolution of harmful point percentages across 20 training generations for overall dataset (black, solid), minority class readmissions (red, dashed), and majority class no-readmissions (blue, dotted). Shaded bands indicate 95% confidence intervals from bootstrap resampling (1000 iterations). The minority class consistently exhibits 3-5× higher harmful rates than majority class across all generations (Spearman ρ = -0.78, p < 0.001 for downward trend). N = 10,000 synthetic points per generation evaluated against 10,000 real test samples.

---

## Data to Collect from 20 Generations

**Per generation, save:**
1. Full CSV: `gen_{i:02d}_evaluation.csv` (per-point utilities)
2. Summary JSON: `gen_{i:02d}_summary.json` (aggregate statistics)
3. Hyperparameters: `gen_{i:02d}_hyperparams.json` (from Level 3 tuning)

**Aggregate analysis file:**
```json
{
  "generations": {
    "1": {
      "harmful_overall": 8.5,
      "harmful_minority": 28.4,
      "harmful_majority": 7.1,
      "auroc_gap": 7.4,
      "threshold_real": 0.390,
      "threshold_synth": 0.050,
      "trees_built": 7626,
      "cv_score_real": 0.6439,
      "cv_score_synth": 0.5746
    },
    "2": {...},
    ...
  },
  "metadata": {
    "dataset": "MIMIC-III",
    "generator": "ARF",
    "n_synthetic": 10000,
    "n_test": 10000,
    "evaluation_date": "2025-12-30"
  }
}
```

This allows easy loading for cross-generation analysis.

---

## Questions to Answer with 20 Generations

1. **Does quality improve?** (Trend analysis)
2. **Does quality stabilize?** (Variance analysis)
3. **Are failures consistent?** (Overlap analysis)
4. **Does class gap close?** (Differential evolution)
5. **Is improvement meaningful?** (Effect size: generation 20 vs generation 1)
6. **Can we predict quality?** (Regression: generation → harmful rate)

These analyses would strengthen the paper significantly!
