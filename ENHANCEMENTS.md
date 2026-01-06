# Potential Enhancements

This document tracks proposed enhancements and future improvements to the sdvaluation package.

---

## 1. Automated Baseline Comparison

**Status:** Proposed
**Priority:** Medium
**Effort:** ~2-4 hours
**Proposed Date:** 2026-01-05

### Problem

Currently, users must manually run leaf alignment twice to compare synthetic data quality against real training data:

```bash
# Run 1: Baseline
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/real_train.csv \
  --real-test data/real_test.csv \
  --output baseline_scores.csv

# Run 2: Evaluation
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/gen2_train.csv \
  --real-test data/real_test.csv \
  --output gen2_scores.csv

# Manual comparison required
```

This requires:
- Running the script twice
- Manual CSV comparison
- Manual calculation of ratio metrics

### Proposed Solution

Add automatic baseline comparison when real training data is provided.

#### API Design

**Python API:**

```python
from sdvaluation.leaf_alignment import run_leaf_alignment_with_baseline

results = run_leaf_alignment_with_baseline(
    X_synthetic=synthetic_train_df,
    y_synthetic=synthetic_train_labels,
    X_real_train=real_train_df,        # NEW: Real training for baseline
    y_real_train=real_train_labels,    # NEW
    X_real_test=real_test_df,
    y_real_test=real_test_labels,
    lgbm_params=params,
    n_estimators=500,
    output_file=Path("comparison_results.csv")
)

# Returns:
{
    'synthetic': {
        'n_hallucinated': 9339,
        'pct_hallucinated': 93.39,
        'n_beneficial': 54,
        'pct_beneficial': 0.54,
        ...
    },
    'baseline': {
        'n_hallucinated': 25,
        'pct_hallucinated': 0.25,
        'n_beneficial': 8969,
        'pct_beneficial': 89.69,
        ...
    },
    'comparison': {
        'harmful_ratio': 373.56,        # Synthetic/Baseline
        'beneficial_ratio': 0.006,      # Synthetic/Baseline
        'quality_score': 0.0027,        # Overall quality metric
    }
}
```

**CLI Design:**

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/gen2_train.csv \
  --real-train data/real_train.csv \      # NEW: Optional baseline
  --real-test data/real_test.csv \
  --encoding-config config/encoding.yaml \
  --lgbm-params config/lgbm_params.json \
  --compare-baseline \                    # NEW: Enable comparison
  --output comparison_results.csv
```

#### Enhanced Output

```
═══════════════════════════════════════════════════
         Leaf Alignment: Baseline Comparison
═══════════════════════════════════════════════════

Dataset Statistics:
  Real training:   10,000 points
  Synthetic:       10,000 points
  Real test:        8,000 points

Classification Results:

                        Real Train    Synthetic    Ratio
  ─────────────────────────────────────────────────────────
  Reliably Harmful:     25 (0.25%)   9,339 (93%)  373.6×
  Reliably Beneficial:  8,969 (90%)  54 (0.5%)    0.006×
  Uncertain:            1,006 (10%)  607 (6%)     0.6×

Quality Metrics:
  Harmful Ratio:     373.6× more hallucinated
  Beneficial Ratio:  0.006× less beneficial (166× worse)

Conclusion: Synthetic data quality is SIGNIFICANTLY DEGRADED
  → 373× more hallucinated than real training data
  → Only 0.5% of points are beneficial vs 90% in real data
  → RECOMMENDATION: Reject this synthetic dataset
```

#### Implementation Notes

1. **Reuse existing code:** Wrap `run_leaf_alignment()` to run twice
2. **Minimal changes:** No modification to core algorithm
3. **Backward compatible:** Original function unchanged
4. **Output format:** Add comparison section to existing output
5. **CSV output:** Include comparison metrics in summary section

**File structure:**

```python
# sdvaluation/leaf_alignment.py

def run_leaf_alignment(...):
    # Existing implementation (unchanged)
    pass

def run_leaf_alignment_with_baseline(
    X_synthetic, y_synthetic,
    X_real_train, y_real_train,  # NEW parameters
    X_real_test, y_real_test,
    lgbm_params,
    output_file=None,
    n_estimators=500,
    empty_leaf_penalty=-1.0,
    n_jobs=1,
    random_state=42,
):
    """
    Run leaf alignment with automatic baseline comparison.

    Trains two models:
      1. On real training data (baseline)
      2. On synthetic data (evaluation)

    Both are evaluated on the same real test data.
    """
    console.print("[bold]Running baseline analysis on real training data...[/bold]")
    baseline_results = run_leaf_alignment(
        X_real_train, y_real_train,
        X_real_test, y_real_test,
        lgbm_params, None, n_estimators,
        empty_leaf_penalty, n_jobs, random_state
    )

    console.print("\n[bold]Running evaluation on synthetic data...[/bold]")
    synthetic_results = run_leaf_alignment(
        X_synthetic, y_synthetic,
        X_real_test, y_real_test,
        lgbm_params, output_file, n_estimators,
        empty_leaf_penalty, n_jobs, random_state
    )

    # Compute comparison metrics
    comparison = {
        'harmful_ratio': synthetic_results['pct_hallucinated'] / baseline_results['pct_hallucinated'],
        'beneficial_ratio': synthetic_results['pct_beneficial'] / baseline_results['pct_beneficial'],
    }

    # Display comparison
    _display_comparison(baseline_results, synthetic_results, comparison)

    return {
        'baseline': baseline_results,
        'synthetic': synthetic_results,
        'comparison': comparison,
    }
```

#### Testing

- Test with Real vs Real (ratio should be ~1.0)
- Test with Real vs Gen2 (should show 373× ratio)
- Test with small datasets
- Test with imbalanced classes

---

## 2. Marginal Point Classification

**Status:** Proposed
**Priority:** Medium
**Effort:** ~4-6 hours
**Proposed Date:** 2026-01-05

### Problem

Current three-way classification (harmful/uncertain/beneficial) treats all "reliably beneficial" points equally:

```python
Point A: mean = +0.0612, CI = [+0.0605, +0.0619]  → Beneficial ✓
Point B: mean = +0.0008, CI = [+0.0003, +0.0013]  → Beneficial ✓
```

But Point A contributes 76× more utility than Point B. Should we treat them the same?

**Issue:** Points can be **statistically significant** but **practically negligible**.

### Proposed Solution

Add five-tier classification system to distinguish strong vs marginal contributions.

#### Classification Tiers

```python
# Tier 1: Strongly Harmful
#   - Clear negative impact, large effect
#   - CI_upper < -threshold (e.g., -0.01)

# Tier 2: Marginally Harmful
#   - Statistically harmful but small effect
#   - -threshold <= CI_upper < 0

# Tier 3: Uncertain
#   - CI spans 0
#   - Insufficient evidence

# Tier 4: Marginally Beneficial
#   - Statistically beneficial but small effect
#   - CI_lower > 0 AND mean < threshold

# Tier 5: Strongly Beneficial
#   - Clear positive impact, large effect
#   - mean >= threshold
```

#### API Design

```python
from sdvaluation.leaf_alignment import classify_points_detailed

results_classified = classify_points_detailed(
    results_df,
    marginal_threshold=0.01,     # Absolute utility threshold
    use_percentile=False,        # Or use data-driven threshold
    percentile_cutoff=0.25,      # If use_percentile=True
)

# Adds 'category' column:
# - 'strongly_harmful'
# - 'marginally_harmful'
# - 'uncertain'
# - 'marginally_beneficial'
# - 'strongly_beneficial'
```

#### Threshold Selection Methods

**Method 1: Absolute Threshold (Recommended)**

```python
# Based on practical significance
# 0.01 = 2% of maximum possible utility (0.5)
marginal_threshold = 0.01
```

**Method 2: Percentile-Based**

```python
# Keep top 75% of beneficial points
reliable = results[results['utility_ci_lower'] > 0]
marginal_threshold = reliable['utility_score'].quantile(0.25)
```

**Method 3: Effect Size**

```python
# Points must be >10 standard errors from zero
effect_size_threshold = 10
strong_beneficial = (results['utility_ci_lower'] > 0) & \
                   (results['utility_score'] / results['utility_se'] > effect_size_threshold)
```

**Method 4: Visual Inspection**

```python
# Plot histogram and look for natural gaps
import matplotlib.pyplot as plt
plt.hist(results[results['utility_ci_lower'] > 0]['utility_score'], bins=50)
plt.show()
# Set threshold at visible gap
```

#### Enhanced Output

```
Statistical Confidence (5-tier classification):
  Strongly Harmful:         8,138 (81.38%)  ← CI_upper < -0.01
  Marginally Harmful:       1,201 (12.01%)  ← -0.01 ≤ CI_upper < 0
  Uncertain:                  607 ( 6.07%)  ← CI spans 0
  Marginally Beneficial:       42 ( 0.42%)  ← CI_lower > 0, mean < 0.01
  Strongly Beneficial:         12 ( 0.12%)  ← mean ≥ 0.01

Threshold used: 0.01 (2% of max utility)

Quality Assessment:
  Strong signal points: 8,150 (81.5%)
  Marginal signal points: 1,243 (12.4%)
  Uncertain points: 607 (6.1%)

Recommendation: Filter out strongly + marginally harmful (9,339 points)
                Keep only strongly beneficial if quality is critical (12 points)
```

#### Implementation

```python
def classify_points_detailed(
    results: pd.DataFrame,
    marginal_threshold: float = 0.01,
    use_percentile: bool = False,
    percentile_cutoff: float = 0.25,
) -> pd.DataFrame:
    """
    Classify points into 5 tiers with marginal distinction.

    Args:
        results: DataFrame with utility_score, utility_ci_lower, utility_ci_upper
        marginal_threshold: Absolute threshold for marginal vs strong
        use_percentile: If True, compute threshold from data
        percentile_cutoff: Percentile for data-driven threshold

    Returns:
        DataFrame with 'category' column added
    """
    results = results.copy()

    # Compute threshold
    if use_percentile:
        reliable = results[results['utility_ci_lower'] > 0]
        threshold = reliable['utility_score'].quantile(percentile_cutoff)
    else:
        threshold = marginal_threshold

    # Initialize
    results['category'] = 'uncertain'

    # Classify
    results.loc[results['utility_ci_upper'] < -threshold, 'category'] = 'strongly_harmful'
    results.loc[(results['utility_ci_upper'] < 0) &
                (results['utility_ci_upper'] >= -threshold), 'category'] = 'marginally_harmful'
    results.loc[(results['utility_ci_lower'] <= 0) &
                (results['utility_ci_upper'] >= 0), 'category'] = 'uncertain'
    results.loc[(results['utility_ci_lower'] > 0) &
                (results['utility_score'] < threshold), 'category'] = 'marginally_beneficial'
    results.loc[results['utility_score'] >= threshold, 'category'] = 'strongly_beneficial'

    return results
```

#### Configuration Options

Add to CLI:

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/gen2_train.csv \
  --real-test data/real_test.csv \
  --marginal-threshold 0.01 \        # NEW: Set threshold
  --five-tier-classification \       # NEW: Enable 5-tier output
  --output results_detailed.csv
```

---

## 3. Class-Specific Thresholds

**Status:** Proposed
**Priority:** Low
**Effort:** ~2-3 hours
**Related to:** Enhancement #2

### Problem

Positive and negative classes may have different utility distributions. A single threshold may not be appropriate for both.

**Example:**
- Positive class (minority): Harder to predict, lower typical utilities
- Negative class (majority): Easier to predict, higher typical utilities

Using a single threshold (e.g., 0.01) might:
- Keep too many marginal negative class points
- Filter out useful positive class points

### Proposed Solution

Allow different thresholds per class:

```python
results_classified = classify_points_detailed(
    results_df,
    marginal_threshold={'positive': 0.005, 'negative': 0.015},  # Class-specific
    class_column='class',
)
```

---

## 4. Uncertainty Budget Analysis

**Status:** Proposed
**Priority:** Low
**Effort:** ~3-4 hours

### Problem

Large "uncertain" regions (e.g., 30-40% with 100 trees) make classification difficult. Users don't know if they should:
- Get more trees (expensive)
- Accept uncertainty (risky)
- Use different approach

### Proposed Solution

Add analysis showing how many more trees would be needed to reduce uncertainty to target level.

```python
Uncertainty Analysis:
  Current trees: 100
  Uncertain points: 3,456 (34.6%)

  To reduce uncertain to 10%:
    → Need ~500 trees (5× more)
    → Estimated runtime: ~5 minutes (vs current 2 min)

  To reduce uncertain to 5%:
    → Need ~1,000 trees (10× more)
    → Estimated runtime: ~10 minutes

Recommendation: Use 500 trees for production analysis
```

---

## 5. Export Filtered Datasets

**Status:** Proposed
**Priority:** Low
**Effort:** ~1-2 hours

### Problem

After identifying harmful points, users must manually filter the original synthetic dataset.

### Proposed Solution

Add option to export filtered dataset directly:

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/gen2_train.csv \
  --real-test data/real_test.csv \
  --output-filtered data/gen2_filtered.csv \  # NEW: Export filtered data
  --filter-strategy conservative               # conservative|moderate|liberal
```

Strategies:
- **conservative**: Keep only strongly beneficial
- **moderate**: Remove only reliably harmful
- **liberal**: Remove strongly harmful only

---

## Implementation Priority

**High Priority (Recommended for next release):**
1. Automated Baseline Comparison (most user value)
2. Marginal Point Classification (better quality assessment)

**Medium Priority:**
3. Class-Specific Thresholds
4. Export Filtered Datasets

**Low Priority (Nice to have):**
5. Uncertainty Budget Analysis

---

## Contributing

If you'd like to implement any of these enhancements:

1. Create a feature branch: `git checkout -b feature/baseline-comparison`
2. Implement the enhancement following the API design above
3. Add tests in `tests/test_leaf_alignment.py`
4. Update documentation in `README.md`
5. Submit a pull request

For questions or discussions, open an issue on GitHub.

---

**Last Updated:** 2026-01-05
**Maintainer:** Aleksandar Babic
