# Hyperparameter Analysis Guide

## Purpose

Compare hyperparameters across three datasets (dseed55, dseed2025, dseed6765) to understand why they show dramatically different leaf alignment behaviors despite similar AUROC performance.

## Quick Start

Run the comparison script:

```bash
python compare_hyperparams.py ../rd-lake/dseed55 ../rd-lake/dseed2025 ../rd-lake/dseed6765
```

## What We're Investigating

### The Mystery: Different Datasets, Different Behaviors

| Dataset | Cross-test A Hallucination | Cross-test A AUROC | Pattern |
|---------|---------------------------|-------------------|----------|
| dseed55 | 0.3% (excellent) | 0.6285 | Good alignment → Good performance |
| dseed2025 | 98.7% (terrible) | 0.6414 | Bad alignment → Good performance |
| dseed6765 | 1.5% (good) | 0.6036 | Good alignment → Moderate performance |

**Key Question**: Why does dseed2025 achieve BEST performance despite WORST alignment?

## Hypotheses to Test

### Hypothesis 1: Tree Complexity Drives Alignment Quality

**Prediction**: dseed2025's optimal params create deeper, more complex trees that don't align well at the leaf level but maintain good decision boundaries.

**Parameters to check**:
- `max_depth`: Higher depth = more specific leaves = worse alignment
- `num_leaves`: More leaves = more fragmentation = worse alignment
- `min_child_samples`: Lower values = smaller leaves = worse alignment

**Expected pattern for dseed2025**:
```
max_depth: 7-8 (vs 4-5 for dseed55)
num_leaves: 40-50 (vs 20-30 for dseed55)
min_child_samples: 5-10 (vs 20-40 for dseed55)
```

---

### Hypothesis 2: Regularization Differences

**Prediction**: dseed2025 has less regularization, allowing overfitting to training data, which creates specific trees with poor alignment.

**Parameters to check**:
- `reg_lambda`: Lower values = less L2 regularization
- `reg_alpha`: Lower values = less L1 regularization
- `min_split_gain`: Lower values = easier to create splits

**Expected pattern for dseed2025**:
```
reg_lambda: 0.5-1.0 (vs 2.0-5.0 for dseed55)
reg_alpha: 0.0-0.1 (vs 0.5-1.0 for dseed55)
```

---

### Hypothesis 3: Learning Rate and Ensemble Size

**Prediction**: dseed2025 uses more boosting iterations with lower learning rate, creating more complex ensemble.

**Parameters to check**:
- `learning_rate`: Lower = more trees needed
- `n_estimators`: Higher = more trees = more complexity

**Expected pattern for dseed2025**:
```
learning_rate: 0.01-0.03 (vs 0.05-0.1 for dseed55)
n_estimators: 300-500 (vs 100-200 for dseed55)
```

---

### Hypothesis 4: Threshold Selection

**Prediction**: Threshold doesn't affect AUROC, but might indicate different class balance handling.

**Parameters to check**:
- `optimal_threshold`: Very low thresholds suggest aggressive positive class prediction

**Expected pattern**:
```
All datasets should have low thresholds (0.02-0.10) due to 10% positive class
No strong correlation with alignment quality
```

---

## Interpretation Guide

### When Comparing Deployment vs Optimal Within Each Dataset

**Look for**:
1. **CV Score Gap**: Negative gap indicates overfitting on smaller training data
2. **Parameter Differences**: Which parameters diverge most between scenarios?
3. **Complexity Pattern**: Does optimal consistently choose more/less complex models?

**Example Analysis**:
```
dseed2025:
  Deployment CV: 0.625 (on 41K samples)
  Optimal CV: 0.610 (on 10K samples)
  Gap: -0.015 (optimal worse, likely overfitting)

  Deployment max_depth: 5
  Optimal max_depth: 8 (much deeper!)

Interpretation: Optimal params found complex model that overfits to 10K samples
```

---

### When Comparing Across Datasets

**Look for patterns that correlate with alignment quality**:

1. **Complexity Metrics** (sum of max_depth + log(num_leaves)):
   - dseed55: Low complexity → Good alignment
   - dseed2025: High complexity → Bad alignment
   - dseed6765: Medium complexity → Medium alignment

2. **Regularization Strength** (reg_lambda + reg_alpha):
   - Higher regularization → Simpler trees → Better alignment
   - Lower regularization → Complex trees → Worse alignment

3. **Consistency Across Scenarios**:
   - Similar deployment vs optimal params → Stable results
   - Very different params → Dataset-specific behavior

---

## Expected Findings

### Scenario A: Complexity Hypothesis Confirmed

If we find dseed2025 has much higher max_depth and num_leaves:

**Interpretation**: Deep, specific trees achieve good AUROC by fitting decision boundaries well, but create highly fragmented leaf space that doesn't generalize to different training data distribution (41K unsampled data). The test points co-occur in leaves because they share the same feature space, even though training points (from 10K subset) don't.

**Action**: Document that leaf alignment quality depends on tree complexity. For fair comparison, could enforce consistent complexity constraints across datasets.

---

### Scenario B: Sample Size Effect

If optimal params are consistently more complex across all datasets:

**Interpretation**: The 10K training sample size systematically leads Optuna to find overly complex models. The adaptive search space (from TUNING_FIXES.md) helped reduce overfitting but didn't eliminate the fundamental issue of tuning on different sample sizes.

**Action**: Consider tuning both scenarios on same 10K subset for fair comparison, or apply stronger complexity penalties for small datasets.

---

### Scenario C: No Clear Pattern

If hyperparameters are similar across datasets:

**Interpretation**: The alignment differences may be driven by data distribution differences, not hyperparameters. Different random seeds create populations with different feature distributions, and some distributions happen to align better across train/unsampled splits.

**Action**: Examine data distributions (feature statistics, class balance, feature correlations) across datasets to identify what makes dseed2025 special.

---

## Key Metrics to Extract

From the comparison output, calculate:

1. **Complexity Score** = max_depth × log(num_leaves)
2. **Regularization Score** = reg_lambda + reg_alpha
3. **Ensemble Complexity** = n_estimators × learning_rate
4. **CV-to-Test Gap** = test_auroc - best_cv_score

Create correlation matrix:
```
Complexity Score vs Hallucination Rate
Regularization Score vs Alignment Quality
CV-to-Test Gap vs Performance Stability
```

---

## Next Steps After Analysis

### If Complexity is the Driver:

1. **Add complexity constraint** to tuning:
   ```python
   max_complexity_score = max_depth * np.log(num_leaves)
   if max_complexity_score > 50:  # threshold based on alignment quality
       return float('inf')  # reject this trial
   ```

2. **Report separate analyses**:
   - Low complexity models (max_depth ≤ 5, num_leaves ≤ 30)
   - High complexity models (max_depth > 6, num_leaves > 40)

### If Sample Size is the Driver:

1. **Retune optimal on 41K unsampled** (same as deployment)
2. **Use nested CV** to prevent overfitting on small 10K dataset
3. **Report both**: "Fair comparison" (same data) vs "Realistic" (different data)

### If No Clear Pattern:

1. **Analyze data distributions** across dseeds
2. **Check feature importance** for each dataset
3. **Examine class balance** and feature correlations
4. **Consider dataset quality** as a confounding factor

---

## Documentation for Paper/Report

Include in your analysis:

1. **Table**: Hyperparameter comparison across datasets
2. **Scatter plot**: Complexity score vs hallucination rate
3. **Discussion**: Why optimal params create different tree structures
4. **Limitation**: Tuning on different sample sizes makes direct comparison challenging
5. **Recommendation**: Future work should control for sample size when comparing scenarios

---

## Summary

This analysis will help us understand whether:
- Leaf alignment quality is a hyperparameter artifact (fixable)
- Or a fundamental property of the data distribution (not fixable)

The goal is to determine if we can **predict** when optimal params will have poor alignment, and whether poor alignment is actually a **problem** given that AUROC remains good.
