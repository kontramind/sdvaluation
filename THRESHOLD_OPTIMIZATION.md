# Threshold Optimization Feature

## Overview

Added CV-based threshold optimization to the dual evaluation workflow. Each scenario now uses its own optimized classification threshold instead of a fixed 0.5.

## What Changed

### Workflow Enhancement

**Before:**
```
40k tuning → best_params_40k
10k tuning → best_params_10k

Scenario 1: Use params_10k with threshold=0.5
Scenario 2: Use params_40k with threshold=0.5
```

**After:**
```
40k tuning → best_params_40k + threshold_40k (optimized on CV)
10k tuning → best_params_10k + threshold_10k (optimized on CV)

Scenario 1: Use params_10k with threshold_10k
Scenario 2: Use params_40k with threshold_40k
```

### Optimization Strategy

1. **Hyperparameter tuning**: Find best LGBM parameters via Optuna (as before)
2. **CV predictions**: Run one more CV with best params to collect predictions
3. **Threshold search**: Test thresholds from 0.10 to 0.88 (step 0.02)
4. **Select optimal**: Choose threshold that maximizes chosen metric

## Supported Metrics

| Metric | When to Use | Medical Context |
|--------|-------------|-----------------|
| **f1** (default) | Balanced precision/recall | General use |
| **precision** | Minimize false alarms | Limited intervention capacity |
| **recall** | Minimize missed cases | High-stakes outcomes (readmissions) |
| **youden** | Balance sensitivity/specificity | Diagnostic tests |

## Usage

### CLI

```bash
# Default: F1-optimized threshold
sdvaluation dual-eval \
  --tuning-data population_40k.csv \
  --real-train real_train_10k.csv \
  --synth-train synth_train_10k.csv \
  --real-test real_test_10k.csv \
  --encoding-config encoding.yaml

# Optimize for recall (minimize false negatives)
sdvaluation dual-eval \
  --tuning-data population_40k.csv \
  --real-train real_train_10k.csv \
  --synth-train synth_train_10k.csv \
  --real-test real_test_10k.csv \
  --encoding-config encoding.yaml \
  --threshold-metric recall

# Optimize for precision (minimize false positives)
sdvaluation dual-eval \
  ... \
  --threshold-metric precision
```

### Python API

```python
from sdvaluation.dual_evaluation import run_dual_evaluation

results = run_dual_evaluation(
    tuning_data=Path("population_40k.csv"),
    real_train=Path("real_train_10k.csv"),
    synth_train=Path("synth_train_10k.csv"),
    real_test=Path("real_test_10k.csv"),
    encoding_config=Path("encoding.yaml"),
    threshold_metric='recall',  # or 'f1', 'precision', 'youden'
    ...
)

print(f"40k threshold: {results['scenario_2_deployment']['threshold']:.3f}")
print(f"10k threshold: {results['scenario_1_optimal']['threshold']:.3f}")
print(f"Threshold gap: {results['threshold_gap']:.3f}")
```

## Output Changes

### Terminal Output

```
Phase 1: Hyperparameter Tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/2] Tuning on 40k Population Data
  Running 100 Optuna trials with 5-fold CV...
  Optimizing threshold for: f1
  ✓ Best CV AUROC: 0.8234
  ✓ Optimal threshold: 0.32                    ← NEW
  ✓ Saved: params_40k_20251211_143022.json

[2/2] Tuning on 10k Real Train
  Running 100 Optuna trials with 5-fold CV...
  Optimizing threshold for: f1
  ✓ Best CV AUROC: 0.8156
  ✓ Optimal threshold: 0.28                    ← NEW
  ✓ Saved: params_10k_20251211_143022.json

Threshold Comparison:                          ← NEW
  40k-optimized: 0.320
  10k-optimized: 0.280
  Gap: 0.040
```

### JSON Files

**params_40k_*.json:**
```json
{
  "params": {
    "objective": "binary",
    "num_leaves": 31,
    ...
  },
  "threshold": 0.32,                    // NEW
  "threshold_metric": "f1",             // NEW
  "cv_score": 0.8234,
  "n_trials": 100,
  "n_folds": 5,
  "source": "tuning_data_40k"
}
```

**summary_*.json:**
```json
{
  "timestamp": "20251211_143022",
  "threshold_metric": "f1",             // NEW
  "transfer_gap": 0.0089,
  "threshold_gap": 0.04,                // NEW
  "scenario_1_optimal": {
    "params_source": "real_train_10k",
    "threshold": 0.28,                  // NEW
    "real_auroc": 0.8012,
    "synth_auroc": 0.7234,
    "gap": -0.0778
  },
  "scenario_2_deployment": {
    "params_source": "tuning_data_40k",
    "threshold": 0.32,                  // NEW
    "real_auroc": 0.7923,
    "synth_auroc": 0.7102,
    "gap": -0.0821
  },
  ...
}
```

## Why This Matters

### 1. More Realistic Deployment
In practice, you'd tune thresholds on validation data, not use 0.5 blindly.

### 2. Better for Imbalanced Data
MIMIC-III has ~10% positive class (readmissions). Optimal threshold is typically < 0.5.

### 3. Fair Comparison Maintained
Within each scenario, Real and Synth use the **same** threshold:
- Scenario 1: Both use `threshold_10k`
- Scenario 2: Both use `threshold_40k`

### 4. Tests Threshold Transfer
Similar to hyperparameter transfer, we can now check:
- Does `threshold_40k` differ from `threshold_10k`?
- If gap is large → another dimension of distribution shift

## Interpretation Guide

### Threshold Gap

| Gap | Interpretation |
|-----|----------------|
| < 0.05 | Excellent - thresholds transfer well |
| 0.05-0.10 | Acceptable - minor distribution shift |
| > 0.10 | Poor - significant distribution mismatch |

### Example Scenarios

**Scenario A: Good Transfer**
```
threshold_40k = 0.30
threshold_10k = 0.28
gap = 0.02 ✓

→ Distributions are similar
→ Both scenario comparisons are valid
```

**Scenario B: Poor Transfer**
```
threshold_40k = 0.45
threshold_10k = 0.25
gap = 0.20 ✗

→ 40k data has different characteristics
→ Scenario 2 results may be confounded
```

## Implementation Details

### Threshold Search Algorithm

```python
def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    thresholds = np.arange(0.1, 0.9, 0.02)  # Test 40 values
    best_score = -inf
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = compute_metric(y_true, y_pred, metric)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold
```

### CV Prediction Collection

```python
def get_cv_predictions(X_train, y_train, params, n_folds=5):
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

    # Get out-of-fold predictions for entire training set
    y_pred_proba = cross_val_predict(
        model, X_train, y_train,
        cv=cv,
        method='predict_proba'
    )[:, 1]

    return y_pred_proba
```

**Why this works:**
- Each sample gets predicted when it's in validation fold
- Predictions are made by models that **didn't** see that sample during training
- Simulates deployment scenario (predicting on unseen data)

## Performance Impact

**Additional Runtime:**
- One extra CV run per tuning phase: ~2-5 minutes
- Threshold search: ~1 second
- **Total overhead**: ~4-10 minutes (negligible compared to hyperparameter tuning)

**Total Runtime Estimate:**
- Hyperparameter tuning (2x): 30-60 minutes
- Threshold optimization (2x): 4-10 minutes
- Evaluation (4x): 5-10 minutes
- Leaf alignment (2x): 10-20 minutes
- **Total**: 50-100 minutes

## Medical Use Case Example

### MIMIC-III Readmission Prediction

**Context:**
- ~10% of patients readmitted within 30 days
- Goal: Identify high-risk patients for intervention

**Threshold Choice:**

| Metric | Threshold | Precision | Recall | Use Case |
|--------|-----------|-----------|--------|----------|
| **F1** | 0.28 | 18% | 40% | Balanced |
| **Recall** | 0.18 | 12% | 55% | Catch more readmissions |
| **Precision** | 0.45 | 25% | 25% | Limit interventions |
| **Youden** | 0.32 | 19% | 38% | Diagnostic test |

**Clinical Trade-off:**
- Lower threshold (recall): More interventions, catch more readmissions
- Higher threshold (precision): Fewer interventions, miss some readmissions

## Next Steps

Now that threshold optimization is implemented, you can:

1. **Test with your data**:
   ```bash
   sdvaluation dual-eval \
     --tuning-data your_40k.csv \
     --real-train your_real_10k.csv \
     --synth-train your_synth_10k.csv \
     --real-test your_test_10k.csv \
     --encoding-config your_encoding.yaml \
     --threshold-metric f1
   ```

2. **Compare metrics**: Try different `--threshold-metric` values

3. **Analyze threshold gaps**: Check if thresholds transfer well

4. **Report findings**: Include threshold information in your evaluation

## Files Modified

- ✅ `sdvaluation/tuning.py` (added threshold optimization)
- ✅ `sdvaluation/dual_evaluation.py` (integrated thresholds)
- ✅ `sdvaluation/cli.py` (added CLI flag)

---

**Status**: ✅ Ready for testing with threshold optimization!
