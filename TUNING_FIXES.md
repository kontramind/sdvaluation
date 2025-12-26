# Hyperparameter Tuning Fixes

## Summary of Bugs Found and Fixed

### Bug #1: Search Space Allows Massive Overfitting on Small Datasets ✅ FIXED

**Problem:**
- Original search space allowed up to 100 leaves, depth 15, and zero regularization
- With only 10,000 training samples, this caused severe overfitting
- CV scores looked good (0.64) but test performance collapsed (0.57-0.59)

**Fix:**
- Added adaptive search space based on dataset size
- For datasets < 15K samples:
  - `num_leaves`: 10-50 (was 4-100)
  - `max_depth`: 3-8 (was 1-15)
  - `reg_lambda`: 0.5-10.0 (was 0.0-10.0) - forced regularization
  - `min_child_samples`: 5-60 (was 1-60)

**Impact:** Should reduce overfitting and improve test performance for optimal scenario

---

### Bug #2: Threshold Search Range Too Narrow ✅ FIXED

**Problem:**
- Original threshold search: 0.1 to 0.9
- With 10.5% positive class, optimal F1 threshold often < 0.1
- Could not explore low thresholds needed for imbalanced data

**Fix:**
- Expanded threshold search to 0.01-0.9 (was 0.1-0.9)
- Updated both threshold optimization functions

**Impact:** Allows finding better thresholds for imbalanced datasets

---

### Bug #3: Sample Size Mismatch (Acknowledged but Not Fixed)

**Problem:**
- Deployment params tuned on 41,532 samples
- Optimal params tuned on 10,000 samples
- Comparing hyperparameters from different sample sizes is unfair

**Why Not Fixed:**
- This is a fundamental design choice in your experiment
- Deployment = realistic scenario (large population data)
- Optimal = best-case scenario (small clean data)
- Fixing would require redesigning the experimental framework

**Recommendation:**
- Document this limitation in your analysis
- Consider reporting separate results or normalizing for sample size
- Or: tune both on the same 10K subset for fair comparison

---

### Bug #4: No Validation Warning (Added)

**Added:**
- Warning message when optimal CV score < deployment CV score
- Alerts user to potential overfitting on smaller training dataset

---

## Testing the Fixes

Re-run tuning on one of your datasets:

```bash
# Test on dseed55 (showed largest tuning failure: -0.0534 gap)
uv run sdvaluation tune --dseed-dir ../rd-lake/dseed55/ --n-trials 100 --n-jobs 4
```

### Expected Improvements:

**Before fixes:**
- Optimal CV: ~0.642
- Optimal Test: ~0.571 (gap: -0.071, severe overfitting)
- Deployment always beats optimal

**After fixes:**
- Optimal CV: ~0.610-0.620 (lower due to forced regularization)
- Optimal Test: ~0.590-0.610 (gap: -0.010 to -0.020, much better!)
- Optimal should now be competitive with or beat deployment

---

## What Changed in the Code

### `sdvaluation/tuner.py`

1. **Lines 212-243**: Added adaptive search space based on dataset size
2. **Line 538**: Expanded threshold range to 0.01-0.9 (was 0.1-0.9)
3. **Line 791**: Expanded threshold range in alternate function
4. **Lines 988-1000**: Added validation warning for suboptimal tuning

---

## Next Steps

1. **Test the fixes:**
   ```bash
   uv run sdvaluation tune --dseed-dir ../rd-lake/dseed55/ --n-trials 100 --n-jobs 4
   ```

2. **Compare results:**
   - Check if optimal test AUROC improves (should be closer to CV score)
   - Verify deployment params still work well
   - Compare leaf alignment results with new hyperparameters

3. **If results improve:**
   - Re-run tuning on all dseeds (dseed5, dseed55, dseed6765)
   - Re-run leaf alignment analysis with new hyperparameters

4. **If results don't improve enough:**
   - Consider more aggressive fixes:
     - Further reduce max_depth to 5-6 for small datasets
     - Increase minimum reg_lambda to 1.0
     - Add min_data_in_leaf constraint (e.g., 50-100 for 10K samples)

---

## Additional Recommendations

### For Future Experiments:

1. **Use holdout validation set:**
   - Split training data into 80% train + 20% calibration
   - Tune hyperparameters on train
   - Optimize threshold on calibration
   - Evaluate on test
   - Prevents double-dipping on same data

2. **Add early stopping for tuning:**
   ```python
   study.optimize(
       objective,
       n_trials=100,
       callbacks=[optuna.study.MaxTrialsCallback(n_trials=100, states=(optuna.trial.TrialState.COMPLETE,))]
   )
   ```

3. **Try simpler threshold strategy:**
   - Use class prevalence as threshold (0.105 for 10.5% positive class)
   - Or use fixed threshold (0.5) and adjust via class weights

4. **Document limitations:**
   - The 0.617-0.625 AUROC ceiling suggests fundamental feature limitations
   - Even perfect tuning can't overcome weak predictive features
   - Consider feature engineering or external data sources

---

## Summary

**Root Cause:**
Hyperparameter search space was too permissive for small datasets (10K samples), allowing the tuning process to find overfitted parameters that achieved high CV scores but poor test performance.

**Primary Fix:**
Adaptive search space that constrains complexity and enforces regularization for datasets < 15K samples.

**Expected Result:**
Optimal scenario should now achieve test AUROC closer to its CV score, potentially matching or beating deployment scenario.
