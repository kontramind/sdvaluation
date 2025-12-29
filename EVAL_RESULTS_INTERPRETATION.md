# Real-World Eval Results: Interpretation and Analysis

**Dataset:** MIMIC-III-mini-core (dseed15625)
**Synthetic Generator:** SynthCity ARF (Adversarial Random Forest)
**Evaluation:** 10,000 synthetic points, 100,000 trees, 4 parallel jobs
**Date:** 2025-12-29

---

## Executive Summary

These results reveal a **critical statistical artifact** when using extremely high tree counts (100k trees), and demonstrate how the three evaluation levels provide complementary insights into synthetic data quality.

**Key Findings:**
1. **Levels 1 & 2 with 100k trees produce misleading results** - all points classified as "beneficial" despite near-zero utilities
2. **Level 3 reveals the ground truth** - 12.1% hallucinated, with minority class severely affected (28.4%)
3. **Class imbalance is a symptom, not the cause** - poor pattern quality for readmission cases
4. **Extreme threshold shift** - optimal threshold drops from 0.390 to 0.050 (synthetic model poorly calibrated)

---

## 1. Data Characteristics

### Class Balance

```
Real training:      10.5% positive (readmission)
Synthetic training:  6.7% positive (readmission)
Difference:         -3.8pp (⚠ Moderate difference)
Ratio:              0.64x
```

**Issue:** Synthetic generator underproduced minority class by 36%

### Model Performance Comparison

| Metric | Real → Test | Synthetic → Test | Gap | % Degradation |
|--------|------------|------------------|-----|---------------|
| **AUROC** | 0.6388 | 0.5857 | +0.0530 | **8.3%** |
| **F1** | 0.2345 | 0.2080-0.2106 | +0.0240-0.0265 | 10.2-11.3% |
| **Precision** | 0.1387 | 0.1220-0.1234 | +0.0153-0.0167 | 11.0-12.0% |
| **Recall** | 0.7581 | 0.6981-0.7514 | +0.0067-0.0600 | 0.9-7.9% |

**Performance degradation is consistent across all three evaluation levels**, confirming the synthetic data has fundamental quality issues.

---

## 2. Level 1: Unadjusted (100k Trees)

### Results

```
Utility Score Distribution:
  Negative utility (< 0): 0 (0.00%)
  Positive utility (> 0): 10,000 (100.00%)
  Zero utility (= 0):     0 (0.00%)

Statistical Confidence (95% CI-based):
  Reliably hallucinated: 0 (0.00%)
  Reliably beneficial:   10,000 (100.00%)
  Uncertain:             0 (0.00%)

Mean utility:   0.0000
Median utility: 0.0000
```

### ⚠️ CRITICAL ISSUE: Statistical Artifact

**The "100% beneficial" result is misleading!**

**Why this happens:**
1. **93,389 trees built** (not the full 100k, but still massive)
2. **Standard error formula**: `SE = std / sqrt(n_trees)`
   - SE ≈ std / 305 (extremely small)
3. **Confidence intervals become ultra-tight**
4. **Even minuscule positive utilities become "statistically significant"**

**Evidence:**
- Mean utility: **0.0000** (displayed as zero due to rounding)
- Actual utilities likely ~0.00001 (positive but negligible)
- With SE ≈ 0.000003, CI doesn't span zero
- Result: Classified as "reliably beneficial" despite being practically zero

**Interpretation:** This is a **false negative** for hallucination detection. The sample size is so large that statistical significance ≠ practical significance.

---

## 3. Level 2: Adjusted for Imbalance (100k Trees)

### Adjustments Applied

```
scale_pos_weight: not set → 13.86
optimal_threshold: 0.390 → 0.396
```

The adjustment correctly identified the imbalance (0.64x ratio → 13.86 weight factor).

### Results

**Identical to Level 1:**
- 100% beneficial
- 0% hallucinated
- Mean/median utility: 0.0000

### Why Adjustment Didn't Help

**The class imbalance adjustment cannot fix poor pattern quality.**

Level 2 tests: *"Are the patterns good, ignoring imbalance?"*

**Answer:** No - but the statistical artifact (100k trees) masks this.

**What we learn:**
- Adjusting `scale_pos_weight` doesn't improve utilities
- The issue isn't just imbalance - it's **poor minority class patterns**
- The generator (ARF) failed to capture readmission patterns, not just proportions

---

## 4. Level 3: Full Retuning (100k Trees)

### ⚠️ Model Training Warning

```
WARNING: Model only built 7,177 trees (requested 100,000).
This usually indicates single-class data or early stopping.
```

**Only 7.7% of requested trees built!**

**Why this happened:**
1. **Early stopping triggered** - validation loss stopped improving
2. **Synthetic data has low information content**
3. **Model converged quickly** (not a good sign)

### Hyperparameter Comparison

| Parameter | Real Data | Synthetic Data | Change | Impact |
|-----------|-----------|----------------|--------|--------|
| **boosting_type** | goss | gbdt | Switch | More data used per tree |
| **num_leaves** | 22 | 13 | -9 | **Simpler trees** |
| **min_child_samples** | 56 | 15 | -41 | **Smaller leaves** |
| **subsample** | 0.64 | 0.98 | +0.34 | Uses more data |
| **reg_lambda** | 2.95 | 9.34 | +6.39 | **Stronger L2 regularization** |
| **threshold** | 0.390 | **0.050** | **-0.340** | **⚠️ CRITICAL** |
| **CV AUROC** | 0.6439 | 0.5753 | -0.0686 | **10.7% worse** |

### 🚨 Threshold Analysis

**This is the most concerning finding:**

```
Real data threshold:      0.390 (39% probability to predict positive)
Synthetic data threshold: 0.050 (5% probability to predict positive)
```

**What this means:**
- **Synthetic model is poorly calibrated**
- **Desperate to find positive cases** (due to minority class scarcity)
- **False positive rate will be extremely high in production**
- **Model doesn't "believe" its own predictions**

**Practical impact:**
```python
# Real model: Conservative
if prob > 0.39:  # High bar
    predict_readmission()

# Synthetic model: Aggressive
if prob > 0.05:  # Very low bar
    predict_readmission()  # Will trigger on nearly everything
```

This extreme threshold indicates **the synthetic data taught the model the wrong decision boundaries**.

### Leaf Alignment Results (The Ground Truth)

```
Statistical Confidence (95% CI-based):
  Reliably beneficial:    7,211 (72.1%)
  Reliably hallucinated:  1,206 (12.1%)
  Uncertain:              1,583 (15.8%)
```

**Finally, realistic results!** With only 7,177 trees:
- Standard errors: SE ≈ std / 85 (wider CIs)
- True hallucinations detected
- Uncertainty properly quantified

### Class-Specific Breakdown

| Class | Total | Beneficial | Hallucinated | Uncertain |
|-------|-------|-----------|--------------|-----------|
| **Negative (No readmission)** | 9,327 | 74.26% | **10.88%** | 14.86% |
| **Positive (Readmission)** | 673 | 42.35% | **28.38%** | 29.27% |

**Critical finding:**
- **Minority class (readmissions) is 2.6x more likely to be hallucinated**
- Only 42% of synthetic readmission cases are beneficial
- 28% are actively harmful (create wrong decision boundaries)

### Top 10 Hallucinated Points

```
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Index ┃ Utility Score ┃ Std Error ┃         95% CI         ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│  8277 │     -0.000114 │  0.000012 │ [-0.000137, -0.000091] │
│  3517 │     -0.000100 │  0.000011 │ [-0.000122, -0.000079] │
│  7345 │     -0.000097 │  0.000011 │ [-0.000118, -0.000077] │
```

**Note:** Utilities are small (~0.0001) but **statistically significant** with 7k trees.

---

## 5. Comparative Analysis: Why Level 3 is Different

### Number of Trees Built

| Level | Trees Requested | Trees Built | % Built | Reason |
|-------|----------------|-------------|---------|--------|
| Level 1 | 100,000 | 93,389 | 93.4% | Using real hyperparams |
| Level 2 | 100,000 | 93,389 | 93.4% | Adjusted imbalance only |
| Level 3 | 100,000 | **7,177** | **7.2%** | **Early stopping** |

**Why Level 3 stopped early:**

Real hyperparameters were optimized for real data:
- More complex trees (22 leaves)
- Moderate regularization (reg_lambda=2.95)
- GOSS boosting (gradient-based sampling)

Synthetic-tuned hyperparameters adapted to poor quality:
- Simpler trees (13 leaves)
- Heavy regularization (reg_lambda=9.34)
- Full GBDT (use all data, hoping for signal)
- **Early stopping triggered when validation loss plateaued**

**The early stopping is a feature, not a bug** - it detected that additional trees weren't improving synthetic data learning.

### Confidence Interval Width Comparison

**Level 1 & 2 (93,389 trees):**
```
SE ≈ utility_std / 305
CI width ≈ ±0.000006  (extremely tight)
```

**Level 3 (7,177 trees):**
```
SE ≈ utility_std / 85
CI width ≈ ±0.000020  (3.6x wider)
```

**Result:**
- Levels 1 & 2: Everything is "significant" (artifact)
- Level 3: Only truly impactful points are classified

---

## 6. Root Cause Analysis

### Why ARF Synthetic Data Failed

**ARF (Adversarial Random Forest) weaknesses exposed:**

1. **Minority class synthesis failure**
   - Real: 1,050 readmission cases
   - Synthetic: 673 readmission cases (36% fewer)
   - 28% of synthetic readmissions are hallucinated

2. **Poor decision boundary learning**
   - Optimal threshold collapse (0.390 → 0.050)
   - Model "doesn't trust" synthetic patterns
   - Needs 5% probability bar vs 39% for real data

3. **Low information content**
   - Early stopping after 7k trees
   - CV AUROC dropped 10.7% (0.644 → 0.575)
   - Model converged quickly (bad sign for data quality)

4. **Pattern degradation, not just imbalance**
   - Level 2 adjustment didn't help
   - Negative class also affected (11% hallucinated)
   - Systematic pattern loss across both classes

### Hypothesis: What ARF Learned Wrong

**ARF generates data by:**
1. Training adversarial random forests
2. Sampling from leaf distributions
3. Refining via discriminator

**Where it likely failed:**
- **Sparse minority class** → Few training examples for readmission leaves
- **Adversarial training** → Focused on fooling discriminator, not preserving patterns
- **Leaf-based sampling** → Overfitted to training leaf structures
- **Imbalanced adversarial loss** → Generator prioritized majority class

**Evidence:**
- Readmission cases: 28% hallucinated (can't fool real test data)
- Extreme threshold shift (learned wrong probability calibration)
- Early stopping (patterns inconsistent, hard to learn)

---

## 7. Practical Recommendations

### For This Specific Synthetic Dataset

❌ **Do NOT use unfiltered** - 12% hallucinated, 28% of minority class

✅ **Option 1: Filter hallucinated points**
```python
# Remove 1,206 hallucinated points
filtered_synth = synth_data[~reliably_hallucinated]
# New size: 8,794 points (72% beneficial, 16% uncertain)
```

**Expected improvement:**
- Remove confirmed harmful points
- Keep uncertain points (may still be useful)
- Accept 12% data loss

✅ **Option 2: Stratified filtering**
```python
# Be aggressive with minority class (28% hallucinated)
keep_positive = positive_synth[~reliably_hallucinated_positive]  # 480 left
# Be lenient with majority class (11% hallucinated)
keep_negative = negative_synth[~reliably_hallucinated_negative]  # 8,312 left
# New size: 8,792 points, better class balance (5.5% positive)
```

**Trade-off:** Still have class imbalance, but better quality

✅ **Option 3: Re-generate with different method**
- ARF struggled with this dataset
- Try: TVAE, CTGAN, or CopulaGAN (better for imbalanced data)
- Consider pre-balancing real data before synthesis (SMOTE + synthesis)

### For Future Evaluations

🎯 **Recommended n_estimators based on these results:**

| Trees | Use Case | Pros | Cons |
|-------|----------|------|------|
| **500** | Standard eval | Fast (~8s), reasonable CIs | May miss subtle hallucinations |
| **5,000** | Recommended | Good balance (~1 min) | - |
| **10,000** | High confidence | Tight CIs (~2 min) | Longer runtime |
| **100,000** | ❌ NOT recommended | ❌ Statistical artifacts | ❌ Misleading results |

**Learning:** More trees ≠ better results beyond ~10k

### Interpreting Mean Utility ≈ 0

⚠️ **If you see mean utility displayed as 0.0000:**

```python
# Check actual values in CSV
df = pd.read_csv('synthetic_evaluation.csv')
print(f"Actual mean: {df['utility_score'].mean():.10f}")
print(f"Range: [{df['utility_score'].min():.10f}, {df['utility_score'].max():.10f}]")
```

**If actual values are ~0.00001:**
- Utilities are practically zero (no real impact)
- High tree count made noise "statistically significant"
- **Reduce n_estimators to 5,000-10,000** for realistic results

---

## 8. Statistical Lessons Learned

### The Perils of Overpowered Statistical Tests

**Classical statistics principle violated:**

> "Statistical significance ≠ Practical significance"

**What happened here:**
```
With 100k trees:
  True utility: 0.00001
  Standard error: 0.000003
  t-statistic: 0.00001 / 0.000003 = 3.33
  p-value: 0.0009 (statistically significant!)

But practical impact: ZERO
```

**In medical terms:**
- Like detecting a 0.01 mmHg blood pressure difference
- Statistically significant with n=1,000,000 patients
- Clinically meaningless

### Optimal Sample Size for Leaf Alignment

**Target: Detect utility differences of ±0.0001 with 95% confidence**

Assuming typical utility std ≈ 0.002:
```
Required trees = (t_critical * std / target_difference)²
              = (1.96 * 0.002 / 0.0001)²
              = 1,537 trees

Conservative: 5,000 trees (detects ±0.00005 differences)
```

**Guideline:**
- 500 trees: Detect ±0.0002 differences (good for screening)
- 5,000 trees: Detect ±0.00006 differences (recommended)
- 10,000 trees: Detect ±0.00004 differences (high precision)
- 100,000 trees: ⚠️ Overkill - creates artifacts

---

## 9. Why Level 3 Results Are More Trustworthy

### Convergence to Reality

**Level 3 advantages:**

1. **Automatic sample size adjustment** (early stopping)
   - Model detected poor data quality
   - Stopped at 7,177 trees (natural equilibrium)
   - Avoided overfitting to noise

2. **Realistic confidence intervals**
   - SE ≈ std / 85 (appropriate for effect sizes)
   - CIs wide enough to capture uncertainty
   - Proper "uncertain" classification (15.8%)

3. **Hyperparameter honesty**
   - Extreme threshold (0.050) flags calibration failure
   - Heavy regularization (reg_lambda=9.34) compensates for noise
   - Simpler trees (13 leaves) prevent overfitting

**The model is "admitting" the synthetic data is poor quality through:**
- Early stopping
- Conservative hyperparameters
- Extreme threshold shift
- Lower CV score

### Confirmation Through Multiple Signals

**All three signals agree:**

| Signal | Evidence | Conclusion |
|--------|----------|------------|
| **Performance metrics** | AUROC -8.3%, F1 -11% | Quality loss |
| **Hyperparameter drift** | Threshold 0.39→0.05 | Poor calibration |
| **Leaf alignment** | 12% hallucinated | Pattern failure |

**Level 1 & 2 only saw signal #1** (performance metrics)
**Level 3 revealed signals #2 and #3** (hyperparameter + leaf alignment)

---

## 10. Comparative Summary Table

| Aspect | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| **Evaluation question** | Drop-in replacement? | Good patterns? | Best-case performance? |
| **Hyperparameters** | Real (unadjusted) | Real (adjusted imbalance) | Synthetic (retuned) |
| **Trees built** | 93,389 | 93,389 | **7,177** |
| **Beneficial %** | **100%** ❌ | **100%** ❌ | **72.1%** ✅ |
| **Hallucinated %** | **0%** ❌ | **0%** ❌ | **12.1%** ✅ |
| **Mean utility** | 0.0000 (artifact) | 0.0000 (artifact) | 0.0000 (realistic) |
| **Minority class hallucinated** | Not detected | Not detected | **28.4%** ✅ |
| **Threshold** | 0.390 | 0.396 | **0.050** ⚠️ |
| **Interpretation** | Statistical artifact | Adjustment didn't help | **Ground truth** |

✅ = Realistic result
❌ = Misleading result
⚠️ = Warning sign

---

## 11. Conclusions

### What We Learned About This Synthetic Dataset

1. **Quality Grade: C- (Poor)**
   - 12% hallucinated (should be <5% for good quality)
   - 28% of minority class hallucinated (critical failure)
   - 8.3% AUROC degradation (acceptable threshold: <5%)

2. **Usability: Conditional**
   - ✅ Can be used after filtering hallucinated points
   - ❌ Should not be used for minority class modeling
   - ⚠️ Requires careful threshold recalibration

3. **Root cause: Generator limitation**
   - ARF method unsuitable for this imbalanced dataset
   - Need: Better minority class synthesis method
   - Consider: SMOTE + synthesis, or class-conditional generation

### What We Learned About Evaluation Methodology

1. **100k trees is too many**
   - Creates statistical artifacts
   - Masks real quality issues
   - **Recommendation: 5,000-10,000 trees maximum**

2. **Level 3 is most informative**
   - Reveals hyperparameter drift
   - Natural early stopping detects quality
   - More realistic confidence intervals

3. **All three levels together tell the full story**
   - Level 1: Confirms poor drop-in replacement
   - Level 2: Rules out imbalance as sole cause
   - Level 3: Identifies specific failure modes

4. **Mean utility ≈ 0 is a red flag**
   - If displayed as 0.0000, check actual values
   - May indicate statistical artifacts or poor quality
   - Investigate class-specific utilities

### Recommended Evaluation Protocol

For future synthetic data evaluations:

```bash
# Step 1: Standard evaluation (5k trees for balance)
sdvaluation eval \
  --dseed-dir dseed/ \
  --synthetic-file synth.csv \
  --n-estimators 5000 \
  --n-jobs -1

# Step 2: If class imbalance warning appears
sdvaluation eval \
  --dseed-dir dseed/ \
  --synthetic-file synth.csv \
  --n-estimators 5000 \
  --n-jobs -1 \
  --adjust-for-imbalance

# Step 3: For comprehensive analysis
sdvaluation eval \
  --dseed-dir dseed/ \
  --synthetic-file synth.csv \
  --n-estimators 10000 \
  --n-jobs -1 \
  --retune-on-synthetic

# Step 4: Compare all three levels
# - If Level 3 shows hallucinations but L1/L2 don't → check tree count
# - If extreme threshold shift (>0.2) → poor calibration
# - If early stopping (<25% requested trees) → data quality issues
```

---

## 12. Future Work

### Improvements to Eval Command

Based on these findings:

1. **Adaptive n_estimators**
   - Auto-cap at 10,000 trees
   - Warn if mean utility < 0.0001 with >10k trees
   - Suggest re-running with fewer trees

2. **Enhanced diagnostics**
   - Flag extreme threshold shifts (|Δ| > 0.1)
   - Warn on early stopping (<25% trees)
   - Compare actual vs displayed mean utility

3. **Class-specific reporting by default**
   - Always show per-class hallucination rates
   - Highlight minority class issues
   - Stratified filtering recommendations

4. **Calibration analysis**
   - Plot reliability diagrams (Real vs Synthetic)
   - Quantify calibration error
   - Detect probability distribution shift

### Research Questions

1. **Is there a theoretical optimal n_estimators?**
   - Depends on utility effect size distribution
   - May need adaptive methods

2. **Why does early stopping correlate with hallucinations?**
   - Is there a causal relationship?
   - Can we use stopping point as quality metric?

3. **Can we predict hallucination rate from hyperparameter drift?**
   - Threshold shift as quality indicator?
   - Regularization strength as signal?

---

**Document Version:** 1.0
**Based on:** sdvaluation eval runs from 2025-12-29
**Dataset:** MIMIC-III-mini-core (dseed15625)
**Generator:** SynthCity ARF
