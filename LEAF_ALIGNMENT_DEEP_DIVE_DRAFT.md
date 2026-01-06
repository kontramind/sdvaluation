# LEAF ALIGNMENT DEEP DIVE: A Comprehensive Guide

**A Complete Guide to Evaluating Synthetic Data Quality Using Leaf Co-Occurrence Analysis**

---

## Table of Contents

1. [Introduction & Overview](#1-introduction--overview)
2. [The Methodology: Step-by-Step](#2-the-methodology-step-by-step)
3. [Core Concepts](#3-core-concepts)
4. [Comprehensive Worked Example](#4-comprehensive-worked-example)
5. [Advanced Topics](#5-advanced-topics)
6. [Practical Application](#6-practical-application)
7. [Comparisons with Other Methods](#7-comparisons-with-other-methods)
8. [FAQs & Troubleshooting](#8-faqs--troubleshooting)
9. [Appendices](#9-appendices)

---

## 1. Introduction & Overview

### 1.1 What is Leaf Alignment?

Leaf alignment is a method for evaluating synthetic data quality by measuring how well decision boundaries learned from synthetic data generalize to real test data. Unlike traditional metrics that only measure aggregate model performance, leaf alignment provides **point-level quality scores** for each synthetic data point, classifying them as:

- **Reliably Beneficial**: Creates decision boundaries that correctly classify real data
- **Reliably Harmful** (Hallucinated): Creates decision boundaries that misclassify real data
- **Uncertain**: Inconsistent evidence across different decision trees

### 1.2 The Core Idea

The fundamental insight behind leaf alignment is:

> **Train a model on synthetic data, then check if the decision boundaries it learned actually work on real data.**

If synthetic data points teach the model wrong patterns (e.g., "young patients with many medications don't get readmitted" when in reality they do), those points will create decision boundaries that fail on real test data. These are **hallucinated data points** - they have incorrect labels for their feature combinations.

### 1.3 Why Leaf Alignment?

**The Problem with Traditional Metrics**

Traditional evaluation approaches have critical blind spots:

1. **Aggregate metrics** (precision, recall, F1) only show overall performance, not which specific data points are problematic
2. **Statistical distance metrics** (KL divergence, Wasserstein distance) measure distribution similarity but don't directly measure impact on model training
3. **Data Shapley** is computationally expensive (~90 minutes) and can miss systematic hallucination patterns

**What Leaf Alignment Provides**

- **Point-level diagnosis**: Identifies exactly which synthetic points are harmful
- **Fast evaluation**: ~5 minutes vs. hours for alternative methods
- **Systematic hallucination detection**: Catches cases where entire classes are hallucinated
- **Actionable filtering**: Remove harmful points before training production models
- **Interpretable results**: Clear explanation of why each point is beneficial/harmful

### 1.4 Real-World Impact: The Gen2 Case Study

In our evaluation of synthetic data from MIMIC-III hospital readmission prediction:

**Real Training Data (Baseline)**:
- Reliably harmful: 0.25% ✓
- Reliably beneficial: 89.69% ✓
- Performance: Precision 18.34%, Recall 39.96%, F1 25.14%

**Gen2 Synthetic Data (Recursive Generation)**:
- Reliably harmful: **93.39%** ✗✗✗
- Reliably beneficial: **0.54%** ✗✗✗
- Performance: Precision 7.86%, Recall 10.49%, F1 8.99%
- **373× more hallucinated than real data**

**Key Finding**: Data Shapley reported only 3.39% harmful points for the same dataset, missing the catastrophic failure. Leaf alignment detected that 93% of the synthetic data was teaching wrong patterns, explaining the severe performance degradation.

### 1.5 How It Works: 30-Second Summary

```
1. Train LightGBM (500 trees) on synthetic data
   → Creates decision boundaries based on synthetic patterns

2. Pass both synthetic AND real data through the trained trees
   → Track which "leaf" (decision region) each point lands in

3. For each leaf, check: Does it correctly classify REAL data?
   → Calculate leaf utility: accuracy on real data - 0.5

4. Assign utility scores to synthetic points
   → Points in good leaves get positive scores
   → Points in bad leaves get negative scores

5. Aggregate across 500 trees
   → Compute mean, confidence intervals
   → Classify: beneficial / harmful / uncertain
```

### 1.6 What You'll Learn

This guide provides comprehensive coverage of:

**Core Methodology** (Sections 2-4):
- Step-by-step algorithm walkthrough
- Mathematical foundations
- Complete worked example with real calculations

**Advanced Understanding** (Sections 5-6):
- Why we don't check synthetic label alignment
- Class-specific analysis for imbalanced tasks
- Marginal point classification
- Practical interpretation and decision-making

**Practical Application** (Sections 6-8):
- Quality assessment tiers (excellent to catastrophic)
- How to choose hyperparameters (n_estimators)
- Common pitfalls and troubleshooting
- Comparison with alternative methods

**Deep Dives** (Section 9 - Appendices):
- Mathematical derivations (gradient boosting, confidence intervals)
- Code walkthrough with line numbers
- Glossary of technical terms

### 1.7 Prerequisites

**Required Knowledge**:
- Basic machine learning: classification, training/test splits
- Decision trees: nodes, leaves, splits
- Basic statistics: mean, standard deviation, confidence intervals

**Helpful but Not Required**:
- Gradient boosting algorithms (LightGBM, XGBoost)
- Log-odds and logistic regression
- t-distribution vs normal distribution

**No Prior Knowledge Needed**:
- Leaf alignment methodology (we explain from scratch)
- Data Shapley or other data valuation methods
- Synthetic data generation techniques

### 1.8 Notation & Terminology

Throughout this guide, we use the following notation:

**Data**:
- `X_synthetic`: Synthetic training data features
- `y_synthetic`: Synthetic training data labels
- `X_real_test`: Real test data features
- `y_real_test`: Real test data labels (ground truth)
- `n_synthetic`: Number of synthetic training points
- `n_real_test`: Number of real test points

**Model**:
- `model`: Trained LightGBM classifier
- `n_estimators` or `n_trees`: Number of trees in ensemble (default: 500)
- `leaf_value`: LightGBM's prediction contribution for a leaf (log-odds)
- `predicted_class`: 0 or 1, derived from leaf_value

**Leaf Alignment**:
- `leaf_id`: Unique identifier for a leaf node in a tree
- `leaf_utility`: How well a leaf classifies real data (range: -0.5 to +0.5)
- `utility_score`: Average utility for a synthetic point across all trees
- `CI_lower`, `CI_upper`: 95% confidence interval bounds
- `co-occurrence`: When synthetic and real points land in the same leaf

**Example Naming**:
- `Synth#0`, `Synth#1`, etc.: Individual synthetic training points
- `Real#A`, `Real#B`, etc.: Individual real test points
- `LEAF 0`, `LEAF 1`, etc.: Leaf nodes in a decision tree

---

**Ready to dive in?** The next section walks through the methodology step-by-step with concrete examples.

---

## 2. The Methodology: Step-by-Step

This section provides a detailed walkthrough of the leaf alignment algorithm. We'll build intuition through a concrete example using hospital readmission prediction.

### 2.1 High-Level Overview

The leaf alignment method consists of six main steps:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Train Model on Synthetic Data                      │
│   Input: X_synthetic, y_synthetic                          │
│   Output: Trained LightGBM model (500 trees)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Pass Both Datasets Through Model                   │
│   Synthetic → model → leaf assignments [n_synthetic, 500]  │
│   Real test → model → leaf assignments [n_real, 500]       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Calculate Leaf Utility (for each leaf)             │
│   Question: Does this leaf correctly classify real data?   │
│   Formula: utility = accuracy_on_real_data - 0.5           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Assign Utility to Synthetic Points                 │
│   Distribute leaf utility to synthetic points in that leaf │
│   Weight by: (real_points_in_leaf / total_real_points)     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Aggregate Across All Trees                         │
│   Each point has 500 utility scores (one per tree)         │
│   Compute: mean, std, confidence intervals                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Classify Points                                    │
│   CI_upper < 0       → RELIABLY HARMFUL                    │
│   CI_lower > 0       → RELIABLY BENEFICIAL                 │
│   CI spans 0         → UNCERTAIN                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Step 1: Train Model on Synthetic Data

**What Happens**: Train a LightGBM classifier using only the synthetic training data.

**Code** (from `sdvaluation/leaf_alignment.py:303`):
```python
# Configure LightGBM
params = {
    'n_estimators': 500,        # Build 500 sequential trees
    'objective': 'binary',      # Binary classification
    'verbose': -1,              # Suppress output
    **lgbm_params                # Additional hyperparameters
}

# Train on synthetic data
model = LGBMClassifier(**params)
model.fit(X_synthetic, y_synthetic)
```

**What the Model Learns**:
- Decision boundaries based on synthetic data patterns
- Each tree makes sequential corrections (boosting)
- Each tree has "leaves" (terminal nodes) with prediction values
- The model has NO knowledge of real test data at this stage

**Key Point**: The model optimizes for synthetic training performance. We'll later check if these learned boundaries generalize to real data.

### 2.3 Step 2: Pass Both Datasets Through the Model

**What Happens**: Run both synthetic training data AND real test data through the trained model to get leaf assignments.

**Code** (from `sdvaluation/leaf_alignment.py:178`):
```python
# Get leaf assignments for synthetic data
synthetic_leaves = model.predict(X_synthetic, pred_leaf=True)
# Shape: [n_synthetic, n_trees]

# Get leaf assignments for real test data
real_leaves = model.predict(X_real_test, pred_leaf=True)
# Shape: [n_real_test, n_trees]
```

**Understanding Leaf Assignments**:

Each data point flows through each tree and lands in a specific leaf:

```
Example for Tree #0:

Synth#0: features=[AGE=70, DIAGNOSIS=Diabetes, NUM_MEDS=3]
         → flows through tree → lands in LEAF 2

Synth#1: features=[AGE=55, DIAGNOSIS=Heart, NUM_MEDS=2]
         → flows through tree → lands in LEAF 0

Real#A:  features=[AGE=71, DIAGNOSIS=Diabetes, NUM_MEDS=5]
         → flows through tree → lands in LEAF 2  ← Co-occurs with Synth#0!

Real#B:  features=[AGE=58, DIAGNOSIS=Heart, NUM_MEDS=1]
         → flows through tree → lands in LEAF 0  ← Co-occurs with Synth#1!
```

**The "Co-Occurrence" Concept**:

When a synthetic point and real point land in the same leaf, they **co-occur**. This means:
- They have similar features (followed the same decision path)
- The model makes the same prediction for both
- We can check if that prediction works for the real point

**Output Data Structure**:
```python
synthetic_leaves = [
    [2, 5, 1, 3, ...],  # Synth#0: leaf 2 (tree0), leaf 5 (tree1), etc.
    [0, 3, 0, 1, ...],  # Synth#1: leaf 0 (tree0), leaf 3 (tree1), etc.
    ...
]

real_leaves = [
    [2, 4, 1, 3, ...],  # Real#A: leaf 2 (tree0), leaf 4 (tree1), etc.
    [0, 3, 2, 1, ...],  # Real#B: leaf 0 (tree0), leaf 3 (tree1), etc.
    ...
]
```

### 2.4 Step 3: Calculate Leaf Utility

**What Happens**: For each leaf in each tree, measure how well it classifies real test data.

**The Formula** (from `sdvaluation/leaf_alignment.py:42`):
```python
def calculate_leaf_utility(y_true: np.ndarray, leaf_value: float) -> float:
    """
    Calculate utility of a leaf based on real test data alignment.

    Args:
        y_true: Ground truth labels of real points in this leaf
        leaf_value: LightGBM's prediction value for this leaf

    Returns:
        utility: Range [-0.5, +0.5]
    """
    # 1. Determine what class this leaf predicts
    predicted_class = 1 if leaf_value > 0 else 0

    # 2. Check accuracy on real points in this leaf
    accuracy = np.mean(y_true == predicted_class)

    # 3. Convert to utility (-0.5 to +0.5 range)
    utility = accuracy - 0.5

    return utility
```

**Understanding the Utility Range**:

```
utility = +0.5:  Perfect leaf (100% accuracy on real data)
                 All real points correctly classified

utility = 0.0:   Random leaf (50% accuracy)
                 No better than coin flip

utility = -0.5:  Terrible leaf (0% accuracy on real data)
                 All real points misclassified
```

**Example Calculation**:

```
LEAF 0 in Tree #0:
  leaf_value = -0.8  (negative → predicts Class 0)

  Real points in LEAF 0:
    Real#B: label = 0 (no readmission)
    Real#D: label = 0 (no readmission)
    Real#H: label = 0 (no readmission)

  Calculation:
    predicted_class = 0  (since -0.8 < 0)
    y_true = [0, 0, 0]
    accuracy = mean([0, 0, 0] == 0) = 3/3 = 1.00
    utility = 1.00 - 0.5 = +0.5  ✓ Perfect alignment!
```

**Why "accuracy - 0.5"?**

This centers the utility around zero:
- Positive utility = better than random guessing = beneficial
- Negative utility = worse than random = harmful
- Zero utility = random performance = neutral

### 2.5 Step 4: Assign Utility to Synthetic Points

**What Happens**: Distribute each leaf's utility to the synthetic points that landed in that leaf.

**The Process** (from `sdvaluation/leaf_alignment.py:122`):

```python
# 1. Calculate raw leaf utility
leaf_utility = calculate_leaf_utility(y_true_in_leaf, leaf_value)

# 2. Weight by importance (how much real data is in this leaf)
weight = len(real_points_in_leaf) / n_total_real_test
weighted_utility = leaf_utility * weight

# 3. Distribute equally among synthetic points in this leaf
n_synth_in_leaf = len(synthetic_points_in_leaf)
score_per_point = weighted_utility / n_synth_in_leaf

# 4. Assign to each synthetic point
for synth_idx in synthetic_points_in_leaf:
    utility_scores[synth_idx] += score_per_point
```

**Why Weight by Importance?**

Leaves with more real data should have more influence on the final score:

```
LEAF 0: 100 real points, perfect accuracy → High impact
LEAF 7: 2 real points, perfect accuracy   → Low impact
```

The weighting ensures we prioritize getting major decision regions correct.

**Example Calculation**:

```
LEAF 0 in Tree #0:
  Synthetic points: Synth#1, Synth#3, Synth#8  (3 points)
  Real points: Real#B, Real#D, Real#H  (3 points)
  Total real points in dataset: 8

  Leaf utility: +0.5 (from Step 3)

  Weight: 3 real points / 8 total = 0.375
  Weighted utility: 0.5 × 0.375 = 0.1875

  Score per synthetic point: 0.1875 / 3 = 0.0625

  Result:
    Synth#1 gets +0.0625
    Synth#3 gets +0.0625
    Synth#8 gets +0.0625
```

**Key Insight**: If a synthetic point helped create a leaf that misclassifies real data, it receives **negative** utility. This identifies hallucinated points.

### 2.6 Step 5: Aggregate Across All Trees

**What Happens**: Each tree provides an independent utility score for each synthetic point. We aggregate these 500 scores to get statistics.

**Data Structure**:
```python
# After processing all trees
utility_per_tree.shape = [n_synthetic, n_trees]

# Example for Synth#5:
utility_per_tree[5] = [
    -0.0625,  # Tree 0 score
    -0.0301,  # Tree 1 score
    +0.0120,  # Tree 2 score (positive in this tree!)
    -0.0450,  # Tree 3 score
    ...
    -0.0280   # Tree 499 score
]
```

**Statistical Aggregation** (from `sdvaluation/leaf_alignment.py:261`):
```python
# Compute statistics across trees (axis=1)
mean_utility = np.mean(utility_per_tree, axis=1)
std_utility = np.std(utility_per_tree, axis=1, ddof=1)
se_utility = std_utility / np.sqrt(n_trees)

# Compute 95% confidence intervals using t-distribution
from scipy import stats
t_critical = stats.t.ppf(0.975, n_trees - 1)  # Two-tailed, df=499
ci_lower = mean_utility - t_critical * se_utility
ci_upper = mean_utility + t_critical * se_utility
```

**Why Multiple Trees Give Confidence**:

The standard error decreases with √n_trees:

```
SE = σ / √n_trees

100 trees  → SE = σ / 10     → Wider confidence intervals
500 trees  → SE = σ / 22.4   → 55% narrower
1000 trees → SE = σ / 31.6   → 68% narrower
```

**Example for Synth#5**:
```
Scores across 500 trees: [-0.0625, -0.0301, +0.0120, -0.0450, ...]

mean = -0.0234
std  = 0.0189
se   = 0.0189 / √500 = 0.000845

95% CI (t_critical ≈ 1.965):
  ci_lower = -0.0234 - 1.965 × 0.000845 = -0.0251
  ci_upper = -0.0234 + 1.965 × 0.000845 = -0.0217
```

### 2.7 Step 6: Classify Points

**What Happens**: Use confidence intervals to classify each synthetic point into three categories.

**Classification Rules**:
```python
if ci_upper < 0:
    classification = "RELIABLY HARMFUL"
elif ci_lower > 0:
    classification = "RELIABLY BENEFICIAL"
else:  # CI spans 0
    classification = "UNCERTAIN"
```

**Visual Understanding**:

**Case 1: Reliably Harmful** ✗
```
Synth#5:  mean = -0.0234,  CI = [-0.0251, -0.0217]

  ──────┼──────┼──────┼──────┼──────
      -0.03  -0.02  -0.01   0.00   +0.01
               │←─CI─→│
                  ↑
          Entire CI is LEFT of zero
          ci_upper = -0.0217 < 0  ✓

Classification: RELIABLY HARMFUL
Interpretation: 95% confident this point creates bad boundaries
```

**Case 2: Reliably Beneficial** ✓
```
Synth#1:  mean = +0.0612,  CI = [+0.0605, +0.0619]

  ──────┼──────┼──────┼──────┼──────
      -0.01   0.00  +0.01  +0.02  +0.03
                           │←CI→│
                              ↑
                    Entire CI is RIGHT of zero
                    ci_lower = +0.0605 > 0  ✓

Classification: RELIABLY BENEFICIAL
Interpretation: 95% confident this point creates good boundaries
```

**Case 3: Uncertain** ⚠️
```
Synth#99:  mean = -0.0012,  CI = [-0.0032, +0.0008]

  ──────┼──────┼──────┼──────┼──────
     -0.004 -0.002  0.00  +0.002 +0.004
               │←──CI──→│
                   ↑
              CI CROSSES zero

Classification: UNCERTAIN
Interpretation: Mixed evidence - some trees say beneficial, others harmful
```

**Final Output Example**:
```
Statistical Confidence (95% CI-based):
  Reliably harmful:      9,339 (93.39%)  ← Hallucinated data
  Reliably beneficial:      54 (0.54%)   ← Good data
  Uncertain:               607 (6.07%)   ← Inconsistent
```

### 2.8 Summary: The Complete Pipeline

Putting it all together:

```python
# Step 1: Train model
model = LGBMClassifier(n_estimators=500)
model.fit(X_synthetic, y_synthetic)

# Step 2: Get leaf assignments
synthetic_leaves = model.predict(X_synthetic, pred_leaf=True)
real_leaves = model.predict(X_real_test, pred_leaf=True)

# Step 3-4: Process each tree
utility_per_tree = []
for tree_id in range(500):
    tree_scores = process_single_tree(
        tree_id,
        synthetic_leaves[:, tree_id],
        real_leaves[:, tree_id],
        y_real_test
    )
    utility_per_tree.append(tree_scores)

utility_per_tree = np.column_stack(utility_per_tree)

# Step 5: Aggregate
mean = np.mean(utility_per_tree, axis=1)
se = np.std(utility_per_tree, axis=1) / np.sqrt(500)
ci_lower = mean - 1.965 * se
ci_upper = mean + 1.965 * se

# Step 6: Classify
reliably_harmful = ci_upper < 0
reliably_beneficial = ci_lower > 0
uncertain = ~reliably_harmful & ~reliably_beneficial
```

**What's Next?**

Section 3 dives deep into the core concepts:
- What exactly is a decision tree and leaf?
- How does LightGBM compute leaf_value?
- Why use t-distribution for confidence intervals?
- What are empty leaves and why do they matter?

