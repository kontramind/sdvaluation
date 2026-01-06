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

---

## 3. Core Concepts

This section provides deep dives into the fundamental concepts underlying leaf alignment. We'll explain both what LightGBM does internally and what our algorithm adds on top.

### 3.1 Decision Trees & Leaves

#### What is a Decision Tree?

A decision tree is a hierarchical structure that makes predictions by asking a series of yes/no questions about features.

**Visual Example: One Tree from the 500-Tree Ensemble**

```
                          [Root: Internal Node]
                        "Start here with all data"
                                  |
                     ┌────────────┴────────────┐
                     │ Split: AGE <= 65?       │
                     └────────────┬────────────┘
                                  |
                ┌─────────────────┴─────────────────┐
               YES                                  NO
          (AGE <= 65)                          (AGE > 65)
                |                                   |
                ▼                                   ▼
        [Internal Node]                      [Internal Node]
    ┌───────────────────────┐            ┌──────────────────────┐
    │ Split: DIAGNOSIS      │            │ Split: NUM_MEDS <= 5?│
    │       = "Heart"?      │            └──────────┬───────────┘
    └───────────┬───────────┘                       |
                |                    ┌──────────────┴──────────────┐
        ┌───────┴───────┐          YES                            NO
       YES             NO      (NUM_MEDS <= 5)              (NUM_MEDS > 5)
   (is Heart)     (not Heart)       |                            |
        |               |            ▼                            ▼
        ▼               ▼      ╔═══════════╗                ╔═══════════╗
  ╔═══════════╗   ╔═══════════╗║ LEAF 2    ║                ║ LEAF 3    ║
  ║ LEAF 0    ║   ║ LEAF 1    ║║           ║                ║           ║
  ║           ║   ║           ║║leaf_index:║                ║leaf_index:║
  ║leaf_index:║   ║leaf_index:║║    2      ║                ║    3      ║
  ║    0      ║   ║    1      ║║           ║                ║           ║
  ║           ║   ║           ║║leaf_value:║                ║leaf_value:║
  ║leaf_value:║   ║leaf_value:║║  +0.6     ║                ║  -0.4     ║
  ║  -0.8     ║   ║  +0.3     ║║           ║                ║           ║
  ║           ║   ║           ║║Predicts:  ║                ║Predicts:  ║
  ║Predicts:  ║   ║Predicts:  ║║ Class 1   ║                ║ Class 0   ║
  ║ Class 0   ║   ║ Class 1   ║║(readmit)  ║                ║(no readmit)║
  ╚═══════════╝   ╚═══════════╝╚═══════════╝                ╚═══════════╝
```

**Components Explained**:

1. **Internal Nodes** (rectangles): Ask questions and split data
   - Example: "AGE <= 65?" sends data left (YES) or right (NO)

2. **Leaf Nodes** (double-lined boxes): Terminal nodes that make predictions
   - `leaf_index`: Unique identifier (0, 1, 2, 3...)
   - `leaf_value`: Prediction score from LightGBM training
   - `Predicts`: Derived from leaf_value (positive → Class 1, negative → Class 0)

#### What LightGBM Stores

The actual JSON structure from `model.booster_.dump_model()`:

```python
{
  "tree_structure": {
    "split_feature": "AGE",
    "threshold": 65.0,
    "left_child": {
      "split_feature": "DIAGNOSIS",
      "threshold": "Heart",
      "left_child": {
        "leaf_index": 0,
        "leaf_value": -0.8    # ← Computed during training
      },
      "right_child": {
        "leaf_index": 1,
        "leaf_value": 0.3
      }
    },
    "right_child": {
      "split_feature": "NUM_MEDS",
      "threshold": 5.0,
      "left_child": {
        "leaf_index": 2,
        "leaf_value": 0.6
      },
      "right_child": {
        "leaf_index": 3,
        "leaf_value": -0.4
      }
    }
  }
}
```

### 3.2 Leaf Value Computation (LightGBM Internals)

**Critical Separation**: `leaf_value` is computed by **LightGBM during training** - NOT by our leaf alignment algorithm. We simply read these values.

#### How LightGBM Computes leaf_value

**The Formula** (from gradient boosting theory):

```
leaf_value = - Σ(gradients) / (Σ(hessians) + λ)

Where:
  gradients = first derivative of loss (how wrong current predictions are)
  hessians  = second derivative of loss (confidence in the gradient)
  λ         = regularization parameter (prevents overfitting)
```

#### Step-by-Step Example: Computing LEAF 0's Value

**Setup**: Training on Synthetic Data

Synthetic points that land in LEAF 0 (from tree structure):
- Path: AGE ≤ 65? → YES → DIAGNOSIS = Heart? → YES → LEAF 0

```
┌─────────┬─────┬───────────┬──────────┬────────────────────┐
│ Index   │ AGE │ DIAGNOSIS │ NUM_MEDS │ IS_READMISSION_30D │
├─────────┼─────┼───────────┼──────────┼────────────────────┤
│ Synth#1 │ 55  │ Heart     │ 2        │ 0 (no readmit)    │
│ Synth#3 │ 60  │ Heart     │ 1        │ 0 (no readmit)    │
│ Synth#8 │ 50  │ Heart     │ 3        │ 0 (no readmit)    │
└─────────┴─────┴───────────┴──────────┴────────────────────┘
```

**Step 1**: Initial Predictions (before this tree)

Assume this is Tree #3 (Trees 0-2 already trained). Current ensemble prediction:

```python
# Sum of previous trees' predictions
Synth#1: raw_score = -0.2  → probability = sigmoid(-0.2) = 0.45
Synth#3: raw_score = -0.1  → probability = sigmoid(-0.1) = 0.48
Synth#8: raw_score = -0.3  → probability = sigmoid(-0.3) = 0.43

# Sigmoid function: σ(x) = 1 / (1 + e^(-x))
```

**Step 2**: Compute Gradients and Hessians

For binary log loss:

```python
gradient_i = prediction_i - true_label_i
hessian_i = prediction_i × (1 - prediction_i)
```

For each point in LEAF 0:

```
Synth#1:
  prediction = 0.45
  true_label = 0
  gradient = 0.45 - 0 = +0.45
  hessian = 0.45 × (1 - 0.45) = 0.45 × 0.55 = 0.2475

Synth#3:
  prediction = 0.48
  true_label = 0
  gradient = 0.48 - 0 = +0.48
  hessian = 0.48 × (1 - 0.48) = 0.48 × 0.52 = 0.2496

Synth#8:
  prediction = 0.43
  true_label = 0
  gradient = 0.43 - 0 = +0.43
  hessian = 0.43 × (1 - 0.43) = 0.43 × 0.57 = 0.2451
```

**Step 3**: Sum Over All Points in Leaf

```
Σ(gradients) = 0.45 + 0.48 + 0.43 = 1.36
Σ(hessians) = 0.2475 + 0.2496 + 0.2451 = 0.7422
```

**Step 4**: Apply Formula with Learning Rate

```python
# Raw leaf value
λ = 0.1  # L2 regularization
raw_leaf_value = - Σ(gradients) / (Σ(hessians) + λ)
               = - 1.36 / (0.7422 + 0.1)
               = - 1.36 / 0.8422
               = -1.615

# Apply learning rate shrinkage
learning_rate = 0.5
final_leaf_value = raw_leaf_value × learning_rate
                 = -1.615 × 0.5
                 = -0.8075 ≈ -0.8  ✓
```

**What This Means Intuitively**:

```
LEAF 0: leaf_value = -0.8

All 3 synthetic points in this leaf have:
  - True label = 0 (no readmission)
  - Current predictions ≈ 0.45 (predicting some chance of readmission)

Gradients are POSITIVE (predictions too high)
→ Need to DECREASE predictions
→ Negative leaf_value pushes predictions DOWN
→ "These patients should NOT readmit"
```

#### Full Computation Table

```
┌────────┬────────────┬──────────┬─────────────┬─────────────┬─────────────────┐
│ Leaf   │ Synth Pts  │ Labels   │ Σ(gradient) │ Σ(hessian)  │ leaf_value      │
│        │            │          │             │             │ (with lr=0.5)   │
├────────┼────────────┼──────────┼─────────────┼─────────────┼─────────────────┤
│ LEAF 0 │ #1,#3,#8   │ 0,0,0    │ +1.36       │ 0.7422      │ -1.615×0.5=-0.8│
│ LEAF 1 │ #5         │ 1        │ -0.52       │ 0.2184      │ +0.52×0.5=+0.3 │
│ LEAF 2 │ #0,#4,#7   │ 1,1,1    │ -1.25       │ 0.6891      │ +1.25×0.5=+0.6 │
│ LEAF 3 │ #2,#6,#9   │ 0,0,0    │ +0.85       │ 0.5124      │ -0.85×0.5=-0.4 │
└────────┴────────────┴──────────┴─────────────┴─────────────┴─────────────────┘

Pattern:
  Leaves with all label=0 → Positive gradients → Negative leaf_value
  Leaves with all label=1 → Negative gradients → Positive leaf_value
```

**Key Takeaway**:
- `leaf_value` is computed from **synthetic training data only**
- Based on which synthetic points land in that leaf
- It's a correction term optimized for synthetic labels
- **The critical question for leaf alignment**: Do these values also work for real data?

### 3.3 LightGBM vs Our External Algorithm

This is crucial to understand: there's a clear separation between what LightGBM does and what our algorithm does.

#### What LightGBM Does (Internal, During Training)

```python
# LightGBM training on synthetic data
model = LGBMClassifier(n_estimators=500)
model.fit(X_synthetic, y_synthetic)  # ← LightGBM computes all leaf_values here
```

**LightGBM internally**:
1. Builds 500 trees sequentially
2. For each tree, decides where to split
3. Computes `leaf_value` for each leaf using gradients/hessians
4. Stores everything in the trained model

**Output**: A trained model with fixed tree structure and leaf_values

#### What Our Leaf Alignment Algorithm Does (External, After Training)

```python
# Our algorithm - just READS the leaf_values
booster = model.booster_
tree_dump = booster.dump_model()  # ← Reading what LightGBM stored

# Extract leaf_value (already computed by LightGBM)
leaf_value = get_leaf_value_from_tree(tree_dump["tree_structure"], leaf_id)

# Then USE that value to check alignment
predicted_class = 1 if leaf_value > 0 else 0  # ← Our interpretation
accuracy = np.mean(y_true_real_data == predicted_class)  # ← Our calculation
utility = accuracy - 0.5  # ← Our metric
```

**Our algorithm**:
1. Takes the already-trained model as input
2. **Reads** the `leaf_value` (doesn't compute it)
3. Uses `leaf_value` to determine what class the leaf predicts
4. Checks if that prediction matches **real test data** labels
5. Scores synthetic points based on alignment

#### Visual Separation

```
┌─────────────────────────────────────────────────────────────┐
│  LightGBM Training (Black Box)                              │
│  ───────────────────────────────                            │
│  Input: X_synthetic, y_synthetic                            │
│                                                              │
│  Internal Process:                                          │
│    • Compute gradients/hessians                             │
│    • Find best splits                                       │
│    • Compute leaf_value = -Σgrad/(Σhess+λ)  ← LightGBM     │
│    • Apply learning rate                                    │
│                                                              │
│  Output: Trained model with leaf_values baked in            │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    (model is ready)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Our Leaf Alignment Algorithm                               │
│  ────────────────────────────────                           │
│  Input: trained_model, X_synthetic, X_real_test, y_real    │
│                                                              │
│  Our Process:                                               │
│    • Get leaf assignments (pred_leaf=True)                  │
│    • READ leaf_value from model dump  ← Just reading!      │
│    • Interpret: leaf_value > 0 → predicts Class 1          │
│    • Check: does prediction match y_real? ← Our check!     │
│    • Compute utility = accuracy - 0.5  ← Our calculation   │
│    • Assign scores to synthetic points                      │
│                                                              │
│  Output: Utility scores for each synthetic point            │
└─────────────────────────────────────────────────────────────┘
```

#### Code Evidence

From `sdvaluation/leaf_alignment.py:117`:

```python
# Get leaf value (prediction contribution) from tree structure
leaf_value = get_leaf_value_from_tree(tree_dump["tree_structure"], leaf_id)
#            ↑ Just READING, not computing!
```

And the function at line 51:

```python
def get_leaf_value_from_tree(tree_node: dict, target_leaf_id: int) -> float:
    """
    Recursively find the leaf value for a given leaf_id in tree structure.
    """
    if "leaf_value" in tree_node:
        if tree_node.get("leaf_index") == target_leaf_id:
            return tree_node["leaf_value"]  # ← Just returning stored value
```

#### Summary Table

| Component | Who Computes It | When | Purpose |
|-----------|----------------|------|---------|
| `leaf_value` | LightGBM | During `.fit()` | Optimize predictions on synthetic training data |
| `leaf utility` | Our algorithm | After training | Measure how well leaf works on real test data |
| `utility score` | Our algorithm | After training | Score synthetic points based on alignment |

**The key insight**: LightGBM optimized `leaf_value` for synthetic data. We check if those same values work well for real data. If they don't → synthetic data is hallucinated!

### 3.4 Empty Leaf Penalty

#### What is an "Empty Leaf"?

An empty leaf is a leaf that contains:
- ✅ Synthetic training points (landed there during training)
- ❌ ZERO real test points (no real data lands there)

```
LEAF 7:
  ├─ Synthetic points: Synth#42, Synth#87, Synth#99
  └─ Real test points: (none)  ← EMPTY!
```

#### Why Does This Happen?

**Scenario: Hallucinated Feature Combinations**

The synthetic data created combinations of features that don't exist in reality.

**Example**:

```
Synthetic data has:
  Synth#42: AGE=25, NUM_MEDS=20, DIAGNOSIS=Heart, Label=1
  Synth#87: AGE=28, NUM_MEDS=18, DIAGNOSIS=Heart, Label=1
  Synth#99: AGE=22, NUM_MEDS=22, DIAGNOSIS=Heart, Label=1

Real data: No patients with AGE<30 AND NUM_MEDS>15
           (This combination doesn't happen in reality)

What happens:
  1. LightGBM creates a split: AGE <= 30 AND NUM_MEDS > 15
  2. Creates LEAF 7 for this region
  3. Synthetic points land there during training
  4. But NO real test patients have these features!
  5. → LEAF 7 is empty of real data
```

**Analogy**: Training a self-driving car on synthetic data that includes "flying cars" as a category. The model learns rules for flying cars, but real test data has zero flying cars. That decision boundary is useless!

#### Why is This a RED FLAG?

```
Empty leaf = Synthetic data created a decision boundary
             in a region where real data doesn't exist

           = The model learned to make predictions
             for impossible/non-existent cases

           = Wasted model capacity

           = Hallucinated synthetic data
```

#### The Empty Leaf Penalty

From `sdvaluation/leaf_alignment.py:136`:

```python
# Handle empty leaves (synthetic points in leaves with NO real data)
synth_unique_leaves = np.unique(synthetic_leaves_k)
empty_leaves = np.setdiff1d(synth_unique_leaves, unique_leaves)

for leaf_id in empty_leaves:
    synth_mask = synthetic_leaves_k == leaf_id
    synth_indices = np.where(synth_mask)[0]

    if len(synth_indices) > 0:
        # Penalize: these synthetic points created regions with no real data
        utility_scores[synth_indices] += empty_leaf_penalty / len(synth_indices)

# Default: empty_leaf_penalty = -1.0
```

#### Example Calculation

**Setup: Tree with Empty Leaf**

```
                    [Root: AGE <= 30?]
                    /                 \
                  YES                 NO
                   |                   |
        [NUM_MEDS > 15?]          [LEAF 2]
           /        \
         YES        NO
          |          |
      LEAF 7     LEAF 1
    (EMPTY!)
```

**Synthetic Points Distribution**:
```
Synth#42: AGE=25, NUM_MEDS=20 → LEAF 7 (empty)
Synth#87: AGE=28, NUM_MEDS=18 → LEAF 7 (empty)
Synth#99: AGE=22, NUM_MEDS=22 → LEAF 7 (empty)
Synth#10: AGE=27, NUM_MEDS=10 → LEAF 1
Synth#50: AGE=70, NUM_MEDS=5  → LEAF 2
```

**Real Test Points Distribution**:
```
Real#A: AGE=72, NUM_MEDS=3  → LEAF 2 ✓
Real#B: AGE=68, NUM_MEDS=7  → LEAF 2 ✓
Real#C: AGE=25, NUM_MEDS=5  → LEAF 1 ✓
Real#D: AGE=28, NUM_MEDS=8  → LEAF 1 ✓

Notice: NO real points go to LEAF 7!
```

**Penalty Calculation**:

```python
# Identify leaves with real data
unique_leaves = [1, 2]  # Leaves with real test points

# Identify leaves with synthetic data
synth_unique_leaves = [1, 2, 7]  # Synthetic points' leaves

# Find empty leaves (synthetic but no real)
empty_leaves = [7]  # ← LEAF 7 is empty!

# Apply penalty
leaf_id = 7
synth_in_leaf_7 = [42, 87, 99]  # 3 synthetic points
empty_leaf_penalty = -1.0

penalty_per_point = -1.0 / 3 = -0.333

Synth#42 gets -0.333
Synth#87 gets -0.333
Synth#99 gets -0.333
```

#### Why -1.0 Specifically?

**Comparison to Regular Utility Range**:

```
Regular leaf utility range: -0.5 to +0.5 (based on accuracy - 0.5)

But we weight by importance:
  weighted_utility = utility × (n_real_in_leaf / n_total_real)

Maximum possible weighted utility ≈ 0.5 × 1.0 = 0.5
  (if ALL real data in one leaf with perfect accuracy)

Typical weighted utility ≈ 0.5 × 0.1 = 0.05
  (10% of real data in leaf with perfect accuracy)

Empty leaf penalty = -1.0 is chosen to be:
  • Stronger than typical negative utility (which might be -0.05)
  • Clear signal that this is a special bad case
  • But not infinitely negative (still allows other trees to compensate)
```

#### When Empty Leaves Are Actually OK

Empty leaves aren't always bad! Acceptable scenarios:

1. **Small test set**: Legitimate rare cases might not appear
2. **Intentional augmentation**: Adding rare edge cases on purpose
3. **Different evaluation goal**: Testing robustness to outliers

**How to adjust**: Use `--empty-leaf-penalty` parameter:
```bash
# More lenient (for small test sets)
--empty-leaf-penalty -0.1

# Default (standard)
--empty-leaf-penalty -1.0

# More strict (for production)
--empty-leaf-penalty -2.0
```

### 3.5 Aggregation Across Trees: Statistical Foundations

#### Why Multiple Trees?

Each tree provides an **independent measurement** of each synthetic point's utility. More trees = better statistical confidence.

#### The Data Structure

After processing all 500 trees:

```python
utility_per_tree.shape = [n_synthetic, n_trees]

# Each row: one point's scores across all trees
# Each column: all points' scores from one tree

# Example for Synth#5 (row 5):
utility_per_tree[5, :] = [
    -0.0625,  # Tree 0
    -0.0301,  # Tree 1
    +0.0120,  # Tree 2 (positive!)
    -0.0450,  # Tree 3
    ...
    -0.0280   # Tree 499
]
```

#### From Scores to Statistics

```python
# Compute along axis=1 (across trees, for each point)
mean_utility = np.mean(utility_per_tree, axis=1)  # [n_synthetic]
std_utility = np.std(utility_per_tree, axis=1, ddof=1)  # [n_synthetic]
se_utility = std_utility / np.sqrt(n_trees)  # Standard error
```

**Why `axis=1`?**

```python
utility_per_tree.shape = [n_synthetic, n_trees]
                          ↑             ↑
                        axis 0      axis 1

# axis=1 means: compute across columns (trees)
# Result: one value per row (per synthetic point)
```

**Example for Synth#5**:

```python
# Scores across 500 trees
scores = [-0.0625, -0.0301, +0.0120, -0.0450, ..., -0.0280]

mean = -0.0234  # Average across all trees
std  = 0.0189   # Standard deviation (measures variability)
se   = 0.0189 / √500 = 0.000845  # Standard error (uncertainty in mean)
```

#### Why Standard Error Decreases with √n

**The Statistical Principle**:

```
SE = σ / √n

Where:
  σ = population standard deviation (unknown, estimate with sample std)
  n = number of independent samples (trees)
```

**Practical Impact**:

```
100 trees:   SE = σ / √100  = σ / 10     → Wider CIs
500 trees:   SE = σ / √500  = σ / 22.4   → 55% narrower
1000 trees:  SE = σ / √1000 = σ / 31.6   → 68% narrower
```

**Why this matters**:

More trees → Smaller SE → Narrower confidence intervals → Fewer "uncertain" classifications

**Example**:

```
With 100 trees (SE = 0.00189):
  Point #42: mean = -0.002, CI = [-0.006, +0.002]  ← Spans 0, "Uncertain"

With 500 trees (SE = 0.000845):
  Point #42: mean = -0.002, CI = [-0.004, -0.0004]  ← CI_upper < 0, "Harmful"!
```

### 3.6 Confidence Intervals & t-Distribution

#### Why Use Confidence Intervals?

**The Problem with Using Mean Alone**:

Two points with same mean but different certainty:

```
Synth#A:
  mean = -0.010
  SE   = 0.002
  CI   = [-0.014, -0.006]  ← Tight CI, clearly negative

Synth#B:
  mean = -0.010
  SE   = 0.012
  CI   = [-0.034, +0.014]  ← Wide CI, uncertain!
```

If we only looked at mean:
- Both have mean = -0.010 → both seem harmful
- We'd treat them the same ❌

With CI classification:
- Synth#A: `CI_upper = -0.006 < 0` → **RELIABLY HARMFUL** ✗
- Synth#B: CI spans 0 → **UNCERTAIN** ⚠️

**Why the difference?**

- Synth#A: Consistently negative across all 500 trees → High confidence
- Synth#B: Mix of positive and negative → Could go either way

#### Why t-Distribution Instead of Normal?

**Common Misconception**: "We use t-distribution because the data is skewed due to class imbalance"

❌ **WRONG REASON**

✓ **CORRECT REASON**: We use t-distribution because we **estimate** the standard deviation from our sample (500 trees) rather than knowing the true population standard deviation.

**The Statistics**:

With **known** σ (population standard deviation):
```
CI = mean ± z_critical × (σ / √n)
     Use normal distribution (z_critical = 1.96 for 95%)
```

With **estimated** σ (sample standard deviation s):
```
CI = mean ± t_critical × (s / √n)
     Use t-distribution (t_critical depends on degrees of freedom)
```

**Why This Matters**:

The t-distribution has **heavier tails** than the normal distribution to account for the uncertainty in estimating σ from the sample.

**Visual Comparison**:

```
Small sample (n=10):
  ┌────────────────────────────────────┐
  │    t-dist (wider tails)            │
  │      ╱‾‾╲                          │
  │     ╱    ╲    normal (narrower)    │
  │    ╱  ╱‾╲ ╲   /‾╲                 │
  │   ╱  ╱   ╲ ╲ /   ╲                │
  │  ╱  ╱     ╲ V     ╲               │
  ├─┴──┴───────┴───────┴──────────────┤
            mean

  t_critical = 2.262 > z_critical = 1.96
  (Need wider CI due to uncertainty in s)

Large sample (n=500):
  ┌────────────────────────────────────┐
  │  t-dist and normal almost overlap  │
  │         ╱‾‾╲                       │
  │        ╱    ╲                      │
  │       ╱      ╲                     │
  │      ╱        ╲                    │
  ├─────┴──────────┴───────────────────┤
            mean

  t_critical = 1.965 ≈ z_critical = 1.96
  (s is reliable with 500 samples)
```

#### The Formula

From `sdvaluation/leaf_alignment.py:261`:

```python
from scipy import stats

# Compute t-critical value
# For 95% confidence, two-tailed test, df = n_trees - 1
t_critical = stats.t.ppf(0.975, n_trees - 1)
#                        ↑ (1 + 0.95) / 2 = 0.975
#                                 ↑ degrees of freedom = 499

# Compute confidence intervals
ci_lower = mean_utility - t_critical * se_utility
ci_upper = mean_utility + t_critical * se_utility
```

**With 500 trees**:

```python
t_critical ≈ 1.965

ci_lower = mean - 1.965 × se
ci_upper = mean + 1.965 × se
```

The interval extends ~2 standard errors in each direction from the mean.

#### Three-Way Classification

**The Rules**:

```python
if ci_upper < 0:
    classification = "RELIABLY HARMFUL"
    # 95% confident true utility is negative
    # Even the most optimistic estimate is still harmful

elif ci_lower > 0:
    classification = "RELIABLY BENEFICIAL"
    # 95% confident true utility is positive
    # Even the most pessimistic estimate is still beneficial

else:  # CI spans 0
    classification = "UNCERTAIN"
    # Not enough evidence
    # True utility could be positive OR negative
```

**Detailed Example**:

```
Synth#42 (Reliably Harmful):
  Scores: [-0.045, -0.039, -0.042, -0.041, ...]
  mean = -0.0401
  std  = 0.0087
  se   = 0.0087 / √500 = 0.000389

  CI = [-0.0401 - 1.965×0.000389, -0.0401 + 1.965×0.000389]
     = [-0.0409, -0.0393]

  ci_upper = -0.0393 < 0  ✓

  Classification: RELIABLY HARMFUL ✗

  Why? Mean is strongly negative, low variance (consistent),
       tight CI entirely below zero. All 500 trees agree!
```

**What's Next?**

Section 4 provides a comprehensive worked example, walking through the entire process with concrete numbers for a small dataset.

