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

---

## 4. Comprehensive Worked Example

This section walks through the **entire leaf alignment process** with a small, concrete dataset. We'll compute every number step-by-step so you can see exactly how the algorithm works.

### 4.1 The Setup

**Task**: Hospital readmission prediction (binary classification)
- **Target variable**: `IS_READMISSION_30D` (0 = no readmission, 1 = readmission)
- **Features**: AGE, DIAGNOSIS, NUM_MEDS

**Datasets**:
- **Synthetic training data**: 10 points
- **Real test data**: 8 points
- **Model**: LightGBM with `n_estimators=3` (we'll use just 3 trees for clarity)

### 4.2 The Data

#### Synthetic Training Data (10 points)

```
┌─────────┬─────┬───────────┬──────────┬──────────────────────┐
│ Index   │ AGE │ DIAGNOSIS │ NUM_MEDS │ IS_READMISSION_30D   │
├─────────┼─────┼───────────┼──────────┼──────────────────────┤
│ Synth#0 │ 70  │ Diabetes  │ 3        │ 1 (readmitted)      │
│ Synth#1 │ 55  │ Heart     │ 2        │ 0 (no readmit)      │
│ Synth#2 │ 80  │ Cancer    │ 8        │ 0 (no readmit)      │
│ Synth#3 │ 60  │ Heart     │ 1        │ 0 (no readmit)      │
│ Synth#4 │ 72  │ Heart     │ 4        │ 1 (readmitted)      │
│ Synth#5 │ 45  │ Diabetes  │ 2        │ 1 (readmitted)      │
│ Synth#6 │ 68  │ Cancer    │ 7        │ 0 (no readmit)      │
│ Synth#7 │ 75  │ Heart     │ 5        │ 1 (readmitted)      │
│ Synth#8 │ 50  │ Heart     │ 1        │ 0 (no readmit)      │
│ Synth#9 │ 82  │ Diabetes  │ 9        │ 0 (no readmit)      │
└─────────┴─────┴───────────┴──────────┴──────────────────────┘
```

#### Real Test Data (8 points)

```
┌─────────┬─────┬───────────┬──────────┬──────────────────────┐
│ Index   │ AGE │ DIAGNOSIS │ NUM_MEDS │ IS_READMISSION_30D   │
├─────────┼─────┼───────────┼──────────┼──────────────────────┤
│ Real#A  │ 71  │ Diabetes  │ 5        │ 1 (readmitted)      │
│ Real#B  │ 58  │ Heart     │ 1        │ 0 (no readmit)      │
│ Real#C  │ 77  │ Cancer    │ 9        │ 1 (readmitted)      │
│ Real#D  │ 62  │ Heart     │ 2        │ 0 (no readmit)      │
│ Real#E  │ 73  │ Diabetes  │ 4        │ 1 (readmitted)      │
│ Real#F  │ 48  │ Diabetes  │ 3        │ 0 (no readmit)      │
│ Real#G  │ 69  │ Cancer    │ 7        │ 1 (readmitted)      │
│ Real#H  │ 52  │ Heart     │ 1        │ 0 (no readmit)      │
└─────────┴─────┴───────────┴──────────┴──────────────────────┘
```

### 4.3 Tree #0: Structure and Leaf Values

After training on synthetic data, LightGBM produces this tree:

```
                    [Root: AGE <= 65?]
                    /                 \
                  YES                 NO
             (AGE <= 65)          (AGE > 65)
                   |                   |
        [DIAGNOSIS="Heart"?]    [NUM_MEDS <= 5?]
           /        \              /          \
         YES        NO           YES          NO
          |          |            |            |
      LEAF 0     LEAF 1       LEAF 2       LEAF 3
   value=-0.8  value=+0.3   value=+0.6   value=-0.4
   Class 0     Class 1      Class 1      Class 0
```

**Leaf Values** (computed by LightGBM from synthetic training data):
- LEAF 0: `leaf_value = -0.8` → predicts Class 0 (no readmit)
- LEAF 1: `leaf_value = +0.3` → predicts Class 1 (readmit)
- LEAF 2: `leaf_value = +0.6` → predicts Class 1 (readmit)
- LEAF 3: `leaf_value = -0.4` → predicts Class 0 (no readmit)

### 4.4 Step 1: Pass Synthetic Data Through Tree #0

Let's trace where each synthetic point lands:

```
Synth#0: AGE=70 → NO  → NUM_MEDS=3 ≤ 5 → YES  → LEAF 2
Synth#1: AGE=55 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
Synth#2: AGE=80 → NO  → NUM_MEDS=8 ≤ 5 → NO   → LEAF 3
Synth#3: AGE=60 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
Synth#4: AGE=72 → NO  → NUM_MEDS=4 ≤ 5 → YES  → LEAF 2
Synth#5: AGE=45 → YES → DIAGNOSIS=Diabetes → NO → LEAF 1
Synth#6: AGE=68 → NO  → NUM_MEDS=7 ≤ 5 → NO   → LEAF 3
Synth#7: AGE=75 → NO  → NUM_MEDS=5 ≤ 5 → YES  → LEAF 2
Synth#8: AGE=50 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
Synth#9: AGE=82 → NO  → NUM_MEDS=9 ≤ 5 → NO   → LEAF 3
```

**Synthetic Leaf Assignments** (Tree #0):
```python
synthetic_leaves_tree0 = [2, 0, 3, 0, 2, 1, 3, 2, 0, 3]
#                        ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                        #0 #1 #2 #3 #4 #5 #6 #7 #8 #9
```

### 4.5 Step 2: Pass Real Test Data Through Tree #0

```
Real#A: AGE=71 → NO  → NUM_MEDS=5 ≤ 5 → YES  → LEAF 2
Real#B: AGE=58 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
Real#C: AGE=77 → NO  → NUM_MEDS=9 ≤ 5 → NO   → LEAF 3
Real#D: AGE=62 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
Real#E: AGE=73 → NO  → NUM_MEDS=4 ≤ 5 → YES  → LEAF 2
Real#F: AGE=48 → YES → DIAGNOSIS=Diabetes → NO → LEAF 1
Real#G: AGE=69 → NO  → NUM_MEDS=7 ≤ 5 → NO   → LEAF 3
Real#H: AGE=52 → YES → DIAGNOSIS=Heart → YES  → LEAF 0
```

**Real Leaf Assignments** (Tree #0):
```python
real_leaves_tree0 = [2, 0, 3, 0, 2, 1, 3, 0]
#                   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                   A  B  C  D  E  F  G  H
```

### 4.6 Step 3: Co-Occurrence Summary

```
┌────────┬──────────────────┬─────────────────────────┬────────────┐
│ Leaf   │ Synthetic Points │ Real Points (labels)    │ Leaf Value │
├────────┼──────────────────┼─────────────────────────┼────────────┤
│ LEAF 0 │ #1, #3, #8       │ B(0), D(0), H(0)       │ -0.8       │
│ LEAF 1 │ #5               │ F(0)                    │ +0.3       │
│ LEAF 2 │ #0, #4, #7       │ A(1), E(1)             │ +0.6       │
│ LEAF 3 │ #2, #6, #9       │ C(1), G(1)             │ -0.4       │
└────────┴──────────────────┴─────────────────────────┴────────────┘
```

### 4.7 Step 4: Calculate Leaf Utility

#### LEAF 0 ✓✓✓

```python
# What does this leaf predict?
leaf_value = -0.8
predicted_class = 0  # (leaf_value < 0)

# Real points in this leaf
real_labels = [0, 0, 0]  # B, D, H (all no readmit)

# Check alignment
accuracy = np.mean([0, 0, 0] == 0) = 3/3 = 1.00

# Calculate utility
utility = 1.00 - 0.5 = +0.5  ✓ Perfect!

# Weight by importance
weight = 3 real points / 8 total = 0.375

# Weighted utility
weighted_utility = 0.5 × 0.375 = 0.1875
```

#### LEAF 1 ✗✗✗

```python
# What does this leaf predict?
leaf_value = +0.3
predicted_class = 1  # (leaf_value > 0)

# Real points in this leaf
real_labels = [0]  # F (no readmit)

# Check alignment
accuracy = np.mean([0] == 1) = 0/1 = 0.00

# Calculate utility
utility = 0.00 - 0.5 = -0.5  ✗ Completely wrong!

# Weight by importance
weight = 1 real point / 8 total = 0.125

# Weighted utility
weighted_utility = -0.5 × 0.125 = -0.0625
```

#### LEAF 2 ✓✓✓

```python
# What does this leaf predict?
leaf_value = +0.6
predicted_class = 1  # (leaf_value > 0)

# Real points in this leaf
real_labels = [1, 1]  # A, E (both readmitted)

# Check alignment
accuracy = np.mean([1, 1] == 1) = 2/2 = 1.00

# Calculate utility
utility = 1.00 - 0.5 = +0.5  ✓ Perfect!

# Weight by importance
weight = 2 real points / 8 total = 0.25

# Weighted utility
weighted_utility = 0.5 × 0.25 = 0.125
```

#### LEAF 3 ✗✗✗

```python
# What does this leaf predict?
leaf_value = -0.4
predicted_class = 0  # (leaf_value < 0)

# Real points in this leaf
real_labels = [1, 1]  # C, G (both readmitted)

# Check alignment
accuracy = np.mean([1, 1] == 0) = 0/2 = 0.00

# Calculate utility
utility = 0.00 - 0.5 = -0.5  ✗ Completely wrong!

# Weight by importance
weight = 2 real points / 8 total = 0.25

# Weighted utility
weighted_utility = -0.5 × 0.25 = -0.125
```

### 4.8 Step 5: Distribute Utility to Synthetic Points

```python
# LEAF 0: weighted_utility = +0.1875
# Distribute among: Synth#1, #3, #8 (3 points)
score_per_point = 0.1875 / 3 = 0.0625
Synth#1 gets +0.0625
Synth#3 gets +0.0625
Synth#8 gets +0.0625

# LEAF 1: weighted_utility = -0.0625
# Distribute among: Synth#5 (1 point)
score_per_point = -0.0625 / 1 = -0.0625
Synth#5 gets -0.0625

# LEAF 2: weighted_utility = +0.125
# Distribute among: Synth#0, #4, #7 (3 points)
score_per_point = 0.125 / 3 = 0.0417
Synth#0 gets +0.0417
Synth#4 gets +0.0417
Synth#7 gets +0.0417

# LEAF 3: weighted_utility = -0.125
# Distribute among: Synth#2, #6, #9 (3 points)
score_per_point = -0.125 / 3 = -0.0417
Synth#2 gets -0.0417
Synth#6 gets -0.0417
Synth#9 gets -0.0417
```

**Scores from Tree #0**:

```
┌───────────┬──────────────┬────────────────────────────────┐
│ Synthetic │ Utility      │ Why?                           │
│ Point     │ (Tree #0)    │                                │
├───────────┼──────────────┼────────────────────────────────┤
│ Synth#0   │ +0.0417  ✓  │ LEAF 2 (good)                 │
│ Synth#1   │ +0.0625  ✓  │ LEAF 0 (perfect!)             │
│ Synth#2   │ -0.0417  ✗  │ LEAF 3 (bad)                  │
│ Synth#3   │ +0.0625  ✓  │ LEAF 0 (perfect!)             │
│ Synth#4   │ +0.0417  ✓  │ LEAF 2 (good)                 │
│ Synth#5   │ -0.0625  ✗  │ LEAF 1 (terrible!)            │
│ Synth#6   │ -0.0417  ✗  │ LEAF 3 (bad)                  │
│ Synth#7   │ +0.0417  ✓  │ LEAF 2 (good)                 │
│ Synth#8   │ +0.0625  ✓  │ LEAF 0 (perfect!)             │
│ Synth#9   │ -0.0417  ✗  │ LEAF 3 (bad)                  │
└───────────┴──────────────┴────────────────────────────────┘
```

### 4.9 Repeat for Trees #1 and #2

For brevity, let's assume Trees #1 and #2 produce similar scores (with variation):

**Tree #1 Scores**:
```
Synth#0: +0.0203
Synth#1: +0.0512
Synth#2: -0.0301
Synth#3: +0.0598
Synth#4: +0.0381
Synth#5: -0.0514
Synth#6: -0.0389
Synth#7: +0.0299
Synth#8: +0.0701
Synth#9: -0.0421
```

**Tree #2 Scores**:
```
Synth#0: +0.0391
Synth#1: +0.0723
Synth#2: -0.0289
Synth#3: +0.0681
Synth#4: +0.0412
Synth#5: -0.0701
Synth#6: -0.0298
Synth#7: +0.0365
Synth#8: +0.0591
Synth#9: -0.0512
```

### 4.10 Step 6: Aggregate Across All 3 Trees

Now we have a utility matrix:

```python
utility_per_tree = [
    #       Tree0   Tree1   Tree2
    [+0.0417, +0.0203, +0.0391],  # Synth#0
    [+0.0625, +0.0512, +0.0723],  # Synth#1
    [-0.0417, -0.0301, -0.0289],  # Synth#2
    [+0.0625, +0.0598, +0.0681],  # Synth#3
    [+0.0417, +0.0381, +0.0412],  # Synth#4
    [-0.0625, -0.0514, -0.0701],  # Synth#5
    [-0.0417, -0.0389, -0.0298],  # Synth#6
    [+0.0417, +0.0299, +0.0365],  # Synth#7
    [+0.0625, +0.0701, +0.0591],  # Synth#8
    [-0.0417, -0.0421, -0.0512],  # Synth#9
]
```

**Compute Statistics** (example for Synth#5):

```python
# Synth#5 scores across 3 trees
scores = [-0.0625, -0.0514, -0.0701]

# Mean
mean = (-0.0625 + -0.0514 + -0.0701) / 3 = -0.0613

# Standard deviation
std = √[((−0.0625−(−0.0613))² + (−0.0514−(−0.0613))² + (−0.0701−(−0.0613))²) / 2]
    = √[(0.000144 + 0.000980 + 0.000774) / 2]
    = √(0.000949)
    = 0.0308

# Standard error
se = std / √n = 0.0308 / √3 = 0.0308 / 1.732 = 0.0178

# 95% CI using t-distribution (df=2, t_critical=4.303)
ci_lower = mean - t_critical × se
         = -0.0613 - 4.303 × 0.0178
         = -0.0613 - 0.0766
         = -0.1379

ci_upper = mean + t_critical × se
         = -0.0613 + 4.303 × 0.0178
         = -0.0613 + 0.0766
         = +0.0153
```

**Classification for Synth#5**:
```python
ci_lower = -0.1379 < 0
ci_upper = +0.0153 > 0
# CI spans 0 → UNCERTAIN ⚠️
```

**Note**: With only 3 trees, we get wide CIs. With 500 trees:

```python
# Synth#5 with 500 trees (hypothetical)
mean = -0.0613
se = 0.0308 / √500 = 0.00138
ci_lower = -0.0613 - 1.965 × 0.00138 = -0.0640
ci_upper = -0.0613 + 1.965 × 0.00138 = -0.0586
# ci_upper < 0 → RELIABLY HARMFUL ✗
```

### 4.11 Final Results Summary

**With 3 trees** (wide CIs):

```
┌───────────┬─────────┬─────────┬────────────┬────────────┬──────────────┐
│ Synthetic │ Mean    │ SE      │ CI_lower   │ CI_upper   │ Classification│
├───────────┼─────────┼─────────┼────────────┼────────────┼──────────────┤
│ Synth#0   │ +0.0337 │ 0.0061  │ +0.0074    │ +0.0600    │ BENEFICIAL ✓ │
│ Synth#1   │ +0.0620 │ 0.0058  │ +0.0370    │ +0.0870    │ BENEFICIAL ✓ │
│ Synth#2   │ -0.0336 │ 0.0036  │ -0.0491    │ -0.0181    │ HARMFUL ✗    │
│ Synth#3   │ +0.0635 │ 0.0024  │ +0.0532    │ +0.0738    │ BENEFICIAL ✓ │
│ Synth#4   │ +0.0403 │ 0.0009  │ +0.0364    │ +0.0442    │ BENEFICIAL ✓ │
│ Synth#5   │ -0.0613 │ 0.0178  │ -0.1379    │ +0.0153    │ UNCERTAIN ⚠️ │
│ Synth#6   │ -0.0368 │ 0.0034  │ -0.0514    │ -0.0222    │ HARMFUL ✗    │
│ Synth#7   │ +0.0360 │ 0.0034  │ +0.0214    │ +0.0506    │ BENEFICIAL ✓ │
│ Synth#8   │ +0.0639 │ 0.0031  │ +0.0506    │ +0.0772    │ BENEFICIAL ✓ │
│ Synth#9   │ -0.0450 │ 0.0028  │ -0.0570    │ -0.0330    │ HARMFUL ✗    │
└───────────┴─────────┴─────────┴────────────┴────────────┴──────────────┘
```

**Summary**:
- **Reliably beneficial**: 6/10 (60%)
- **Reliably harmful**: 3/10 (30%)
- **Uncertain**: 1/10 (10%)

**With 500 trees** (tighter CIs):
- Synth#5 would likely move from "uncertain" to "reliably harmful"
- All classifications become more confident
- Fewer points span zero

### 4.12 Key Insights from the Example

1. **Synth#5 is hallucinated**:
   - Label says "readmit" (1), but features (AGE=45, Diabetes, NUM_MEDS=2) are similar to Real#F who doesn't readmit
   - Creates LEAF 1 which misclassifies real data
   - Gets negative utility

2. **Synth#1, #3, #8 are beneficial**:
   - All have Heart diagnosis with AGE≤65
   - Create LEAF 0 which perfectly classifies real patients
   - Get strong positive utility

3. **Synth#2, #6, #9 are harmful**:
   - All land in LEAF 3 (older, many meds, predicts no readmit)
   - But real patients with those features (Real#C, #G) DO readmit
   - Teaches wrong patterns

4. **More trees = better confidence**:
   - With 3 trees: 10% uncertain
   - With 500 trees: ~5% uncertain (typical)
   - Narrower CIs allow confident classifications

**What's Next?**

Section 5 explores advanced topics including why we don't check synthetic label alignment, class-specific analysis, and marginal point classification.

---

## 5. Advanced Topics

This section addresses deeper conceptual questions and advanced analysis techniques that strengthen the leaf alignment methodology.

### 5.1 Why We DON'T Check Synthetic Label Alignment

This is one of the most important conceptual aspects of the method and often raises questions from reviewers.

#### The Question

**"Why don't we check if synthetic points' labels align with their leaf predictions?"**

Put another way: "Shouldn't we verify that the model correctly classifies the synthetic training data?"

#### The Short Answer

We evaluate **data quality**, not **model quality**. Checking synthetic label alignment would only tell us if the model memorized the synthetic training set (which it should!), but wouldn't tell us if those learned patterns generalize to real data.

#### The Detailed Explanation

**What We DON'T Measure** (Training Accuracy):
```python
# This would be WRONG to use for data quality assessment
for synth_idx in synthetic_points_in_leaf:
    synth_label = y_synthetic[synth_idx]
    leaf_prediction = 1 if leaf_value > 0 else 0
    synth_aligned = (synth_label == leaf_prediction)  # ← Don't care!
```

**What We DO Measure** (Generalization Quality):
```python
# This is what we actually do - check real data alignment
real_labels = y_real_test[real_indices_in_leaf]
leaf_prediction = 1 if leaf_value > 0 else 0
real_aligned = np.mean(real_labels == leaf_prediction)  # ← This matters!
utility = real_aligned - 0.5
```

#### Why This Design Choice Matters

**The Core Insight**: Synthetic labels could be wrong!

If we measured synthetic alignment, we'd only assess **training accuracy** - which can be high even for hallucinated data (the model memorized wrong patterns perfectly).

By measuring **real alignment**, we detect when synthetic data taught wrong correlations, regardless of training fit.

#### Concrete Example: Hallucinated Point

```
Synth#5: AGE=45, DIAGNOSIS=Diabetes, NUM_MEDS=2, Label=1 (readmit)
         ↓
    Lands in LEAF 1 (leaf_value = +0.3 → predicts Class 1)
         ↓
    Synthetic alignment: ✓ (label=1 matches prediction=1)
         ↓
    Real test data in LEAF 1:
      Real#F: AGE=48, DIAGNOSIS=Diabetes, NUM_MEDS=3, Label=0 (no readmit)
         ↓
    Real alignment: ✗ (leaf predicts 1, but real patient is 0)
         ↓
    Utility: -0.5 (maximum negative)
         ↓
    Synth#5 gets NEGATIVE score → RELIABLY HARMFUL!
```

**Interpretation**:
- Synth#5's **label (1)** happened to match its leaf prediction (+0.3)
- But the leaf **FAILS** on real data
- This means: Synth#5 taught the model that "young, diabetic patients with few meds" should readmit
- **Reality**: Real patients with those features DON'T readmit
- **Conclusion**: Synth#5 has the **wrong label** for its features → it's hallucinated!

#### Can Synthetic Points Land in "Wrong" Leaves?

**YES!** This happens due to:

1. **Regularization**: LightGBM prevents overfitting, so not every training point gets perfect prediction
2. **Gradient Boosting**: Each tree corrects residual errors, not original labels
3. **Outliers**: Points far from their class distribution may land in opposite-class leaves

**Example**:

```
Most synthetic Class 1 patients: AGE=65-85, NUM_MEDS=3-6, Diabetes/Heart

Synth#99: AGE=25, NUM_MEDS=15, Cancer, Label=1 (outlier)

This outlier has features similar to Class 0 patients.
It lands in a leaf with mostly Class 0 training points.
That leaf predicts Class 0.

Synth#99 (label=1) is in a leaf predicting Class 0!
```

**Why this is GOOD for our method**: It highlights that Synth#99 is problematic - it has Class 1 label but Class 0 features!

#### Summary Table

| Aspect | Check Synthetic Alignment? | Check Real Alignment? |
|--------|---------------------------|----------------------|
| What it measures | Training fit (overfitting) | Generalization (quality) |
| Useful for | Model debugging | Data quality assessment |
| Our method uses | ❌ No | ✅ Yes |
| Why | Synthetic labels could be wrong! | Real labels are ground truth |

#### Defense to Reviewers

**Complete Response**:

> "Our method evaluates synthetic **data quality**, not model quality. We assume real test labels are ground truth and ask: do decision boundaries learned from synthetic data generalize to real data?
>
> If we measured synthetic alignment, we'd only assess training accuracy—which can be high even for hallucinated data (the model memorized wrong patterns perfectly). By measuring real alignment, we detect when synthetic data taught wrong correlations, regardless of training fit.
>
> Concrete example: A synthetic readmission case (label=1) with 'AGE=45, non-Heart diagnosis' creates a leaf predicting readmission. But real patients with those features don't readmit. The synthetic point had the wrong label for its features—it's hallucinated. Checking only synthetic alignment would miss this."

### 5.2 Class-Specific Analysis

#### Why It Matters

Leaf alignment provides **overall** quality assessment, but class-specific breakdown reveals **asymmetric failure patterns**, especially critical for imbalanced tasks.

#### The Analysis

From `sdvaluation/leaf_alignment.py:381`:

```python
# Analyze by class
for class_label in [0, 1]:
    class_mask = y_synthetic == class_label
    class_harmful = reliably_hallucinated & class_mask
    class_beneficial = reliably_beneficial & class_mask
    class_uncertain = uncertain & class_mask

    print(f"\nClass {class_label} breakdown:")
    print(f"  Harmful:     {np.sum(class_harmful)} ({100*np.mean(class_harmful):.2f}%)")
    print(f"  Beneficial:  {np.sum(class_beneficial)} ({100*np.mean(class_beneficial):.2f}%)")
    print(f"  Uncertain:   {np.sum(class_uncertain)} ({100*np.mean(class_uncertain):.2f}%)")
```

#### Real Example: Gen2 Synthetic Data

**Overall Results**:
```
Total: 10,000 synthetic points
  Reliably harmful:     9,339 (93.39%)
  Reliably beneficial:     54 (0.54%)
  Uncertain:              607 (6.07%)
```

**Class-Specific Breakdown**:
```
Class 0 (No Readmission) - 8,000 points:
  Harmful:     7,412 (92.65%)
  Beneficial:     54 (0.68%)
  Uncertain:     534 (6.68%)

Class 1 (Readmission) - 2,000 points:
  Harmful:     1,927 (96.35%)  ← WORSE!
  Beneficial:      0 (0.00%)   ← CRITICAL!
  Uncertain:      73 (3.65%)
```

#### Interpreting Class Breakdown

**Both classes bad** (uniform failure):
- Synthetic generator failed to capture patterns for both classes
- Systematic distribution mismatch

**Positive class worse** (asymmetric failure):
- Common in imbalanced datasets (minority class = Class 1)
- Synthetic generator struggled with rare class
- Less training data for minority class → worse synthetic quality

**0% beneficial in minority class** (CRITICAL):
- Not a single useful positive-class synthetic point!
- Model trained on this data **cannot learn** minority class patterns
- Explains catastrophic recall drop (40% → 10%)

#### Impact on Model Performance

**Connection to Metrics**:

```
Class 1 (Readmission):
  0% beneficial synthetic points
  ↓
  Model learns NO correct patterns for readmission
  ↓
  Recall: 39.96% (real data) → 10.49% (Gen2)  ← 74% drop!
  ↓
  Most readmission patients misclassified as "no readmit"
  ↓
  CLINICAL DISASTER: Missing patients who will return to hospital!
```

**Why This Matters for Healthcare**:

In hospital readmission prediction:
- **False Negative** (missed readmission): Patient returns unexpectedly, no preventive care arranged
- **False Positive** (predicted readmission, doesn't happen): Extra follow-up, minor inconvenience

0% beneficial Class 1 points → Can't detect actual readmissions → Dangerous!

#### When to Worry About Class-Specific Results

| Class Breakdown | Interpretation | Action |
|-----------------|----------------|--------|
| Both ≈0.5% harmful | Excellent, matches real data | ✅ Use as-is |
| Both 10-20% harmful | Moderate degradation, symmetric | ⚠️ Filter harmful points |
| Class 1 >>Class 0 harmful | Minority class failure | ❌ Don't use, fix generator |
| Class 1: 0% beneficial | Catastrophic minority failure | ❌ REJECT, critical issue |

### 5.3 Marginal Point Classification

#### The Problem

Current classification is binary (per class):
- CI_upper < 0 → Harmful
- CI_lower > 0 → Beneficial
- CI spans 0 → Uncertain

But **strength** of signal varies widely:

```
Point A: mean = +0.0612, CI = [+0.0605, +0.0619]  ← STRONGLY beneficial
Point B: mean = +0.0008, CI = [+0.0004, +0.0012]  ← Marginally beneficial
```

Both classified as "beneficial", but A is **76× stronger** than B!

#### The Issue

**Marginal points** have:
- CI entirely on one side of zero (reliably classified)
- But **very close** to zero (weak signal)

These points contribute almost nothing to model training. Including them:
- Doesn't improve performance much
- Adds computational cost
- May add noise

#### Proposed Solutions

**Option 1: Absolute Threshold**

```python
# Keep only strong beneficial points
strong_threshold = 0.01
strong_beneficial = (ci_lower > 0) & (mean_utility > strong_threshold)

# Example:
Point A: mean = +0.0612 > 0.01 → Keep ✓
Point B: mean = +0.0008 < 0.01 → Filter ✗
```

**Option 2: Percentile-Based**

```python
# Keep top 50% of beneficial points
beneficial_mask = ci_lower > 0
beneficial_scores = mean_utility[beneficial_mask]
threshold = np.percentile(beneficial_scores, 50)  # Median

strong_beneficial = beneficial_mask & (mean_utility > threshold)
```

**Option 3: Effect Size (Mean/SE Ratio)**

```python
# t-statistic: how many standard errors away from zero?
t_statistic = mean_utility / se_utility

# Keep points with t > 3 (very strong evidence)
strong_beneficial = (ci_lower > 0) & (t_statistic > 3)
```

**Option 4: Five-Tier Classification**

```python
if ci_upper < -0.01:
    classification = "STRONGLY HARMFUL"
elif ci_upper < 0:
    classification = "MARGINALLY HARMFUL"
elif ci_lower > 0.01:
    classification = "STRONGLY BENEFICIAL"
elif ci_lower > 0:
    classification = "MARGINALLY BENEFICIAL"
else:
    classification = "UNCERTAIN"
```

#### Current Status

These enhancements are **proposed** but not yet implemented. See `ENHANCEMENTS.md` for detailed discussion.

For most use cases, the current three-way classification is sufficient. Marginal filtering is only needed when:
- Dataset is very large (>100k points)
- Computational resources are limited
- You need maximum precision (e.g., generating training data for production)

### 5.4 The n_estimators Trade-off

#### What n_estimators Controls

**Dual Role**:
1. **Boosting**: Number of sequential trees in LightGBM ensemble
2. **Statistics**: Number of independent measurements for confidence intervals

#### The Mathematical Relationship

Standard error decreases with square root of n:

```
SE = σ / √n_trees

100 trees:   SE = σ / 10     → Wide CIs
500 trees:   SE = σ / 22.4   → 55% narrower ✓ (recommended)
1000 trees:  SE = σ / 31.6   → 68% narrower
5000 trees:  SE = σ / 70.7   → 86% narrower
```

**Diminishing Returns**:
- 100→500: 55% improvement
- 500→1000: 29% improvement
- 1000→5000: 55% improvement (but 5× slower!)

#### Practical Impact Table

| Trees | CI Width | Uncertain % | Runtime | Use Case |
|-------|----------|-------------|---------|----------|
| 100 | Wider | ~20-30% | ~2 min | Quick exploration |
| 200 | Medium-Wide | ~10-15% | ~3 min | Rapid iteration |
| **500** | **Medium** | **~5-10%** | **~5 min** | **Recommended** ✓ |
| 1000 | Tight | ~3-5% | ~10 min | High precision |
| 2000 | Very Tight | ~2-3% | ~20 min | Publication-ready |
| 5000 | Extremely Tight | ~1-2% | ~50 min | Maximum confidence |

#### Example: Effect of n_trees

**With 100 trees**:
```
Point #42:
  mean = -0.002
  se = 0.002  (larger SE)
  CI = [-0.006, +0.002]  ← Spans 0
  Classification: UNCERTAIN ⚠️
```

**With 500 trees**:
```
Point #42:
  mean = -0.002
  se = 0.000845  (smaller SE)
  CI = [-0.004, -0.0004]  ← Doesn't span 0
  Classification: RELIABLY HARMFUL ✗
```

**With 5000 trees**:
```
Point #42:
  mean = -0.002
  se = 0.000283  (even smaller SE)
  CI = [-0.0026, -0.0014]  ← Very tight
  Classification: RELIABLY HARMFUL ✗ (high confidence)
```

#### Recommendations

**Quick exploration** (100-200 trees):
- Initial data quality check
- Rapid prototyping
- When you just need a rough estimate

**Standard analysis** (500 trees) ← **RECOMMENDED**:
- Balance of speed vs precision
- ~5-10% uncertain (acceptable)
- Suitable for most production use cases

**High precision** (1000-2000 trees):
- Final analysis before deployment
- When decision is critical
- Publication or reporting to stakeholders

**Maximum confidence** (5000+ trees):
- Research publications
- When you need <2% uncertain
- Computational resources not a constraint

#### Cost-Benefit Analysis

```
Your 10,000 synthetic points with varying n_estimators:

100 trees:  ~2 min,  2,500 uncertain (25%)
500 trees:  ~5 min,    607 uncertain (6%)   ← 80% reduction!
1000 trees: ~10 min,   304 uncertain (3%)   ← 50% reduction
5000 trees: ~50 min,    61 uncertain (0.6%) ← 80% reduction

500→1000: 2× time for 50% fewer uncertain
500→5000: 10× time for 90% fewer uncertain
```

**Verdict**: 500 trees hits the sweet spot for most use cases.

#### When to Use More Than 500 Trees

**Justified**:
- Small dataset (<1000 points) where every classification matters
- Publication-quality results
- Critical production deployment
- Very noisy data (high variance across trees)

**Overkill**:
- Large dataset (>10k points) where 5% uncertain is fine
- Exploratory analysis
- When runtime is a bottleneck
- Preliminary data quality check

**What's Next?**

Section 6 covers practical application: interpreting results, quality assessment tiers, and real-world decision-making.

---

## Section 6: Practical Application

### 6.1 The Big Question: "Is My Synthetic Data Good Enough?"

After running leaf alignment, you get numbers like:

```
Reliably harmful:      9,339 (93.39%)
Reliably beneficial:      54 ( 0.54%)
Uncertain:               607 ( 6.07%)
```

**But what does this mean?**
- Is 93% harmful catastrophic?
- What's acceptable?
- Should I filter or reject entirely?

To answer these questions, you need a **baseline for comparison**.

### 6.2 Benchmark: Real Training Data (The Gold Standard)

#### Running Leaf Alignment on Real Training Data

To establish a baseline, run leaf alignment on your real training data as if it were synthetic:

```bash
# Baseline evaluation
uv run sdvaluation eval \
  --dseed-dir path/to/real/data \
  --synthetic-file path/to/real/train.csv \  # Real data AS IF synthetic
  --n-estimators 500 \
  --output baseline_scores.csv
```

#### Expected Results for Good Data

Example from MIMIC-III real training data (n=10,000):

```
Statistical Confidence (95% CI-based):
  Reliably harmful:        25 ( 0.25%)  ✓✓✓
  Reliably beneficial:  8,969 (89.69%)  ✓✓✓
  Uncertain:            1,006 (10.06%)
```

**Key metrics**:
- **0.25% harmful** ← Natural noise/outliers in real data
- **89.69% beneficial** ← Most real data helps the model
- **10.06% uncertain** ← Some variance is normal

**This is your baseline!** Synthetic data should ideally match these percentages.

### 6.3 Quality Assessment Tiers

#### Tier 1: Excellent Quality (Real-Like) ✓✓✓

**Characteristics**:
- Harmful: **0-2%**
- Beneficial: **>85%**
- Uncertain: **5-15%**

**Example**: Real training data

**Decision**: ✅ **USE without filtering**

**Interpretation**:
- Matches real data distribution closely
- Safe to use as-is for training
- Minor filtering optional (remove <2% harmful)

**Code**:
```python
# Optional: Remove the small fraction of harmful points
filtered = synthetic_data[results['utility_ci_upper'] >= 0]
```

---

#### Tier 2: Good Quality (Usable with Minor Filtering) ✓✓

**Characteristics**:
- Harmful: **2-10%**
- Beneficial: **70-85%**
- Uncertain: **10-25%**

**Example**: High-quality synthetic (well-tuned GAN)

**Decision**: ✅ **USE after filtering harmful points**

**Interpretation**:
- Slightly degraded but still useful
- Filter out 2-10% harmful points
- Expect minor performance drop vs real data (~5-10%)

**Code**:
```python
# Filter out harmful points
filtered = synthetic_data[results['utility_ci_upper'] >= 0]

# Use filtered dataset for training
model.fit(filtered[features], filtered[target])
```

---

#### Tier 3: Mediocre Quality (Marginal - Use with Caution) ⚠️

**Characteristics**:
- Harmful: **10-30%**
- Beneficial: **40-70%**
- Uncertain: **20-40%**

**Example**: Poorly tuned synthetic generator

**Decision**: ⚠️ **USE only if no alternative, heavy filtering required**

**Interpretation**:
- Significant quality issues
- Large portion of data is harmful
- Expect 20-40% performance drop
- High uncertainty indicates inconsistent patterns

**Code**:
```python
# Conservative: Keep only reliably beneficial
filtered = synthetic_data[results['utility_ci_lower'] > 0]

# More conservative: Add marginal threshold
strong_beneficial = (results['utility_ci_lower'] > 0) & \
                   (results['utility_score'] > 0.01)
filtered = synthetic_data[strong_beneficial]

# Warning: Expect to lose 60-90% of data after filtering
print(f"Retained: {len(filtered)}/{len(synthetic_data)} "
      f"({100*len(filtered)/len(synthetic_data):.1f}%)")
```

---

#### Tier 4: Poor Quality (Not Recommended) ✗✗

**Characteristics**:
- Harmful: **30-70%**
- Beneficial: **5-40%**
- Uncertain: **10-30%**

**Example**: Mismatched synthetic (wrong hyperparameters)

**Decision**: ❌ **DO NOT USE - investigate generator issues**

**Interpretation**:
- Majority of data is harmful
- Very few beneficial points
- Training on this will degrade performance
- Likely distribution mismatch or generator failure

**Action**:
1. Don't use this data
2. Debug synthetic generator:
   - Check for data leakage
   - Look for mode collapse
   - Verify training data was correct
   - Review hyperparameters

---

#### Tier 5: Catastrophic Quality (Unusable) ✗✗✗

**Characteristics**:
- Harmful: **>70%**
- Beneficial: **<5%**
- Uncertain: **<20%**

**Example**: Gen2 (recursive generation), severely broken GAN

**Decision**: ❌ **REJECT completely - generator is fundamentally broken**

**Interpretation**:
- Almost all data creates wrong decision boundaries
- Synthetic generator has catastrophic failure
- Would destroy model performance if used
- Not salvageable even with aggressive filtering

**Example from MIMIC-III Gen2**:
```
Reliably harmful:      9,339 (93.39%)  ✗✗✗
Reliably beneficial:      54 ( 0.54%)  ✗✗✗
Uncertain:               607 ( 6.07%)

Ratio vs Real: 373× more hallucinated!
```

**Action**:
1. Completely reject this synthetic data
2. Do NOT use even after filtering (only 0.5% usable!)
3. Investigate root cause:
   - Recursive generation? (Real → Gen1 → Gen2)
   - Mode collapse in GAN?
   - Wrong training configuration?
   - Data preprocessing issues?

### 6.4 Decision Matrix

| % Harmful | % Beneficial | Decision | Action |
|-----------|--------------|----------|---------|
| 0-2% | >85% | ✅ USE | No filtering needed |
| 2-10% | 70-85% | ✅ USE | Filter harmful points |
| 10-30% | 40-70% | ⚠️ CAUTION | Heavy filtering, expect performance drop |
| 30-70% | 5-40% | ❌ REJECT | Don't use, fix generator |
| >70% | <5% | ❌ REJECT | Catastrophic failure, start over |

### 6.5 Real-World Examples from MIMIC-III

#### Example 1: Real Training Data (Baseline)

**Dataset**: Real MIMIC-III admissions data (n=10,000)

**Results**:
```
Harmful:      25 (0.25%)
Beneficial:   8,969 (89.69%)
Uncertain:    1,006 (10.06%)
```

**Performance on test set**:
```
Precision: 18.34%
Recall:    39.96%
F1:        25.14%
```

**Conclusion**: ✅ Excellent baseline (Tier 1)

---

#### Example 2: Gen1 Synthetic Data

**Dataset**: SynthCity Marginal Distributions (Real → Gen1, n=10,000)

**Results**:
```
Harmful:      9,444 (94.44%)  ✗✗✗
Beneficial:      45 ( 0.45%)
Uncertain:      511 ( 5.11%)
```

**Performance on test set**:
```
Precision:  7.45%   (-10.89% vs Real)
Recall:     8.92%   (-31.04% vs Real)  ← Catastrophic!
F1:         8.11%   (-17.03% vs Real)
```

**Ratio vs Real**: 378× more hallucinated

**Conclusion**: ❌ Catastrophic failure (Tier 5)

**Action**: REJECT - do not use

---

#### Example 3: Gen2 Synthetic Data (Recursive)

**Dataset**: SynthCity Marginal Distributions (Real → Gen1 → Gen2, n=10,000)

**Results**:
```
Harmful:      9,339 (93.39%)  ✗✗✗
Beneficial:      54 ( 0.54%)
Uncertain:      607 ( 6.07%)
```

**Performance on test set**:
```
Precision:  7.86%   (-10.48% vs Real)
Recall:    10.49%   (-29.47% vs Real)  ← Major degradation
F1:         8.99%   (-16.15% vs Real)
```

**Ratio vs Real**: 373× more hallucinated

**Conclusion**: ❌ Catastrophic failure (Tier 5)

**Key insight**: Slightly better than Gen1 (94.44% → 93.39%), but still unusable. Recursive training (Real → Gen1 → Gen2) did NOT degrade further significantly, but initial failure at Real→Gen1 was catastrophic.

**Action**: REJECT - recursive generation failed at first step

### 6.6 How to Present Findings to Stakeholders

#### Executive Summary Format

```
═══════════════════════════════════════════════════
     Synthetic Data Quality Assessment
═══════════════════════════════════════════════════

Dataset Evaluated: Gen2_SynthCity_10k
Evaluation Method: Leaf Co-Occurrence Analysis
Test Set: MIMIC-III Real Admissions (n=8,000)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL ASSESSMENT: ❌ REJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quality Metrics:
  ✗ 93.39% of data creates wrong decision boundaries
  ✗ Only 0.54% creates useful patterns
  ✗ 373× more hallucinated than real data
  ✗ Recall drops 40% → 10% when using this data

Key Findings:
  1. Synthetic data has fundamentally wrong correlations
  2. Cannot learn readmission patterns (95% of positive
     class hallucinated)
  3. Not salvageable even with aggressive filtering

Recommendation:
  DO NOT USE this synthetic data for model training.
  Root cause: Recursive generation (Real→Gen1→Gen2)
  failed at first step.

Next Steps:
  1. Investigate GAN/VAE training on Real→Gen1
  2. Consider different synthetic generator
  3. Use Real training data for now

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Detailed Report Structure

**Section 1: Methodology**

```
Method: Leaf Co-Occurrence Alignment Analysis
  - Trained LightGBM (500 trees) on synthetic data
  - Evaluated decision boundaries on real test data
  - Scored each synthetic point based on generalization
  - 95% confidence intervals for statistical rigor
```

**Section 2: Results Summary**

```
Classification (95% CI):
  Reliably Harmful:      9,339 (93.39%)
  Reliably Beneficial:      54 ( 0.54%)
  Uncertain:               607 ( 6.07%)

Comparison to Baseline (Real training data):
                  Real      Synthetic    Ratio
  Harmful:        0.25%     93.39%       373×
  Beneficial:    89.69%      0.54%      0.006×
```

**Section 3: Performance Impact**

```
Model Performance (on real test data):

Training Data:   Real      Gen2      Change
Precision:       18.34%    7.86%     -10.48%  ✗
Recall:          39.96%   10.49%     -29.47%  ✗✗✗
F1 Score:        25.14%    8.99%     -16.15%  ✗✗

Critical Finding: Recall drops by 74% (relative)
  → Model loses ability to detect readmissions
```

**Section 4: Root Cause Analysis**

```
Why Gen2 Failed:

Class-Specific Breakdown:
  Negative Class (no readmission):
    Harmful: 8,385 / 8,997 (93.20%)

  Positive Class (readmission):
    Harmful:     954 / 1,003 (95.11%)
    Beneficial:    0 / 1,003 ( 0.00%)  ← ZERO useful points!

Root Cause:
  - Gen2 positive class is 100% hallucinated
  - Cannot learn readmission patterns at all
  - Recursive training compounded errors from Gen1
```

**Section 5: Recommendations**

```
1. REJECT Gen2 synthetic data - unusable quality

2. Investigate Gen1 generation:
   - Check GAN/VAE training logs
   - Verify input data preprocessing
   - Check for mode collapse
   - Review hyperparameters

3. Consider alternatives:
   - Use Real training data (if available)
   - Try different synthetic method (SMOTE, CTGAN, etc.)
   - Mix real + synthetic (if small real dataset available)

4. If must use synthetic, need 373× improvement:
   - Target: <5% harmful (vs current 93%)
   - Require: >70% beneficial (vs current 0.5%)
```

### 6.7 Decision Tree for Quick Reference

```
                    Start
                      |
          ┌───────────┴───────────┐
          │   Run Leaf Alignment  │
          └───────────┬───────────┘
                      |
              Get % Harmful
                      |
          ┌───────────┴───────────────────────┐
          |                                   |
    % Harmful < 2%                    % Harmful > 70%
          |                                   |
          ✅ USE                              ❌ REJECT
    (Excellent)                         (Catastrophic)
                                              |
                    ┌─────────────────────────┤
                    |                         |
            2% < Harmful < 10%        10% < Harmful < 70%
                    |                         |
            ✅ USE (filter)            ⚠️ INVESTIGATE
            (Good quality)              (Poor quality)
                                              |
                                    Check % Beneficial
                                              |
                                    ┌─────────┴─────────┐
                                    |                   |
                            Beneficial > 40%    Beneficial < 40%
                                    |                   |
                            ⚠️ USE (caution)     ❌ REJECT
                            Heavy filter        (Too degraded)
```

### 6.8 Common Pitfalls in Interpretation

#### Pitfall 1: Ignoring Class-Specific Breakdown

**Wrong**:
> "Overall 93% harmful, but maybe the negative class is still good?"

**Right**:
> "Check class-specific stats! If positive class is 95% harmful and negative is 93% harmful, BOTH classes are broken."

**Solution**: The `sdvaluation eval` command always shows class-specific breakdown. Always review both classes!

---

#### Pitfall 2: Filtering Without Understanding Scale

**Wrong**:
> "93% harmful, so I'll filter those out and use the 7% remaining."

**Right**:
> "7% remaining = 700 points, but only 0.5% are reliably beneficial = 50 points. After filtering, dataset is too small to be useful."

**Solution**: Check absolute numbers, not just percentages:

```python
print(f"After filtering: {len(filtered)} points")
print(f"  Beneficial: {n_beneficial}")
print(f"  Uncertain: {n_uncertain}")
print(f"Is this enough data for training? ({len(filtered)} samples)")
```

---

#### Pitfall 3: Comparing to Wrong Baseline

**Wrong**:
> "93% harmful sounds bad, but what's the baseline?"

**Right**:
> "Real training data has 0.25% harmful. 93% is 373× worse than baseline!"

**Solution**: Always run baseline comparison first using real training data.

---

#### Pitfall 4: Ignoring Performance Metrics

**Wrong**:
> "Leaf alignment shows 93% harmful, but confusion matrix shows only 30% recall drop, so it's not that bad."

**Right**:
> "30% absolute drop (40% → 10%) is actually 75% relative drop. That's catastrophic for clinical applications where recall matters!"

**Solution**: Look at both leaf alignment AND downstream performance:

```python
# Compare model performance
print("Real training:   Recall = 39.96%")
print("Synthetic:       Recall = 10.49%")
print(f"Relative drop:   {(39.96-10.49)/39.96*100:.1f}%")  # 74% drop!
```

### 6.9 Summary: Key Takeaways

| Metric | Excellent | Good | Poor | Catastrophic |
|--------|-----------|------|------|--------------|
| % Harmful | 0-2% | 2-10% | 10-70% | >70% |
| % Beneficial | >85% | 70-85% | 5-70% | <5% |
| Ratio vs Real | 1-10× | 10-50× | 50-200× | >200× |
| Action | Use as-is | Filter & use | Reject | Reject |
| Performance | ~Real | -5 to -10% | -20 to -40% | >-50% |

#### The Golden Rule

**If synthetic data is more than 50× more hallucinated than real training data, it's probably not worth using.**

Why 50×?
- Real data: ~0.25% harmful (natural noise)
- 50× worse: 12.5% harmful
- At this threshold, you're borderline Tier 3 (Mediocre)
- Beyond this, filtering becomes impractical

**What's Next?**

Section 7 compares Leaf Alignment with other evaluation methods (Data Shapley, confusion matrices, distribution metrics).

---

## Section 7: Comparisons with Other Methods

This section explores how Leaf Alignment differs from and complements other synthetic data evaluation methods.

### 7.1 Leaf Alignment vs Data Shapley

Both methods evaluate data quality, but they answer fundamentally different questions.

#### Quick Recap: What Each Method Does

**Data Shapley** (Original Method):

**Question**: "What is the marginal contribution of each training point when added to random subsets of other training points?"

**Process**:
1. For each training point, sample random coalitions (subsets) of other points
2. Train model on coalition WITHOUT the point → measure performance
3. Train model on coalition WITH the point → measure performance
4. Marginal contribution = performance difference
5. Average across many random coalitions

**Output**: Shapley value (positive = helpful, negative = harmful)

---

**Leaf Alignment** (This Method):

**Question**: "Do the decision boundaries learned from synthetic data generalize to real test data patterns?"

**Process**:
1. Train ONE model on ALL synthetic data
2. Pass synthetic training + real test data through the model
3. For each leaf: check if it correctly classifies real test data
4. Assign utility to synthetic points based on their leaves
5. Aggregate across 500 trees for statistical confidence

**Output**: Utility score (positive = beneficial, negative = harmful)

#### Key Differences

##### Difference 1: What They Measure

**Data Shapley**:
- **Measures**: Individual point quality in RANDOM SUBSETS
- **Focus**: Marginal contribution
- **Context**: "How much does this point help when combined with others?"

Example: Point A might be:
- Helpful when combined with points {B, C, D}
- Harmful when combined with points {E, F, G}
- → Shapley value = average across all combinations

**Leaf Alignment**:
- **Measures**: Structural quality on FULL DATASET
- **Focus**: Decision boundary alignment
- **Context**: "Does this point create boundaries that work on real data?"

Example: Point A creates a leaf that:
- Predicts Class 1
- Real patients land there with label 0
- → Utility = negative (misaligned)

##### Difference 2: Training Approach

**Data Shapley**:
```
Training: O(num_samples × n) model training runs
          ~100 samples × 10,000 points = 1,000,000 models

Runtime:  ~90 minutes (slow!)

Type:     COUNTERFACTUAL
          "What if we remove this point?"
```

**Leaf Alignment**:
```
Training: O(1) = ONE model training run

Runtime:  ~5 minutes (fast!)

Type:     EVALUATION
          "Given that we trained on this, does it work?"
```

##### Difference 3: What They Detect

**Data Shapley**:

Detects:
- ✓ Duplicate points (no marginal value)
- ✓ Outliers (hurt when added to subsets)
- ✓ Points that conflict with others
- ✓ Individual point quality

Misses:
- ✗ Distributional issues (collective patterns)
- ✗ Hallucinations that look individually plausible
- ✗ Wrong correlations that only show at scale

**Leaf Alignment**:

Detects:
- ✓ Distributional hallucinations
- ✓ Wrong decision boundaries
- ✓ Collective pattern failures
- ✓ Generalization issues

Misses:
- ✗ Individual outlier quality
- ✗ Redundancy (duplicates)
- ✗ Subset-dependent effects

#### The Gen2 Case Study: Why Shapley Missed It

**The Problem**:

Gen2 Synthetic Data (Real → Gen1 → Gen2):

```
Confusion Matrix Analysis:
  Real training → Test:  Recall = 40%
  Gen2 training → Test:  Recall = 10%  ✗ (-30% absolute)

Leaf Alignment:
  Reliably harmful: 93.39%  ✗✗✗ Catastrophic!

Data Shapley:
  Reliably harmful: 3.39%   ✓ Seems fine?!
```

**Shapley said Gen2 was fine, but it clearly wasn't!**

**Why Shapley Missed the Problem**:

Root cause: **Distributional hallucination**

Gen2 points individually look plausible:
- Valid feature ranges (AGE: 25-85, NUM_MEDS: 0-20)
- No obvious outliers
- Pass basic sanity checks

BUT collectively have wrong patterns:
- Wrong correlation: AGE vs NUM_MEDS
- Wrong correlation: DIAGNOSIS vs READMISSION
- Mode collapse toward majority class

```
Individual quality ✓ (Shapley)
Collective quality ✗ (Leaf Alignment)
```

#### The Marginal Contribution Paradox

What Shapley saw:

```python
# Shapley evaluation for Gen2 Point #42
# Test in random subsets:

Coalition 1: {Real points 1, 5, 12, 89, ...}  (mostly real)
  Without #42: Performance = 0.72
  With #42:    Performance = 0.72
  Marginal:    0.00  (neutral - drowned out by real data)

Coalition 2: {Gen2 points 3, 8, 15, 29, ...}  (mostly Gen2)
  Without #42: Performance = 0.30  (bad coalition)
  With #42:    Performance = 0.31  (slightly better)
  Marginal:    +0.01  (helpful relative to other bad points!)

Coalition 3: {Mix of real and Gen2}
  Marginal:    -0.005

Average Shapley value: +0.002  → Appears slightly beneficial!
```

The issue:
- Gen2 points are **consistent with each other** (wrong, but consistent)
- When added to other Gen2 points, they don't hurt much
- When added to real points, they're drowned out
- Marginal contribution looks fine
- But when ALL Gen2 is used: consistent wrong patterns dominate

**What Leaf Alignment Saw**:

```python
# Train model on ALL Gen2 data (10,000 points)
model = train(all_gen2_data)

# Point #42 lands in Leaf 237
# Leaf 237 predicts: Class 1 (readmission)

# Real patients in Leaf 237:
real_labels = [0, 0, 0, 0, 1, 0, 0]  # Mostly Class 0

# Accuracy = 1/7 = 14%
# Utility = 0.14 - 0.5 = -0.36  ✗ Harmful!

# This pattern repeats across 450/500 trees
# Mean utility = -0.0245
# CI = [-0.0251, -0.0239]  → Reliably harmful
```

The key:
- Trains on full Gen2 dataset (not subsets)
- Sees the collective wrong patterns
- Detects that boundaries don't generalize to real data

#### Side-by-Side Comparison

**Gen2 Results**:

| Method | Harmful | Beneficial | Runtime | Detected Problem? |
|--------|---------|------------|---------|-------------------|
| Shapley | 3.39% | 96.61% | 90 min | ❌ NO - looked fine |
| Leaf Alignment | 93.39% | 0.54% | 5 min | ✅ YES - catastrophic |
| Confusion Matrix | N/A | N/A | 2 min | ✅ YES - 30% recall drop |

**Leaf Alignment agreed with confusion matrix. Shapley didn't.**

**Real Training Data Results**:

| Method | Harmful | Beneficial | Runtime |
|--------|---------|------------|---------|
| Shapley | 3.28% | 96.72% | 90 min |
| Leaf Alignment | 0.25% | 89.69% | 5 min |
| Confusion Matrix | N/A (baseline) | N/A | 2 min |

Both methods agree real data is high quality.

#### When to Use Each Method

**Use Data Shapley When**:

✅ **Cleaning individual noisy labels**
- **Scenario**: Real training data with label errors
- **Goal**: Find mislabeled points
- **Example**: Medical records with wrong diagnoses

✅ **Removing redundant points**
- **Scenario**: Large dataset with duplicates
- **Goal**: Reduce dataset size without losing performance
- **Example**: Deduplicate patient records

✅ **Fair data valuation**
- **Scenario**: Multiple data contributors
- **Goal**: Fairly compensate each contributor
- **Example**: Data marketplace with pricing

✅ **Understanding subset interactions**
- **Scenario**: Complex dataset with dependencies
- **Goal**: See how points interact in subsets
- **Example**: Feature selection with correlations

**Use Leaf Alignment When**:

✅ **Evaluating synthetic data quality**
- **Scenario**: Generated data from GAN/VAE/SMOTE
- **Goal**: Check if it matches real distribution
- **Example**: Validating synthetic medical records

✅ **Detecting distributional issues**
- **Scenario**: Data might have wrong collective patterns
- **Goal**: Find if learned boundaries generalize
- **Example**: Check for mode collapse in GAN

✅ **Fast quality screening**
- **Scenario**: Need quick assessment (5 min vs 90 min)
- **Goal**: Rapid evaluation of multiple generations
- **Example**: Iterate on synthetic generation parameters

✅ **Class-specific diagnosis**
- **Scenario**: Imbalanced classes, minority class critical
- **Goal**: Check if both classes are well-represented
- **Example**: Rare disease detection

#### They're Complementary, Not Competing

**Use Both Together**:

Workflow for synthetic data validation:

```
Step 1: Confusion Matrix (~2 min)
  → Quick check: Does aggregate performance drop?

Step 2: Leaf Alignment (~5 min)
  → If performance drops: Which points are hallucinated?
  → Class-specific breakdown

Step 3: Data Shapley (~90 min) - Optional
  → If results differ from leaf alignment: Why?
  → Subset-dependent effects?
  → Individual point quality vs collective quality
```

Example decision tree:

```
Confusion matrix shows:
  Gen2 recall drops 30%  ✗
            ↓
Leaf alignment shows:
  93% hallucinated  ✗✗✗
            ↓
Decision: REJECT Gen2
No need for Shapley - problem is clear!

But if results were ambiguous:
            ↓
Run Shapley to understand:
  - Are points individually bad?
  - Or collectively bad but individually OK?
  - Subset-dependent effects?
```

#### Real Example: Complementary Insights

**Scenario**: Partially Corrupted Dataset

Setup:
- Dataset: 10,000 synthetic points
- Issue: 1,000 points have wrong correlations
- Remaining: 9,000 points are high quality

**Data Shapley Results**:
```
Harmful points: 1,200 (12%)

Top harmful:
  - 1,000 genuinely wrong points (detected ✓)
  - 200 outliers in the good data (detected ✓)
```

**Leaf Alignment Results**:
```
Harmful points: 1,050 (10.5%)

Top harmful:
  - 1,000 wrong correlation points (detected ✓)
  - 50 points in empty leaves (detected ✓)
  - Misses the 200 outliers (individually OK when combined)
```

**Combined insight**:
- 1,000 points are definitely bad (both agree)
- 200 are outliers (Shapley only)
- 50 create empty regions (Leaf alignment only)
- **Filter out all 1,250 points for best quality**

#### Summary Table

| Aspect | Data Shapley | Leaf Alignment |
|--------|--------------|----------------|
| **Measures** | Marginal contribution | Boundary alignment |
| **Training** | Millions of models | One model |
| **Runtime** | ~90 minutes | ~5 minutes |
| **Detects** | Individual quality | Distributional quality |
| **Best for** | Label errors, duplicates | Synthetic data, hallucinations |
| **Type** | Counterfactual | Evaluation |
| **Caught Gen2?** | ❌ No (3.39% harmful) | ✅ Yes (93.39% harmful) |
| **Use when** | Have time, need detail | Need speed, check generation |

#### Key Takeaway

**Data Shapley and Leaf Alignment are NOT competing methods.**

They answer different questions:
- **Shapley**: "Is each point individually useful?"
- **Leaf Alignment**: "Do collective patterns generalize?"

For synthetic data evaluation:
1. Start with Leaf Alignment (fast, catches distributional issues)
2. Use Shapley if needed (deep dive into point-level quality)
3. Always check Confusion Matrix first (sanity check)

**The Gen2 lesson**:

```
Individual plausibility ≠ Collective quality
```

Synthetic data can have points that look fine individually but fail collectively. Leaf alignment catches this, Shapley doesn't.

### 7.2 Leaf Alignment vs Confusion Matrix

Both methods evaluate model performance, but at different granularities.

#### What Confusion Matrix Tells You

**Aggregate Performance**:

```
                Predicted
                 0     1
Actual    0    TN    FP
          1    FN    TP

Metrics:
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 × (Prec × Rec) / (Prec + Rec)
```

**Example - Gen2 vs Real**:

```
Training Data:   Real      Gen2      Change
Precision:       18.34%    7.86%     -10.48%
Recall:          39.96%    10.49%    -29.47%  ✗✗✗
F1 Score:        25.14%    8.99%     -16.15%
```

**What it tells you**: Gen2 is bad (recall dropped 30%)

**What it DOESN'T tell you**:
- Which synthetic points are causing the problem?
- How many points are harmful vs beneficial?
- Can we filter and salvage some data?
- Why did it fail? (distributional? class-specific?)

#### What Leaf Alignment Adds

**Point-Level Diagnosis**:

```
Reliably harmful:      9,339 (93.39%)  ← 93% of points are bad
Reliably beneficial:      54 ( 0.54%)  ← Only 0.5% are good
Uncertain:               607 ( 6.07%)

Class-Specific:
  Positive: 0% beneficial  ← This explains recall drop!
```

**What it tells you**:
- 93% of synthetic data creates wrong boundaries
- Positive class is completely hallucinated (0% beneficial)
- Cannot salvage by filtering (only 0.5% usable)
- Root cause: Distributional hallucination, especially in minority class

#### Complementary Use

**Workflow**:

```
Step 1: Confusion Matrix (2 min)
  → "Is there a problem?"
  → Gen2 recall = 10% vs Real recall = 40%
  → YES, there's a problem!

Step 2: Leaf Alignment (5 min)
  → "Which points are causing it?"
  → 93% harmful, 0% positive class beneficial
  → "How bad is it?"
  → Catastrophic - not salvageable

Step 3: Decision
  → REJECT Gen2 completely
  → Need to fix generation, not filter
```

#### Summary

| Method | Question Answered | Granularity | Output |
|--------|-------------------|-------------|--------|
| **Confusion Matrix** | "Does model perform well?" | Aggregate | Metrics (Prec/Rec/F1) |
| **Leaf Alignment** | "Which points are harmful?" | Point-level | Per-point classification |

**Use together**: Confusion matrix for quick screening, Leaf alignment for diagnosis.

### 7.3 Leaf Alignment vs Distribution Metrics

Distribution metrics (KL divergence, Wasserstein distance, etc.) measure statistical similarity but don't evaluate decision boundary quality.

#### What Distribution Metrics Tell You

**Statistical Distance**:

```python
# Compare real vs synthetic feature distributions
from scipy.stats import ks_2samp

for feature in features:
    real_dist = real_data[feature]
    synth_dist = synthetic_data[feature]

    stat, pval = ks_2samp(real_dist, synth_dist)
    print(f"{feature}: KS-stat={stat:.3f}, p={pval:.3f}")
```

**Example output**:
```
AGE:         KS-stat=0.023, p=0.89  ✓ Similar
NUM_MEDS:    KS-stat=0.031, p=0.67  ✓ Similar
DIAGNOSIS:   KS-stat=0.045, p=0.34  ✓ Similar
```

**Conclusion**: Marginal distributions match! Synthetic data looks good!

**But**:

```
Leaf Alignment:
  Reliably harmful: 93.39%  ✗✗✗ Catastrophic!
```

#### The Problem: Marginals ≠ Joints

**Example**: Perfect marginals, wrong correlations

```
Real data correlations:
  AGE ↔ NUM_MEDS:       +0.67  (older → more meds)
  NUM_MEDS ↔ READMIT:   +0.43  (more meds → readmit)

Synthetic correlations:
  AGE ↔ NUM_MEDS:       +0.12  ✗ Wrong!
  NUM_MEDS ↔ READMIT:   -0.08  ✗ Wrong!
```

Marginal distributions can match perfectly while correlations are completely wrong!

#### What Leaf Alignment Catches

**Joint Pattern Evaluation**:

Leaf alignment evaluates the model's learned decision boundaries, which depend on joint distributions and feature interactions:

```
Real boundary: If (AGE > 65 AND NUM_MEDS > 10) → Readmit
Synthetic boundary: If (AGE > 65 AND NUM_MEDS > 10) → No Readmit

Same features, opposite prediction!
→ Leaf alignment detects this mismatch
```

#### When Each Method Shines

**Use Distribution Metrics When**:
- Quick sanity check of feature ranges
- Detecting obvious outliers or mode collapse
- Validating univariate feature distributions
- Fast screening (seconds)

**Use Leaf Alignment When**:
- Evaluating feature interactions and correlations
- Checking if learned patterns generalize
- Assessing decision boundary quality
- Preparing data for ML training (5 min)

#### Summary

| Aspect | Distribution Metrics | Leaf Alignment |
|--------|---------------------|----------------|
| **Measures** | Marginal distributions | Joint patterns + boundaries |
| **Can miss** | Correlation failures | Individual outliers |
| **Runtime** | Seconds | Minutes |
| **Best for** | Quick screening | ML readiness |

**The lesson**: Matching marginals ≠ good ML training data. Always validate with task-specific methods like Leaf Alignment.

### 7.4 Putting It All Together

#### Recommended Evaluation Pipeline

```
1. Distribution Metrics (30 sec)
   → Quick check: Do features look reasonable?
   → Catch obvious mode collapse or outliers

2. Confusion Matrix (2 min)
   → Aggregate check: Does performance drop?

3. Leaf Alignment (5 min)
   → Point-level diagnosis: Which points are bad?
   → Class-specific analysis: Is minority class salvageable?
   → Decision: Use, filter, or reject?

4. Data Shapley (90 min) - Optional
   → Deep dive: Why are specific points bad?
   → For real data cleaning or detailed analysis
```

#### Decision Matrix

| Distribution Check | Confusion Matrix | Leaf Alignment | Decision |
|-------------------|------------------|----------------|----------|
| ✓ Good | ✓ Good | >85% beneficial | ✅ USE |
| ✓ Good | ✓ Good | 70-85% beneficial | ✅ USE (filter) |
| ✓ Good | ✗ Bad | >70% harmful | ❌ REJECT (wrong correlations) |
| ✗ Bad | ✗ Bad | >70% harmful | ❌ REJECT (all levels fail) |
| ✓ Good | ✗ Bad | 30-70% harmful | ⚠️ Investigate Shapley |

**Key insight**: Distribution metrics can look good while Leaf Alignment detects catastrophic failures (Gen2 case).

**What's Next?**

Section 8 covers FAQs and troubleshooting for common issues encountered during leaf alignment analysis.

