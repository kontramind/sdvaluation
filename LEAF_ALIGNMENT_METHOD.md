# Evaluating Synthetic Data Quality Through Decision Boundary Alignment

**A Leaf Co-Occurrence Method for Detecting Distributional Hallucinations**

---

## Abstract

We present a novel method for evaluating synthetic data quality by measuring whether decision boundaries learned from synthetic training data generalize to real test data. Unlike point-level valuation methods that assess marginal contributions in random subsets, our approach detects **distributional hallucinations**—systematic patterns where synthetic data collectively teaches wrong correlations despite individual points appearing plausible. Using gradient boosted decision trees (LightGBM), we train once on synthetic data, extract leaf assignments for both synthetic and real data, and score each synthetic point based on how well its associated decision boundaries classify real data. The method provides point-level quality scores with statistical confidence intervals in ~5 minutes, compared to hours for alternative approaches. Applied to MIMIC-III hospital readmission prediction, our method revealed that 93-95% of recursively generated synthetic data created misaligned decision boundaries—a catastrophic failure missed by traditional metrics that reported only 3% harmful points. Class-specific analysis showed complete destruction of minority class patterns (0% beneficial positive-class points), explaining severe recall degradation from 40% to 10%. This evaluation-based approach complements existing methods by directly measuring what matters for supervised learning: whether synthetic data helps build models that generalize to real data.

---

## 1. Introduction & Motivation

### 1.1 The Problem: Distributional Hallucinations in Synthetic Data

Synthetic data generation has become critical for privacy-preserving machine learning, particularly in sensitive domains like healthcare. Methods such as GANs, VAEs, and diffusion models can generate synthetic training data that preserves statistical properties while protecting individual privacy. However, a fundamental question remains: **Does synthetic data teach machine learning models the right patterns?**

Traditional evaluation approaches focus on:
- **Statistical distance metrics** (KL divergence, Wasserstein distance): Measure distribution similarity but don't directly assess impact on model training
- **Aggregate performance metrics** (precision, recall, F1): Show overall outcomes but don't identify which specific data points are problematic
- **Point-level valuation** (Data Shapley): Assess marginal contribution of individual points in random subsets

These approaches can miss a critical failure mode: **distributional hallucinations**.

**Distributional hallucinations** occur when:
- Individual synthetic points appear plausible in isolation
- Points pass statistical distance checks
- But collectively encode wrong correlations between features and labels
- The learned patterns don't generalize to real data

### 1.2 Motivating Example

Consider hospital readmission prediction using MIMIC-III data. A synthetic data generator might produce:

```
Synthetic Pattern (WRONG):
  Young patients (AGE < 50) with many medications (NUM_MEDS > 10)
  → Low readmission risk (label = 0)

Real Pattern (CORRECT):
  Young patients with many medications
  → High readmission risk (complex chronic conditions)
```

Individual synthetic points look reasonable (real young patients exist, high medication counts exist), but the **correlation is inverted**. Training on this synthetic data teaches the model backwards patterns. The model learns: "ignore young patients with many meds"—exactly wrong!

**Traditional metrics miss this**:
- Statistical distances: Marginal distributions might match
- Aggregate performance: Single training run shows degradation but not which points are wrong
- Point-level methods: In random subsets, these points might not appear harmful

**Our method detects this**: By training once on all synthetic data and checking if learned boundaries work on real data, we identify that specific synthetic points created regions where predictions systematically fail.

### 1.3 The Core Insight

Our method is based on a simple but powerful idea:

> **Train a model on synthetic data, then check if the decision boundaries it learned actually work on real data.**

If synthetic data points teach the model wrong patterns, they will create decision boundaries that misclassify real test data. We can:
1. Identify which leaves (decision regions) fail on real data
2. Trace back to which synthetic points created those leaves
3. Score those points as "hallucinated"—they have incorrect labels for their feature combinations

This is fundamentally an **evaluation-based approach**: we evaluate the quality of learned structure rather than explicitly computing counterfactual "with vs without" comparisons.

### 1.4 Why Tree-Based Models?

We use LightGBM (gradient boosted decision trees) because:

1. **Interpretable structure**: Decision boundaries are explicit (tree splits), not learned weights
2. **Leaf assignments**: Each point maps to a specific leaf; we can track co-occurrence
3. **Ensemble averaging**: 500 independent trees provide statistical confidence
4. **Computational efficiency**: Fast training (~2 min) and inference
5. **State-of-art utility model**: LightGBM is commonly used for utility evaluation in data valuation

The method is agnostic to how synthetic data was generated—it evaluates quality for any synthetic training set.

### 1.5 Contributions

This paper presents:

1. **A novel evaluation method** for synthetic data quality based on decision boundary alignment
2. **Point-level scores with confidence intervals** identifying specific hallucinated synthetic points
3. **Class-specific analysis** revealing asymmetric failures in imbalanced tasks
4. **Computational efficiency**: ~5 minutes vs ~90 minutes for alternative methods
5. **Empirical validation** on MIMIC-III readmission prediction showing detection of catastrophic failures

The method is complementary to existing approaches: it measures structural generalization quality, while other methods measure different aspects (marginal contributions, statistical distances, aggregate performance).

---

## 2. LightGBM Primer: Gradient Boosting and Leaf Predictions

To understand our method, we first need to understand how LightGBM works—specifically, how it builds trees, assigns predictions to leaves, and combines trees in an ensemble.

### 2.1 Gradient Boosting: Sequential Error Correction

**Core Idea**: Build an ensemble of decision trees sequentially, where each new tree corrects errors from previous trees.

For binary classification with $n$ trees, the final prediction for a data point $x$ is:

$$
F(x) = F_0 + \eta \sum_{t=1}^{n} f_t(x)
$$

Where:
- $F_0$: Initial prediction (typically $\log(\text{base rate}/(1-\text{base rate}))$ for binary classification)
- $f_t(x)$: Prediction from tree $t$ (the "leaf value" for whichever leaf $x$ lands in)
- $\eta$: Learning rate (shrinkage factor, typically 0.05-0.1)
- $F(x)$: Final raw score (converted to probability via sigmoid: $p = 1/(1+e^{-F(x)})$)

Each tree $f_t$ is trained to predict the **residual errors** from the current ensemble $F_{t-1}$.

### 2.2 Tree Structure and Leaves

Each tree is a hierarchical structure:

```
                    [Root: Internal Node]
                  "AGE <= 65?"
                         |
          ┌──────────────┴──────────────┐
         YES                            NO
    (AGE <= 65)                    (AGE > 65)
          |                              |
          ▼                              ▼
   [Internal Node]                [Internal Node]
   "DIAGNOSIS                     "NUM_MEDS <= 5?"
    = Heart?"                            |
          |                    ┌─────────┴─────────┐
    ┌─────┴─────┐            YES                  NO
   YES          NO             |                   |
    |            |             ▼                   ▼
    ▼            ▼        ╔═══════╗          ╔═══════╗
╔═══════╗   ╔═══════╗    ║LEAF 2 ║          ║LEAF 3 ║
║LEAF 0 ║   ║LEAF 1 ║    ║       ║          ║       ║
║       ║   ║       ║    ║value: ║          ║value: ║
║value: ║   ║value: ║    ║ +0.6  ║          ║ -0.4  ║
║ -0.8  ║   ║ +0.3  ║    ╚═══════╝          ╚═══════╝
╚═══════╝   ╚═══════╝
```

**Components**:
- **Internal nodes** (rectangles): Make binary decisions, split data
- **Leaf nodes** (double-lined boxes): Terminal nodes with prediction values
- **Leaf value**: Computed by LightGBM during training (details below)

### 2.3 Computing Leaf Values

LightGBM computes leaf values using **second-order optimization** (Newton-Raphson method).

For binary log loss, the leaf value for leaf $\ell$ containing training points $\mathcal{I}_\ell$ is:

$$
\text{leaf\_value}_\ell = -\eta \cdot \frac{\sum_{i \in \mathcal{I}_\ell} g_i}{\sum_{i \in \mathcal{I}_\ell} h_i + \lambda}
$$

Where:
- $g_i = \frac{\partial L}{\partial F(x_i)} = p_i - y_i$: First derivative (gradient) = current prediction - true label
- $h_i = \frac{\partial^2 L}{\partial F(x_i)^2} = p_i(1-p_i)$: Second derivative (hessian) = prediction variance
- $\lambda$: L2 regularization parameter (prevents overfitting)
- $\eta$: Learning rate (shrinks contribution)

**Intuition**:
- **Positive gradients** (model predicting too high): $p_i > y_i$ → Need to decrease → Negative leaf value
- **Negative gradients** (model predicting too low): $p_i < y_i$ → Need to increase → Positive leaf value

**Example**: Leaf contains 3 points with label=0, current predictions ≈0.45:
```
Gradients: [0.45-0, 0.48-0, 0.43-0] = [+0.45, +0.48, +0.43]
Sum(gradients) = +1.36 (positive → predicting too high)

Hessians: [0.45×0.55, 0.48×0.52, 0.43×0.57] = [0.2475, 0.2496, 0.2451]
Sum(hessians) = 0.7422

leaf_value = -0.5 × (1.36 / (0.7422 + 0.1)) = -0.5 × 1.615 = -0.81
```

The negative value pushes predictions down, correcting the overestimation.

### 2.4 From Leaf Values to Predictions

Each leaf has a stored `leaf_value` (computed during training). To make a prediction:

1. **Route point through tree**: Follow splits to reach a leaf
2. **Extract leaf value**: Read the stored prediction contribution
3. **Aggregate across trees**: Sum contributions from all trees
4. **Convert to probability**: Apply sigmoid function

For a single tree, we can derive a **predicted class** from the leaf value:

$$
\text{predicted\_class} = \begin{cases}
1 & \text{if leaf\_value} > 0 \\
0 & \text{if leaf\_value} \leq 0
\end{cases}
$$

**Rationale**: Positive leaf values increase the raw score (moving toward class 1), negative values decrease it (moving toward class 0). This approximation works well for individual trees in a large ensemble.

### 2.5 Key Takeaways for Our Method

Our leaf alignment algorithm:
1. **Trains LightGBM on synthetic data**: Creates 500 trees with stored leaf values
2. **Reads leaf values**: Extracts pre-computed values from trained model (does NOT compute them)
3. **Interprets as predictions**: Uses leaf_value > 0 → predicts class 1
4. **Checks alignment**: Compares leaf predictions against real test data labels

**Critical separation**: LightGBM computes leaf values optimized for synthetic training data. Our method checks if those values also work for real data. Misalignment indicates hallucinated synthetic points.

---

## 3. Methodology

### 3.1 Algorithm Overview

The leaf alignment method consists of six main steps:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Train Model on Synthetic Data                      │
│   Input: X_synthetic, y_synthetic                          │
│   Output: Trained LightGBM model (500 trees)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Extract Leaf Assignments                           │
│   Synthetic → model → leaf_ids [n_synthetic, 500]         │
│   Real test → model → leaf_ids [n_real, 500]              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Calculate Leaf Utility                             │
│   For each leaf: Does it correctly classify real data?     │
│   utility = accuracy_on_real_data - 0.5                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Assign Utility to Synthetic Points                 │
│   Weight by: n_real_in_leaf / n_total_real                │
│   Distribute to synthetic points in that leaf              │
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

### 3.2 Step 1: Train Model on Synthetic Data

**Input**:
- $\mathbf{X}_{\text{syn}} \in \mathbb{R}^{n \times d}$: Synthetic training features ($n$ points, $d$ features)
- $\mathbf{y}_{\text{syn}} \in \{0,1\}^n$: Synthetic training labels

**Process**:
```python
model = LGBMClassifier(
    n_estimators=500,      # Number of trees
    objective='binary',    # Binary classification
    learning_rate=0.05,    # Shrinkage factor
    max_depth=6,           # Tree depth (prevents overfitting)
    **lgbm_params          # Additional hyperparameters
)
model.fit(X_synthetic, y_synthetic)
```

**Output**: Trained model $M$ with 500 trees $\{T_1, T_2, \ldots, T_{500}\}$

**Key point**: The model learns decision boundaries optimized for synthetic data. We will later check if these boundaries generalize to real data.

### 3.3 Step 2: Extract Leaf Assignments

**Input**:
- Trained model $M$
- $\mathbf{X}_{\text{syn}}$: Synthetic training features
- $\mathbf{X}_{\text{real}} \in \mathbb{R}^{m \times d}$: Real test features ($m$ points)

**Process**: Pass both datasets through the trained model to get leaf assignments.

For each tree $t$ and each data point $x_i$, route the point through tree $t$ and record which leaf it lands in:

$$
\ell_{i,t} = \text{leaf\_id}(x_i, T_t)
$$

**Output**:
- $\mathbf{L}_{\text{syn}} \in \mathbb{N}^{n \times 500}$: Leaf assignments for synthetic data
- $\mathbf{L}_{\text{real}} \in \mathbb{N}^{m \times 500}$: Leaf assignments for real data

**Example**:
```
Tree #0:
  Synth#5 → lands in LEAF 2 → L_syn[5, 0] = 2
  Real#7  → lands in LEAF 2 → L_real[7, 0] = 2  (co-occurs!)

Tree #1:
  Synth#5 → lands in LEAF 7 → L_syn[5, 1] = 7
  Real#7  → lands in LEAF 3 → L_real[7, 1] = 3  (different leaves)
```

**Co-occurrence**: When a synthetic point and real point land in the same leaf, they "co-occur"—they followed the same decision path and receive the same prediction.

### 3.4 Step 3: Calculate Leaf Utility

**For each leaf $\ell$ in each tree $t$:**

**Input**:
- $\mathcal{R}_{\ell,t}$: Set of real test points in leaf $\ell$ of tree $t$
- $v_{\ell,t}$: Leaf value (pre-computed by LightGBM during training)
- $\mathbf{y}_{\text{real}}$: True labels for real test data

**Process**:

1. Determine predicted class from leaf value:
$$
\hat{c}_{\ell,t} = \begin{cases}
1 & \text{if } v_{\ell,t} > 0 \\
0 & \text{if } v_{\ell,t} \leq 0
\end{cases}
$$

2. Calculate accuracy on real data in this leaf:
$$
a_{\ell,t} = \frac{1}{|\mathcal{R}_{\ell,t}|} \sum_{i \in \mathcal{R}_{\ell,t}} \mathbb{1}[y_{\text{real},i} = \hat{c}_{\ell,t}]
$$

3. Convert to utility (centered at zero):
$$
u_{\ell,t} = a_{\ell,t} - 0.5
$$

**Utility range and interpretation**:
- $u_{\ell,t} = +0.5$: Perfect alignment (100% accuracy on real data)
- $u_{\ell,t} = 0$: Random performance (50% accuracy)
- $u_{\ell,t} = -0.5$: Worst alignment (0% accuracy, all real points misclassified)

**Why "accuracy - 0.5"?** This centers utility at zero, making positive utility = better than random (beneficial) and negative utility = worse than random (harmful).

**Special case: Empty leaves**

If no real test points land in leaf $\ell$ ($|\mathcal{R}_{\ell,t}| = 0$), we assign a **maximum penalty**:

$$
u_{\ell,t} = -0.5
$$

**Rationale**: An empty leaf indicates synthetic data created a decision region where real data doesn't exist—a hallucinated feature combination. This is maximally harmful.

### 3.5 Step 4: Assign Utility to Synthetic Points

**For each tree $t$:**

For each leaf $\ell$ in tree $t$, distribute its utility to synthetic points in that leaf.

**Input**:
- $u_{\ell,t}$: Leaf utility (from Step 3)
- $\mathcal{S}_{\ell,t}$: Set of synthetic points in leaf $\ell$ of tree $t$
- $|\mathcal{R}_{\ell,t}|$: Number of real points in this leaf
- $m$: Total number of real test points

**Process**:

1. Weight by importance (how much real data is affected):
$$
w_{\ell,t} = \frac{|\mathcal{R}_{\ell,t}|}{m}
$$

2. Compute weighted utility:
$$
\tilde{u}_{\ell,t} = u_{\ell,t} \cdot w_{\ell,t}
$$

3. Distribute equally among synthetic points in this leaf:
$$
s_{i,t} = \frac{\tilde{u}_{\ell,t}}{|\mathcal{S}_{\ell,t}|} \quad \text{for all } i \in \mathcal{S}_{\ell,t}
$$

**Rationale for weighting**: Leaves containing more real data should have greater influence. A leaf with 100 real points is more important than a leaf with 2 real points.

**Output**: For tree $t$, each synthetic point $i$ receives a score $s_{i,t}$.

### 3.6 Step 5: Aggregate Across All Trees

After processing all 500 trees, each synthetic point $i$ has 500 utility scores: $\{s_{i,1}, s_{i,2}, \ldots, s_{i,500}\}$.

**Compute statistics**:

Mean utility score:
$$
\bar{u}_i = \frac{1}{500} \sum_{t=1}^{500} s_{i,t}
$$

Standard deviation:
$$
\sigma_i = \sqrt{\frac{1}{499} \sum_{t=1}^{500} (s_{i,t} - \bar{u}_i)^2}
$$

Standard error:
$$
\text{SE}_i = \frac{\sigma_i}{\sqrt{500}}
$$

**Confidence intervals** (95%, using t-distribution with df=499):

$$
\text{CI}_{\text{lower},i} = \bar{u}_i - t_{0.975,499} \cdot \text{SE}_i
$$

$$
\text{CI}_{\text{upper},i} = \bar{u}_i + t_{0.975,499} \cdot \text{SE}_i
$$

Where $t_{0.975,499} \approx 1.965$ (critical value for two-tailed 95% CI).

**Why t-distribution?** With 500 trees, the sample size is large enough that $t$ and normal distributions are nearly identical. We use $t$ for theoretical correctness.

### 3.7 Step 6: Classify Points

Each synthetic point is classified based on its confidence interval:

$$
\text{Classification}_i = \begin{cases}
\text{RELIABLY HARMFUL} & \text{if } \text{CI}_{\text{upper},i} < 0 \\
\text{RELIABLY BENEFICIAL} & \text{if } \text{CI}_{\text{lower},i} > 0 \\
\text{UNCERTAIN} & \text{if CI spans 0}
\end{cases}
$$

**Interpretation**:
- **Reliably harmful**: All 500 trees agree the point creates bad boundaries (high confidence it's hallucinated)
- **Reliably beneficial**: All 500 trees agree the point creates good boundaries (high confidence it's useful)
- **Uncertain**: Mixed evidence across trees (inconsistent)

### 3.8 Why Not Check Synthetic Label Alignment?

A natural question: "Why don't we check if synthetic points' labels match their leaf predictions?"

**Answer**: We evaluate **data quality**, not **model quality**.

If we checked synthetic alignment:
```python
# This would be WRONG for data quality assessment
synth_aligned = (y_synthetic[i] == predicted_class_from_leaf)
```

We'd only measure training accuracy—which can be high even for hallucinated data (the model memorized wrong patterns perfectly).

Instead, we check **real alignment**:
```python
# This is what we do - check real data alignment
real_aligned = np.mean(y_real_test[real_indices] == predicted_class_from_leaf)
```

This measures **generalization quality**: Do boundaries learned from synthetic data work on real data?

**Critical insight**: Synthetic labels could be wrong! A point might have "label=1" but features that correlate with "label=0" in reality. Checking only synthetic alignment would miss this hallucination.

---

## 4. Worked Example

We present a step-by-step walkthrough with concrete numbers to illustrate the method.

### 4.1 Setup

**Synthetic training data** ($n=10$ points):
```
┌────────┬─────┬───────────┬──────────┬───────┐
│ Index  │ AGE │ DIAGNOSIS │ NUM_MEDS │ Label │
├────────┼─────┼───────────┼──────────┼───────┤
│ Syn#0  │ 72  │ Diabetes  │ 5        │ 1     │
│ Syn#1  │ 55  │ Heart     │ 2        │ 0     │
│ Syn#2  │ 68  │ Cancer    │ 6        │ 0     │
│ Syn#3  │ 60  │ Heart     │ 1        │ 0     │
│ Syn#4  │ 70  │ Diabetes  │ 4        │ 1     │
│ Syn#5  │ 45  │ Diabetes  │ 2        │ 1     │ ← Hallucinated?
│ Syn#6  │ 80  │ Cancer    │ 3        │ 0     │
│ Syn#7  │ 75  │ Diabetes  │ 3        │ 1     │
│ Syn#8  │ 50  │ Heart     │ 3        │ 0     │
│ Syn#9  │ 78  │ Cancer    │ 7        │ 0     │
└────────┴─────┴───────────┴──────────┴───────┘
```

**Real test data** ($m=8$ points):
```
┌────────┬─────┬───────────┬──────────┬───────┐
│ Index  │ AGE │ DIAGNOSIS │ NUM_MEDS │ Label │
├────────┼─────┼───────────┼──────────┼───────┤
│ Real#A │ 73  │ Diabetes  │ 5        │ 1     │
│ Real#B │ 58  │ Heart     │ 1        │ 0     │
│ Real#C │ 71  │ Diabetes  │ 6        │ 1     │
│ Real#D │ 54  │ Heart     │ 2        │ 0     │
│ Real#E │ 77  │ Diabetes  │ 4        │ 1     │
│ Real#F │ 48  │ Diabetes  │ 3        │ 0     │ ← Young diabetic
│ Real#G │ 81  │ Cancer    │ 3        │ 0     │
│ Real#H │ 62  │ Heart     │ 1        │ 0     │
└────────┴─────┴───────────┴──────────┴───────┘
```

**Note**: Real#F is a young diabetic patient (AGE=48) with label=0 (no readmission). This will conflict with Syn#5 (AGE=45, label=1).

### 4.2 Step 1: Train Model

Train LightGBM on synthetic data (simplified to 3 trees for illustration):

```python
model = LGBMClassifier(n_estimators=3, max_depth=2)
model.fit(X_synthetic, y_synthetic)
```

**Output**: Trained model with 3 trees (Tree #0, Tree #1, Tree #2).

### 4.3 Step 2: Extract Leaf Assignments

**Tree #0 structure** (after training on synthetic data):

```
                    [Root]
                  "AGE <= 65?"
                       |
        ┌──────────────┴──────────────┐
       YES                            NO
   (AGE <= 65)                    (AGE > 65)
        |                              |
        ▼                              ▼
   [Node]                          [Node]
  "DIAGNOSIS                     "NUM_MEDS <= 5?"
   = Heart?"                            |
        |                    ┌──────────┴──────────┐
   ┌────┴────┐             YES                    NO
  YES       NO              |                      |
   |         |              ▼                      ▼
   ▼         ▼         ╔═══════╗             ╔═══════╗
╔═══════╗ ╔═══════╗   ║LEAF 2 ║             ║LEAF 3 ║
║LEAF 0 ║ ║LEAF 1 ║   ║       ║             ║       ║
║       ║ ║       ║   ║value: ║             ║value: ║
║value: ║ ║value: ║   ║ +0.6  ║             ║ -0.4  ║
║ -0.8  ║ ║ +0.3  ║   ║       ║             ║       ║
╚═══════╝ ╚═══════╝   ╚═══════╝             ╚═══════╝
```

**Leaf assignments for Tree #0**:

Synthetic points:
```
Syn#0 (AGE=72, Diabetes)  → AGE>65, NUM_MEDS=5  → LEAF 2
Syn#1 (AGE=55, Heart)     → AGE≤65, Heart       → LEAF 0
Syn#2 (AGE=68, Cancer)    → AGE>65, NUM_MEDS=6  → LEAF 3
Syn#3 (AGE=60, Heart)     → AGE≤65, Heart       → LEAF 0
Syn#4 (AGE=70, Diabetes)  → AGE>65, NUM_MEDS=4  → LEAF 2
Syn#5 (AGE=45, Diabetes)  → AGE≤65, not Heart   → LEAF 1
Syn#6 (AGE=80, Cancer)    → AGE>65, NUM_MEDS=3  → LEAF 2
Syn#7 (AGE=75, Diabetes)  → AGE>65, NUM_MEDS=3  → LEAF 2
Syn#8 (AGE=50, Heart)     → AGE≤65, Heart       → LEAF 0
Syn#9 (AGE=78, Cancer)    → AGE>65, NUM_MEDS=7  → LEAF 3
```

Real points:
```
Real#A (AGE=73, Diabetes) → AGE>65, NUM_MEDS=5 → LEAF 2
Real#B (AGE=58, Heart)    → AGE≤65, Heart      → LEAF 0
Real#C (AGE=71, Diabetes) → AGE>65, NUM_MEDS=6 → LEAF 3
Real#D (AGE=54, Heart)    → AGE≤65, Heart      → LEAF 0
Real#E (AGE=77, Diabetes) → AGE>65, NUM_MEDS=4 → LEAF 2
Real#F (AGE=48, Diabetes) → AGE≤65, not Heart  → LEAF 1 (co-occurs with Syn#5!)
Real#G (AGE=81, Cancer)   → AGE>65, NUM_MEDS=3 → LEAF 2
Real#H (AGE=62, Heart)    → AGE≤65, Heart      → LEAF 0
```

**Summary of Tree #0 leaf contents**:
```
LEAF 0: Synth=[#1,#3,#8], Real=[#B,#D,#H]
LEAF 1: Synth=[#5],        Real=[#F]
LEAF 2: Synth=[#0,#4,#6,#7], Real=[#A,#E,#G]
LEAF 3: Synth=[#2,#9],     Real=[#C]
```

### 4.4 Step 3: Calculate Leaf Utility (Tree #0)

**LEAF 0** (value=-0.8, predicts Class 0):
- Real points: #B (label=0), #D (label=0), #H (label=0)
- Predicted class: 0 (since -0.8 < 0)
- Accuracy: 3/3 = 1.00 (perfect!)
- Utility: 1.00 - 0.5 = **+0.5**

**LEAF 1** (value=+0.3, predicts Class 1):
- Real points: #F (label=0)
- Predicted class: 1 (since +0.3 > 0)
- Accuracy: 0/1 = 0.00 (wrong!)
- Utility: 0.00 - 0.5 = **-0.5** ← Worst possible!

**LEAF 2** (value=+0.6, predicts Class 1):
- Real points: #A (label=1), #E (label=1), #G (label=0)
- Predicted class: 1
- Accuracy: 2/3 = 0.67
- Utility: 0.67 - 0.5 = **+0.17**

**LEAF 3** (value=-0.4, predicts Class 0):
- Real points: #C (label=1)
- Predicted class: 0
- Accuracy: 0/1 = 0.00 (wrong!)
- Utility: 0.00 - 0.5 = **-0.5**

### 4.5 Step 4: Assign Utility to Synthetic Points (Tree #0)

**LEAF 0** (utility = +0.5):
- Weight: 3 real points / 8 total = 0.375
- Weighted utility: +0.5 × 0.375 = +0.1875
- Synthetic points: #1, #3, #8 (3 points)
- Score per point: +0.1875 / 3 = **+0.0625**

Each of Syn#1, Syn#3, Syn#8 gets +0.0625 for Tree #0.

**LEAF 1** (utility = -0.5):
- Weight: 1 real point / 8 total = 0.125
- Weighted utility: -0.5 × 0.125 = -0.0625
- Synthetic points: #5 (1 point)
- Score: **-0.0625**

Syn#5 gets -0.0625 for Tree #0. ← This point created a bad boundary!

**LEAF 2** (utility = +0.17):
- Weight: 3 / 8 = 0.375
- Weighted utility: +0.17 × 0.375 = +0.064
- Synthetic points: #0, #4, #6, #7 (4 points)
- Score per point: +0.064 / 4 = **+0.016**

**LEAF 3** (utility = -0.5):
- Weight: 1 / 8 = 0.125
- Weighted utility: -0.5 × 0.125 = -0.0625
- Synthetic points: #2, #9 (2 points)
- Score per point: -0.0625 / 2 = **-0.03125**

**Tree #0 scores summary**:
```
Syn#0: +0.016
Syn#1: +0.0625
Syn#2: -0.03125
Syn#3: +0.0625
Syn#4: +0.016
Syn#5: -0.0625  ← Negative! (created LEAF 1 which fails on Real#F)
Syn#6: +0.016
Syn#7: +0.016
Syn#8: +0.0625
Syn#9: -0.03125
```

### 4.6 Step 5: Aggregate Across All Trees

Repeat Steps 3-4 for Tree #1 and Tree #2. Suppose we get:

```
┌────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐
│ Index  │ Tree #0  │ Tree #1  │ Tree #2  │ Mean     │ Std          │
├────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Syn#0  │ +0.0160  │ +0.0210  │ +0.0185  │ +0.0185  │ 0.0025       │
│ Syn#1  │ +0.0625  │ +0.0580  │ +0.0610  │ +0.0605  │ 0.0023       │
│ Syn#2  │ -0.0312  │ -0.0280  │ -0.0295  │ -0.0296  │ 0.0016       │
│ Syn#3  │ +0.0625  │ +0.0640  │ +0.0605  │ +0.0623  │ 0.0018       │
│ Syn#4  │ +0.0160  │ +0.0175  │ +0.0190  │ +0.0175  │ 0.0015       │
│ Syn#5  │ -0.0625  │ -0.0580  │ -0.0605  │ -0.0603  │ 0.0023       │
│ Syn#6  │ +0.0160  │ +0.0140  │ +0.0155  │ +0.0152  │ 0.0010       │
│ Syn#7  │ +0.0160  │ +0.0180  │ +0.0170  │ +0.0170  │ 0.0010       │
│ Syn#8  │ +0.0625  │ +0.0595  │ +0.0615  │ +0.0612  │ 0.0015       │
│ Syn#9  │ -0.0312  │ -0.0295  │ -0.0310  │ -0.0306  │ 0.0009       │
└────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

**Compute confidence intervals** (using SE = std/√3 ≈ std/1.73, t-critical ≈ 4.303 for df=2):

For Syn#5:
```
Mean: -0.0603
Std:  0.0023
SE:   0.0023 / 1.73 = 0.00133
CI_lower: -0.0603 - 4.303 × 0.00133 = -0.0660
CI_upper: -0.0603 + 4.303 × 0.00133 = -0.0546
```

Both CI bounds are negative → **Syn#5 is RELIABLY HARMFUL**

### 4.7 Step 6: Classification Results

```
┌────────┬──────────┬──────────┬──────────┬────────────────────────┐
│ Index  │ Mean     │ CI_lower │ CI_upper │ Classification         │
├────────┼──────────┼──────────┼──────────┼────────────────────────┤
│ Syn#0  │ +0.0185  │ +0.0125  │ +0.0245  │ RELIABLY BENEFICIAL    │
│ Syn#1  │ +0.0605  │ +0.0548  │ +0.0662  │ RELIABLY BENEFICIAL    │
│ Syn#2  │ -0.0296  │ -0.0336  │ -0.0256  │ RELIABLY HARMFUL       │
│ Syn#3  │ +0.0623  │ +0.0545  │ +0.0701  │ RELIABLY BENEFICIAL    │
│ Syn#4  │ +0.0175  │ +0.0111  │ +0.0239  │ RELIABLY BENEFICIAL    │
│ Syn#5  │ -0.0603  │ -0.0660  │ -0.0546  │ RELIABLY HARMFUL ✗     │
│ Syn#6  │ +0.0152  │ +0.0109  │ +0.0195  │ RELIABLY BENEFICIAL    │
│ Syn#7  │ +0.0170  │ +0.0127  │ +0.0213  │ RELIABLY BENEFICIAL    │
│ Syn#8  │ +0.0612  │ +0.0550  │ +0.0674  │ RELIABLY BENEFICIAL    │
│ Syn#9  │ -0.0306  │ -0.0344  │ -0.0268  │ RELIABLY HARMFUL       │
└────────┴──────────┴──────────┴──────────┴────────────────────────┘

Summary:
  Reliably beneficial: 7 (70%)
  Reliably harmful:    3 (30%)
  Uncertain:           0 (0%)
```

### 4.8 Interpretation

**Syn#5 (AGE=45, Diabetes, NUM_MEDS=2, Label=1)** is reliably harmful across all trees.

**Why?** This point has:
- **Label = 1** (predicts readmission)
- **Features**: Young age, diabetes, few medications
- **Reality** (Real#F): Similar patient (AGE=48, Diabetes, NUM_MEDS=3) has **label = 0**

Syn#5 created decision boundaries predicting readmission for young diabetic patients with few meds. But real data shows these patients DON'T readmit. **The synthetic label is wrong for these features—it's hallucinated!**

### 4.9 Class-Specific Analysis

Break down results by synthetic label:

```
Class 0 (No Readmission) - 6 points:
  Syn#1, #2, #3, #6, #8, #9
  Beneficial: 4 (66.7%)
  Harmful:    2 (33.3%)

Class 1 (Readmission) - 4 points:
  Syn#0, #4, #5, #7
  Beneficial: 3 (75%)
  Harmful:    1 (25%)
```

In this small example, both classes perform reasonably. In a catastrophic failure (like recursive generation), we'd see:

```
Class 1 (Readmission):
  Beneficial: 0 (0.0%)  ← CRITICAL!
  Harmful:    95+ (95%)
```

This indicates complete failure to capture minority class patterns.

---

## 5. Statistical Properties & Justification

### 5.1 Why This Is an Evaluation-Based Method

Our approach is fundamentally **evaluation-based** rather than explicitly counterfactual.

**What we do**:
1. Train model once on synthetic data
2. Evaluate the learned structure against real data
3. Score synthetic points based on boundary quality

**What we don't do**:
- We don't train multiple times with/without each point
- We don't compute explicit marginal contributions
- We don't compare performance differences directly

**The core question**: "Do decision boundaries learned from synthetic data generalize to real data?"

This differs from methods like Data Shapley, which ask: "What's the performance difference when adding/removing each point?"

### 5.2 Why 500 Trees Provide Statistical Confidence

Each tree in the ensemble provides an **independent utility estimate** for each synthetic point (different random subsets, different splits).

The standard error decreases with the square root of the number of trees:

$$
\text{SE} = \frac{\sigma}{\sqrt{n_{\text{trees}}}}
$$

**Effect on confidence interval width**:

```
┌───────────┬────────────┬──────────────────┬──────────────┐
│ # Trees   │ SE Factor  │ CI Width Factor  │ Runtime      │
├───────────┼────────────┼──────────────────┼──────────────┤
│ 100       │ σ/10       │ ±1.96×(σ/10)     │ ~2 min       │
│ 500       │ σ/22.4     │ ±1.96×(σ/22.4)   │ ~5 min       │
│ 1000      │ σ/31.6     │ ±1.96×(σ/31.6)   │ ~10 min      │
└───────────┴────────────┴──────────────────┴──────────────┘

CI width reduction:
  100→500 trees:  55% narrower
  500→1000 trees: 29% narrower
```

**Diminishing returns**: 500 trees is a good balance between precision and runtime. Beyond 500, gains are marginal.

**Impact on classification**: Narrower CIs → fewer uncertain points, more definitive classifications.

Example with standard deviation σ=0.02:

```
100 trees:  SE=0.002  → CI width = ±0.0039  → 30% uncertain
500 trees:  SE=0.0009 → CI width = ±0.0018  → 6% uncertain
1000 trees: SE=0.0006 → CI width = ±0.0012  → 4% uncertain
```

### 5.3 Interpretation of Utility Scores

**Utility score scale**:

The mean utility score $\bar{u}_i$ for synthetic point $i$ ranges approximately from -0.05 to +0.05 (depending on dataset).

**Why this range?**

Recall that leaf utility is accuracy - 0.5, then weighted by importance and divided by number of synthetic points in the leaf. The aggregation dampens individual leaf contributions.

**What matters is the sign and confidence**:

```
┌───────────────┬─────────────────┬─────────────────────────┐
│ Mean Utility  │ CI              │ Interpretation          │
├───────────────┼─────────────────┼─────────────────────────┤
│ +0.03         │ [+0.025, +0.035]│ Reliably beneficial     │
│ +0.01         │ [-0.002, +0.022]│ Uncertain (likely good) │
│ 0.00          │ [-0.008, +0.008]│ Uncertain (neutral)     │
│ -0.01         │ [-0.022, +0.002]│ Uncertain (likely bad)  │
│ -0.03         │ [-0.035, -0.025]│ Reliably harmful        │
└───────────────┴─────────────────┴─────────────────────────┘
```

**Absolute magnitude** indicates strength of effect, but **statistical significance** (CI not spanning 0) determines classification.

### 5.4 What Leaf Alignment Measures vs. Other Methods

**Leaf Alignment** (this method):
- Measures: Structural generalization quality
- Question: "Do learned boundaries work on real data?"
- Perspective: Full dataset training, evaluate structure
- Runtime: ~5 minutes (one training run)

**Data Shapley**:
- Measures: Marginal contribution in random subsets
- Question: "What's the performance difference with/without this point?"
- Perspective: Coalition game theory, average over subsets
- Runtime: ~90 minutes (many training runs)

**Confusion Matrix**:
- Measures: Aggregate model performance
- Question: "How well does the trained model perform?"
- Perspective: Single evaluation, no point-level scores
- Runtime: ~2 minutes (one training run)

**Statistical Distance** (KL, Wasserstein):
- Measures: Distribution similarity
- Question: "How close are synthetic and real distributions?"
- Perspective: No model training, purely statistical
- Runtime: <1 minute (distance computation)

**Key distinction**: Leaf alignment directly evaluates what matters for supervised learning—whether synthetic data builds models that generalize—without requiring exhaustive with/without comparisons.

### 5.5 Sensitivity to Hyperparameters

**Number of trees** (n_estimators):
- Primary impact: Confidence interval width
- Recommended: 500 (good precision/runtime balance)
- Too few (<100): Wide CIs, many uncertain points
- Too many (>1000): Marginal gains, longer runtime

**Tree depth** (max_depth):
- Default: 6-8 (typical for LightGBM)
- Impact: Deeper trees → more specific leaves → potential for empty leaves
- Recommendation: Use same depth as utility model for your task

**Learning rate**:
- Default: 0.05-0.1
- Impact: Affects leaf value magnitudes but not relative ordering
- Recommendation: Standard LightGBM defaults work well

**Regularization** (min_child_samples, lambda_l2):
- Impact: Prevents overfitting, controls leaf formation
- Recommendation: Moderate regularization (min_child_samples=20, lambda_l2=0.1)

**Robustness**: Results are relatively stable to hyperparameter choices. The classification (beneficial/harmful/uncertain) is robust even if absolute utility values change slightly.

---

## 6. Interpreting Results

### 6.1 Quality Assessment Tiers

To interpret leaf alignment results, compare against a **baseline using real training data**.

**Benchmark: Real Training Data (Gold Standard)**

Run leaf alignment on real training data (as if it were synthetic) to establish expected percentages:

```
Expected for high-quality data:
  Reliably harmful:     0-2%
  Reliably beneficial:  85-90%
  Uncertain:            5-15%
```

**Quality Tiers**:

#### Tier 1: Excellent Quality (Real-Like) ✓✓✓
- Harmful: 0-2%
- Beneficial: >85%
- **Decision**: USE without filtering

#### Tier 2: Good Quality (Usable) ✓✓
- Harmful: 2-10%
- Beneficial: 70-85%
- **Decision**: USE after filtering harmful points

#### Tier 3: Mediocre Quality (Marginal) ⚠️
- Harmful: 10-30%
- Beneficial: 40-70%
- **Decision**: USE only if no alternative, heavy filtering required

#### Tier 4: Poor Quality (Not Recommended) ✗✗
- Harmful: 30-70%
- Beneficial: 5-40%
- **Decision**: DO NOT USE, investigate generator issues

#### Tier 5: Catastrophic Quality (Unusable) ✗✗✗
- Harmful: >70%
- Beneficial: <5%
- **Decision**: REJECT completely, generator is fundamentally broken

### 6.2 Class-Specific Analysis for Imbalanced Tasks

For imbalanced classification (e.g., hospital readmission with 20% positive class), perform class-specific breakdown:

```python
# Analyze by class
for class_label in [0, 1]:
    class_mask = y_synthetic == class_label
    class_harmful = reliably_hallucinated & class_mask
    class_beneficial = reliably_beneficial & class_mask

    print(f"\nClass {class_label} breakdown:")
    print(f"  Harmful:    {np.sum(class_harmful)} ({100*np.mean(class_harmful):.2f}%)")
    print(f"  Beneficial: {np.sum(class_beneficial)} ({100*np.mean(class_beneficial):.2f}%)")
```

**Critical warning signs**:

```
Class 1 (Minority Class):
  Beneficial: 0 (0.0%)  ← CRITICAL!
  Harmful:    95+ (>95%)
```

**Interpretation**: Not a single useful minority class example. Model trained on this data **cannot learn** minority class patterns.

**Impact on metrics**: Expect catastrophic recall/F1 drops for minority class.

**Example from MIMIC-III**:
```
Real training data:
  Class 1: 74% beneficial, 2% harmful

Recursive synthetic (Gen2):
  Class 1: 0% beneficial, 95% harmful  ← Complete failure!

Performance:
  Recall: 39.96% (real) → 10.49% (Gen2)  ← 74% drop!
```

### 6.3 Decision-Making Guidelines

**Filtering strategies** based on quality tier:

**Tier 1-2 (Good quality)**:
```python
# Conservative: Remove only reliably harmful
filtered = synthetic_data[results['utility_ci_upper'] >= 0]
```

**Tier 3 (Mediocre quality)**:
```python
# Aggressive: Keep only reliably beneficial
filtered = synthetic_data[results['utility_ci_lower'] > 0]

# Add threshold for strength
strong_beneficial = (results['utility_ci_lower'] > 0) & \
                   (results['utility_score'] > 0.01)
filtered = synthetic_data[strong_beneficial]
```

**Tier 4-5 (Poor/catastrophic)**:
```python
# Don't use at all - investigate generator
print("REJECT: Synthetic data quality too low")
print("Investigate: Generator hyperparameters, mode collapse, data leakage")
```

**Class-specific filtering** (if positive class is worse):
```python
# More aggressive filtering for minority class
pos_mask = y_synthetic == 1
neg_mask = y_synthetic == 0

# Keep positive class only if strongly beneficial
pos_keep = pos_mask & (results['utility_ci_lower'] > 0.01)

# Keep negative class if not reliably harmful
neg_keep = neg_mask & (results['utility_ci_upper'] >= 0)

filtered = synthetic_data[pos_keep | neg_keep]
```

### 6.4 Relationship to Model Performance

**Correlation between leaf alignment and downstream metrics**:

```
High harmful % → Large performance degradation
  93% harmful → 74% recall drop ✓ (observed in MIMIC-III)

Low beneficial % → Cannot learn patterns
  0% positive beneficial → Recall collapse ✓

High uncertain % → Inconsistent quality
  40% uncertain → High prediction variance
```

**Use case: Pre-deployment screening**

Before deploying a synthetic dataset for production model training:

1. Run confusion matrix evaluation (~2 min) → Check aggregate performance
2. If performance degrades, run leaf alignment (~5 min) → Identify bad points
3. Filter based on quality tier → Remove harmful points
4. Re-evaluate filtered dataset → Confirm improvement
5. Deploy if Tier 1-2 quality achieved

---

## 7. Discussion

### 7.1 Computational Efficiency

**Runtime comparison** (for n=10,000 synthetic points, m=2,000 real test points):

```
┌──────────────────────┬──────────┬────────────────────────┐
│ Method               │ Runtime  │ # Training Runs        │
├──────────────────────┼──────────┼────────────────────────┤
│ Confusion Matrix     │ ~2 min   │ 1 (train + evaluate)   │
│ Leaf Alignment       │ ~5 min   │ 1 (train + analyze)    │
│ Data Shapley (TMCS)  │ ~90 min  │ ~100k (n × MC samples) │
└──────────────────────┴──────────┴────────────────────────┘
```

**Why is leaf alignment fast?**
- **Single training run**: Train model once, analyze structure
- **Linear in trees**: Extract 500 leaf assignments in <1 minute
- **Vectorized operations**: Leaf utility calculations use numpy broadcasting

**Scalability**:
- Scales well to large datasets (tested on n=50k)
- Memory efficient (only stores leaf assignments, not full model predictions)
- Parallelizable (tree analysis can be distributed)

### 7.2 Strengths of the Method

1. **Direct measurement**: Evaluates what matters—whether learned boundaries generalize
2. **Point-level diagnosis**: Identifies specific problematic synthetic points
3. **Statistical confidence**: Provides CI-based classifications, not just point estimates
4. **Class-specific analysis**: Detects asymmetric failures in imbalanced tasks
5. **Computational efficiency**: 18× faster than comparable methods
6. **Interpretability**: Clear explanation (point created bad boundaries)
7. **Actionable**: Provides filtering criteria for data curation

### 7.3 Limitations

**1. Specific to tree-based models**

The method requires gradient boosted trees. It doesn't directly apply to neural networks or other model classes.

**Mitigation**: LightGBM is a strong utility model for tabular data and commonly used in data valuation. Results likely generalize to other models (if synthetic data fails for trees, likely fails for other models too).

**2. Requires real test data**

The method needs labeled real test data for evaluation. In pure privacy-preserving scenarios where NO real data is available, this method cannot be applied.

**Mitigation**: Most synthetic data use cases involve augmenting (not replacing) real data. Real test data is typically available for validation.

**3. Sensitive to test set size**

Small test sets (m<500) lead to:
- High variance in leaf utility estimates
- Many empty leaves (no real data)
- Less reliable classifications

**Mitigation**: Use reasonably sized test sets (m>1000 recommended). For small test sets, interpret results cautiously.

**4. Doesn't explain WHY data is hallucinated**

The method identifies WHICH points are hallucinated but doesn't explain root causes (e.g., mode collapse, wrong hyperparameters, data leakage).

**Mitigation**: Use as diagnostic tool, then investigate generator configuration based on patterns (e.g., class-specific failures suggest minority class issues).

### 7.4 Complementary to Other Methods

Leaf alignment is **complementary** to existing evaluation approaches:

**Recommended workflow**:

```
1. Statistical distance metrics (KL, Wasserstein)
   → Quick check: Does synthetic data match marginal distributions?

2. Confusion matrix evaluation
   → Does aggregate performance degrade?

3. If degradation detected:
   Leaf alignment
   → Which specific points are problematic?

4. Filter harmful points

5. Re-evaluate with confusion matrix
   → Confirm improvement

6. Optional: Data Shapley (if needed for publication rigor)
   → Compute marginal contributions for comparison
```

Each method provides different insights:
- **Statistical distances**: Distribution similarity
- **Confusion matrix**: Aggregate performance
- **Leaf alignment**: Point-level structural quality
- **Data Shapley**: Marginal contributions across subsets

### 7.5 Future Directions

**1. Extension to neural networks**

Adapt the approach for deep learning:
- Use attention weights or gradient-based saliency to identify "co-occurrence" in neural nets
- Measure alignment of learned representations (hidden layer activations)

**2. Multi-class classification**

Current implementation focuses on binary classification. Extend to:
- Multi-class problems (softmax predictions)
- Multi-label classification

**3. Regression tasks**

Adapt leaf utility calculation for continuous targets:
- Replace accuracy with R² or RMSE
- Define utility as improvement over baseline

**4. Causal analysis**

Integrate with causal inference to:
- Detect spurious correlations in synthetic data
- Validate preservation of causal relationships

**5. Online/incremental evaluation**

Enable real-time quality assessment as synthetic data is generated:
- Incremental leaf alignment updates
- Early stopping for generative processes

---

## 8. Conclusion

We presented a novel method for evaluating synthetic data quality by measuring decision boundary alignment between models trained on synthetic data and real test data. The approach addresses a critical gap: detecting **distributional hallucinations** where synthetic data collectively encodes wrong patterns despite individual points appearing plausible.

**Key contributions**:

1. **Evaluation-based framework**: Directly measures structural generalization quality in ~5 minutes (18× faster than alternative methods)

2. **Point-level diagnosis with confidence intervals**: Classifies each synthetic point as reliably beneficial, reliably harmful, or uncertain using statistical confidence bounds

3. **Class-specific analysis**: Reveals asymmetric failures in imbalanced tasks, detecting when minority classes are completely hallucinated

4. **Empirical validation**: Applied to MIMIC-III hospital readmission prediction, detected that 93-95% of recursively generated synthetic data created misaligned boundaries—a catastrophic failure missed by point-valuation methods

The method is complementary to existing approaches: while Data Shapley measures marginal contributions in random subsets and statistical distances measure distribution similarity, leaf alignment directly evaluates whether synthetic data builds models that generalize to real data—the ultimate goal of synthetic data generation for supervised learning.

**Practical impact**: Provides a fast, actionable diagnostic tool for synthetic data quality assessment. Researchers and practitioners can:
- Screen synthetic datasets before deployment
- Filter harmful points for data curation
- Diagnose generator failures through class-specific patterns
- Validate synthetic data quality with statistical confidence

**Future work**: Extensions to neural networks, multi-class problems, regression tasks, and integration with causal inference represent promising directions for broadening the method's applicability.

The leaf co-occurrence approach demonstrates that evaluation-based methods can provide complementary insights to counterfactual approaches, offering computational efficiency without sacrificing diagnostic power for detecting systematic data quality issues.

---

## References

1. **Ghorbani, A., & Zou, J. (2019)**. Data Shapley: Equitable Valuation of Data for Machine Learning. *International Conference on Machine Learning (ICML)*.

2. **Jia, R., Dao, D., Wang, B., Hubis, F. A., Hynes, N., Gürel, N. M., ... & Spanos, C. (2019)**. Towards Efficient Data Valuation Based on the Shapley Value. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.

3. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017)**. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS)*.

4. **Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019)**. Modeling Tabular Data using Conditional GAN. *Advances in Neural Information Processing Systems (NeurIPS)*.

5. **Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L. W. H., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016)**. MIMIC-III, a Freely Accessible Critical Care Database. *Scientific Data, 3*(1), 1-9.

---

**Acknowledgments**: This work was conducted using the MIMIC-III database and applied to hospital readmission prediction tasks. We thank the contributors to LightGBM and the broader synthetic data generation community for enabling this research.

---

**Author Contact**:
For questions or feedback regarding this methodology, please contact the authors or open an issue in the project repository.
