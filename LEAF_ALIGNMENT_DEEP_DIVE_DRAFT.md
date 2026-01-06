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

