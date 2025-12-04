# sdvaluation

Data Shapley valuation for synthetic data quality assessment.

## Overview

**sdvaluation** is a Python package that computes Data Shapley values to quantify the contribution of individual synthetic data points to machine learning model performance. Using the Truncated Monte Carlo Shapley (TMCS) algorithm, it efficiently estimates how valuable (or harmful) each training sample is when evaluated against real test data.

**Key Features:**
- **Data Shapley valuation**: Principled game-theoretic approach to data valuation
- **TMCS algorithm**: Efficient Monte Carlo estimation with optional early truncation
- **LightGBM utility model**: Fast gradient boosting for performance evaluation
- **RDT encoding**: Automatic handling of mixed data types (categorical, numerical, datetime)
- **Confidence intervals**: Statistical bounds for value estimates
- **CLI and Python API**: Flexible usage for scripts or integration

This package is specifically designed for evaluating synthetic data quality in healthcare applications, with built-in support for MIMIC-III readmission prediction tasks.

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/kontramind/sdvaluation.git
cd sdvaluation
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Using pip

```bash
git clone https://github.com/kontramind/sdvaluation.git
cd sdvaluation
pip install -e .
```

## Quick Start

### 1. Basic usage

```bash
sdvaluation shapley \
  -t data/synthetic_train.csv \
  -e data/real_test.csv \
  -c IS_READMISSION_30D
```

### 2. Fast valuation with early truncation

```bash
sdvaluation shapley \
  -t data/synthetic_train.csv \
  -e data/real_test.csv \
  -n 50 \
  -m 100 \
  -o experiments/quick_valuation
```

### 3. Using pre-trained hyperparameters

```bash
sdvaluation shapley \
  -t data/synthetic_train.csv \
  -e data/real_test.csv \
  --lgbm-params-json experiments/training/best_params.json \
  --encoding-config configs/encoding.yaml
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--train-data` | `-t` | Path to synthetic training CSV (required) | - |
| `--test-data` | `-e` | Path to real test CSV (required) | - |
| `--target-column` | `-c` | Name of the target column | `IS_READMISSION_30D` |
| `--encoding-config` | - | Path to RDT encoding YAML file | `None` |
| `--num-samples` | `-n` | Number of Monte Carlo samples (min: 10) | `100` |
| `--max-coalition-size` | `-m` | Max coalition size for truncation (min: 10) | `None` |
| `--random-state` | `-s` | Random seed for reproducibility | `42` |
| `--output-dir` | `-o` | Output directory for results | `experiments/data_valuation` |
| `--include-features` | - | Include feature columns in output CSV | `True` |
| `--no-include-features` | - | Exclude feature columns from output CSV | - |
| `--lgbm-params-json` | - | Path to JSON with LGBM hyperparameters | `None` |

## Output Format

The valuation results are saved as a CSV file in the specified output directory. Each row represents one training sample with the following columns (in order):

| Column | Description |
|--------|-------------|
| `{feature_columns}` | Original feature columns with their names (if `include_features=True`) |
| `target` | Target column value |
| `shapley_value` | Estimated Data Shapley value |
| `shapley_std` | Standard deviation of the estimate |
| `shapley_se` | Standard error of the estimate |
| `shapley_ci_lower` | Lower bound of 95% confidence interval |
| `shapley_ci_upper` | Upper bound of 95% confidence interval |
| `sample_index` | Index of the sample in the training data |

**Note:** Rows are sorted by `shapley_value` in ascending order (most harmful samples first).

## Understanding Shapley Values

### Interpreting Results

- **Positive values** (> 0): The data point contributes positively to model performance and is valuable
- **Negative values** (< 0): The data point hurts model performance and is harmful
- **Zero values** (≈ 0): The data point has no significant impact on model performance

### Confidence Intervals

Confidence intervals provide statistical bounds for the value estimates:

- **CI upper < 0**: The sample is definitively harmful with high confidence
- **CI lower > 0**: The sample is definitively valuable with high confidence
- **CI spans 0**: The effect is uncertain and may not be statistically significant

Use confidence intervals to filter low-quality synthetic data:
```python
# Remove definitively harmful samples
filtered = df[df['shapley_ci_upper'] >= 0]

# Keep only definitively valuable samples
high_quality = df[df['shapley_ci_lower'] > 0]
```

## How It Works

The Data Shapley algorithm computes the contribution of each training sample using cooperative game theory:

1. **Coalition Formation**: For each training sample, generate random subsets (coalitions) of other training samples
2. **Utility Evaluation**: Train a model on each coalition with and without the target sample
3. **Marginal Contribution**: Compute performance difference (marginal utility) on test data
4. **Shapley Estimation**: Average marginal contributions across all coalitions
5. **Truncation (optional)**: Stop early when coalition size exceeds `max_coalition_size`

**Computational Cost:**
- Full TMCS: `O(num_samples × n)` where `n` is training set size
- Truncated TMCS: `O(num_samples × max_coalition_size)`

For a dataset with 1000 samples and 100 Monte Carlo samples:
- Full: ~100,000 model training iterations
- Truncated (max=100): ~10,000 model training iterations (10× speedup)

## RDT Encoding Configuration

The package supports automatic encoding of mixed data types using RDT (Reversible Data Transforms). Specify encoding configuration in YAML format:

```yaml
sdtypes:
  SUBJECT_ID: id
  HADM_ID: id
  AGE: numerical
  GENDER: categorical
  ETHNICITY: categorical
  ADMISSION_TYPE: categorical
  DIAGNOSIS: categorical
  NUM_PROCEDURES: numerical
  NUM_MEDICATIONS: numerical
  LOS_DAYS: numerical
  PREV_ADMISSIONS: numerical
  IS_READMISSION_30D: categorical

transformers:
  AGE:
    type: NumericalTransformer
    config:
      dtype: float
  GENDER:
    type: FrequencyEncoder
  ETHNICITY:
    type: FrequencyEncoder
  ADMISSION_TYPE:
    type: FrequencyEncoder
  DIAGNOSIS:
    type: FrequencyEncoder
  IS_READMISSION_30D:
    type: LabelEncoder
```

**Supported Transformers:**
- `NumericalTransformer`: Standard scaling for numerical features
- `FrequencyEncoder`: Encode categorical by frequency
- `LabelEncoder`: Encode target labels
- `OneHotEncoder`: One-hot encoding for categoricals
- `BinaryEncoder`: Binary encoding for high-cardinality categoricals

## Python API

You can also use sdvaluation programmatically:

```python
from pathlib import Path
from sdvaluation.core import run_data_valuation

# Run data valuation
run_data_valuation(
    train_file=Path("data/synthetic_train.csv"),
    test_file=Path("data/real_test.csv"),
    target_column="IS_READMISSION_30D",
    encoding_config=Path("configs/encoding.yaml"),
    num_samples=100,
    max_coalition_size=None,
    random_state=42,
    output_dir=Path("experiments/data_valuation"),
    include_features=True,
    lgbm_params=None
)
```

For custom workflows:

```python
import pandas as pd
from sdvaluation.valuator import LGBMDataValuator
from sdvaluation.encoding import apply_encoding

# Load data
train_df = pd.read_csv("data/synthetic_train.csv")
test_df = pd.read_csv("data/real_test.csv")

# Apply encoding
train_encoded, test_encoded = apply_encoding(
    train_df, test_df,
    target_column="IS_READMISSION_30D",
    encoding_config_path="configs/encoding.yaml"
)

# Initialize valuator
valuator = LGBMDataValuator(
    X_train=train_encoded.drop(columns=["IS_READMISSION_30D"]),
    y_train=train_encoded["IS_READMISSION_30D"],
    X_test=test_encoded.drop(columns=["IS_READMISSION_30D"]),
    y_test=test_encoded["IS_READMISSION_30D"],
    num_samples=100,
    random_state=42
)

# Compute Shapley values
results = valuator.compute_data_shapley_values()
print(f"Mean Shapley value: {results['shapley_value'].mean():.4f}")
```

## Requirements

- **Python**: >=3.10,<3.13
- **Core dependencies**:
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `scikit-learn>=1.3.0`
  - `lightgbm>=4.0.0`
  - `rdt>=1.12.1`
  - `pyyaml>=6.0`
- **CLI dependencies**:
  - `typer>=0.9.0`
  - `rich>=13.0.0`
- **Optional**:
  - `jupyter>=1.0.0` (for notebook examples)

See `pyproject.toml` for complete dependency specifications.

## Use Case: MIMIC-III Readmission

This package was developed for evaluating synthetic data quality in the context of **30-day hospital readmission prediction** using MIMIC-III data. The task involves:

- **Target**: Binary classification of 30-day readmission risk
- **Features**: Patient demographics, admission details, procedures, medications, length of stay
- **Challenge**: Synthetic data generated by models (e.g., GANs, VAEs, CTGAN) may contain low-quality or harmful samples
- **Solution**: Use Data Shapley to identify and filter out harmful synthetic samples, retaining only high-quality training data

By removing samples with negative Shapley values, model performance on real test data can be significantly improved.

## Leaf Co-Occurrence Hallucination Detection

### Overview

In addition to Data Shapley valuation, **sdvaluation** provides a complementary method for detecting **hallucinated synthetic data** using **leaf co-occurrence analysis**. This method identifies synthetic training points that create decision boundaries misaligned with real test data patterns.

**Key Insight:** Data Shapley measures individual point quality in random subsets (marginal contribution), but hallucinations are often **distributional issues** that manifest when training on the full dataset. Leaf co-occurrence directly measures whether synthetic points help build decision boundaries that generalize to real data.

### Why Use Leaf Co-Occurrence?

**Shapley Analysis Alone Can Miss Distributional Hallucinations:**

```
Scenario: Synthetic data with good individual point quality but wrong collective patterns

Shapley says:     "3.4% of points are harmful"  ← Point-level analysis
Reality:          "93% of points create wrong decision boundaries"
Confusion matrix: "Recall drops from 40% to 10%"

Root cause: Individual points look plausible but collectively have wrong correlations
```

**Use Both Methods Together:**

| Method | Measures | Detects | Runtime |
|--------|----------|---------|---------|
| **Confusion Matrix** | Aggregate model performance | Performance degradation | ~2 min |
| **Leaf Co-Occurrence** | Decision boundary quality | Individual hallucinated points | ~5 min |
| **Data Shapley** | Marginal contribution | Point-level harmful samples | ~90 min |

### How It Works

The leaf co-occurrence algorithm adapts "In-Run Shapley" for tree models:

1. **Train LightGBM once** on synthetic training data
2. **Extract leaf assignments**: Pass synthetic training + real test data through the model
3. **Score each leaf**: For every leaf in every tree:
   - Calculate how well it classifies **real test data** (leaf utility)
   - Assign utility to **synthetic points** that fell into that leaf
4. **Aggregate across trees**: Each tree provides an independent utility estimate
5. **Compute confidence intervals**: Identify reliably hallucinated points (CI upper < 0)

**The Core Idea:**
```
Synthetic point creates leaf → Real test data falls (or doesn't fall) into that leaf
→ Does the leaf correctly classify real data?
→ Low utility = Hallucinated point that created wrong decision boundaries
```

### Evaluation-Based with Implicit Counterfactual Reasoning

**Primary Mechanism: Alignment Evaluation**

The leaf co-occurrence method is fundamentally **evaluation-based** rather than explicitly counterfactual:

```python
# What actually happens (one training run)
model = train(synthetic_data)  # Train once on all synthetic points
structure = model.get_decision_boundaries()  # Extract learned structure

# Evaluate the learned structure
for each boundary:
    utility = does_it_work_for_real_data(boundary)
    assign_utility_to(synthetic_points_that_created_boundary)
```

**Implicit Counterfactual Evidence**

While not explicitly counterfactual, the method provides **implicit counterfactual reasoning**:

```
Factual observation:
  "Synthetic point X created leaf L with these boundaries"
  → Real data in leaf L gets misclassified
  → Utility score: -0.015 (harmful)

Implicit counterfactual:
  "Without point X (or with better synthetic data)..."
  → Different boundaries would have formed
  → Real data might be correctly classified
  → Evidence: Current boundaries don't work
```

**Contrast with Explicit Counterfactuals (Shapley)**

| Aspect | Shapley (Explicit Counterfactual) | Leaf Co-Occurrence (Evaluation-Based) |
|--------|----------------------------------|--------------------------------------|
| **Comparison** | Train WITH vs WITHOUT each point | Evaluate structure created WITH all points |
| **Training runs** | O(n × samples) ≈ millions | O(1) = once |
| **Evidence type** | Direct performance difference | Structural quality assessment |
| **Question** | "What's the marginal contribution?" | "Does the learned structure generalize?" |
| **Runtime** | ~90 minutes | ~5 minutes |

**Why This Distinction Matters**

1. **Computational efficiency**: Single training run vs. millions
2. **Different perspectives**:
   - Shapley measures point-level marginal contribution (individual quality)
   - Leaf alignment measures structural generalization (collective quality)
3. **Complementary insights**: Shapley can miss distributional issues that leaf alignment catches

**Describing the Method**

Most accurate description:
> "An **alignment-based evaluation** method that measures whether synthetic training points create decision boundaries that generalize to real data patterns, with **implicit counterfactual reasoning** about what better boundaries would look like."

Alternative framing emphasizing the counterfactual aspect:
> "Through structural evaluation, provides **implicit counterfactual evidence**: synthetic points creating boundaries that fail on real data suggest better data would create boundaries that succeed."

### Usage

#### Basic Leaf Co-Occurrence Analysis

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/synthetic_train.csv \
  --real-test data/real_test.csv \
  --encoding-config config/encoding.yaml \
  --lgbm-params config/lgbm_params.json
```

#### With Class-Specific Breakdown

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/synthetic_train.csv \
  --real-test data/real_test.csv \
  --encoding-config config/encoding.yaml \
  --lgbm-params config/lgbm_params.json \
  --by-class  # Show positive vs negative class separately
```

#### With Tighter Confidence Intervals

```bash
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train data/synthetic_train.csv \
  --real-test data/real_test.csv \
  --encoding-config config/encoding.yaml \
  --lgbm-params config/lgbm_params.json \
  --n-estimators 500  # More trees = tighter CIs (default: 100)
  --by-class
```

**Effect of More Trees:**

| Trees | CI Width | Uncertain % | Runtime | Use Case |
|-------|----------|-------------|---------|----------|
| 100 | Wider | ~30-40% | ~2 min | Quick exploration |
| 500 | Medium | ~5-10% | ~5 min | **Recommended** |
| 1000 | Tight | ~3-5% | ~10 min | Final analysis |

### Output

The script generates a CSV file (`hallucination_scores.csv`) with:

| Column | Description |
|--------|-------------|
| `synthetic_index` | Index in synthetic training data |
| `utility_score` | Mean utility across all trees |
| `utility_se` | Standard error |
| `utility_ci_lower` | Lower 95% confidence bound |
| `utility_ci_upper` | Upper 95% confidence bound |
| `reliably_hallucinated` | True if CI upper < 0 |

### Interpreting Results

#### Utility Scores

- **Positive utility (> 0)**: Synthetic point creates leaves that correctly classify real data
- **Negative utility (< 0)**: Synthetic point creates leaves that misclassify real data
- **CI upper < 0**: Reliably hallucinated (all trees agree it's harmful)
- **CI lower > 0**: Reliably beneficial (all trees agree it's helpful)
- **CI spans 0**: Uncertain

#### Example Output

```
Statistical Confidence (95% CI-based):
  Reliably hallucinated (CI upper < 0): 9,339 (93.39%)  ← 93% of data is hallucinated!
  Reliably beneficial (CI lower > 0):   54 (0.54%)
  Uncertain (CI spans 0):                607 (6.07%)

Class-Specific Statistics:

Negative (No Readmission) - 8,997 points
  Reliably hallucinated: 8,385 (93.20%)  ← Both classes are bad

Positive (Readmission) - 1,003 points
  Reliably hallucinated: 954 (95.11%)  ← But positive class is worse
  Positive utility:      0 (0.00%)      ← EVERY point is harmful!
```

#### What This Means

**Good synthetic data (like Real training data):**
- ~90% reliably beneficial
- ~0.25% reliably hallucinated
- Creates decision boundaries that generalize to test data

**Bad synthetic data (Gen1/Gen2 in our tests):**
- ~93-95% reliably hallucinated
- ~0% reliably beneficial
- Creates decision boundaries in wrong places
- Cannot learn target patterns (e.g., readmissions)

### Confusion Matrix Comparison (Fast Screening)

For even faster validation (~2 minutes), use the confusion matrix tool:

```bash
python generate_confusion_matrices.py \
  --real-train data/real_train.csv \
  --next-gen-train data/synthetic_train.csv \
  --test-file data/test.csv \
  --encoding-config config/encoding.yaml \
  --lgbm-params config/lgbm_params.json
```

**Output:**
```
Real Data:  Precision: 18.34%, Recall: 39.96%, F1: 25.14%
Gen2 Data:  Precision:  7.86%, Recall: 10.49%, F1:  8.99%

Metric Changes (Gen2 vs Real):
  Precision: -10.48% (degraded)
  Recall:    -29.47% (degraded)  ← Major red flag!
  F1 Score:  -16.15% (degraded)
```

### Recommended Validation Workflow

For synthetic data quality assessment, use all three methods:

```bash
# 1. Fast screening (~2 min)
python generate_confusion_matrices.py \
  --real-train real_train.csv \
  --next-gen-train synthetic_train.csv \
  --test-file test.csv \
  --encoding-config encoding.yaml \
  --lgbm-params lgbm_params.json

# If performance degrades significantly:

# 2. Leaf co-occurrence analysis (~5 min)
python detect_hallucinations_leaf_alignment.py \
  --synthetic-train synthetic_train.csv \
  --real-test test.csv \
  --encoding-config encoding.yaml \
  --lgbm-params lgbm_params.json \
  --by-class \
  --n-estimators 500

# 3. Full Shapley analysis (~90 min) - if needed for point-level diagnosis
python -m sdvaluation.core \
  --train-file synthetic_train.csv \
  --test-file test.csv \
  --encoding-config encoding.yaml \
  --lgbm-params lgbm_params.json \
  --num-samples 200 \
  --max-coalition-size 5000
```

**What Each Method Tells You:**

1. **Confusion Matrix** → "Is there a problem?" (aggregate performance)
2. **Leaf Co-Occurrence** → "Which points are hallucinated?" (decision boundaries)
3. **Shapley Analysis** → "What's the marginal contribution?" (random subsets)

### Real-World Example: MIMIC-III Gen2 Synthetic Data

**Scenario:** Recursive synthetic training (Real → Gen1 → Gen2) using SynthCity Marginal Distributions

**Results:**

| Dataset | Confusion Matrix | Leaf Co-Occurrence | Shapley Analysis |
|---------|-----------------|-------------------|------------------|
| **Real** | Recall: 40% | 0.25% hallucinated | 3.28% reliably harmful |
| **Gen2** | Recall: 10% ❌ | 93.39% hallucinated ❌ | 3.39% reliably harmful ✓ |

**Diagnosis:**
- Confusion matrix detected 30% recall drop
- Leaf co-occurrence revealed 93% of Gen2 creates wrong decision boundaries
- Shapley alone missed the problem (only 3.39% harmful)
- **Root cause:** Gen2 points look individually plausible but collectively have wrong patterns

**Positive Class Analysis:**
```
Real positive class:  74% helpful, 2% harmful
Gen2 positive class:  0% helpful, 95% harmful  ← Completely destroyed!
```

**Conclusion:** Gen2 cannot learn readmissions because 95% of positive class examples are hallucinated patterns.

### When to Use Each Method

**Use Confusion Matrix when:**
- Quick screening for performance degradation
- Comparing multiple synthetic datasets
- Checking if aggregate metrics changed

**Use Leaf Co-Occurrence when:**
- Confusion matrix shows degradation
- Need to identify specific hallucinated points
- Want to understand class-specific issues
- Faster than Shapley, more detailed than confusion matrix

**Use Shapley Analysis when:**
- Need marginal contribution across random subsets
- Building point removal/weighting strategies
- Academic/publication rigor required
- Have time for comprehensive analysis

### Key Findings from Our Research

Testing Real, Gen1, and Gen2 MIMIC-III readmission data:

1. **Gen2 > Gen1** (marginally): Gen2 is 1-8% less hallucinated than Gen1
2. **Both unusable**: Gen1: 94.44% hallucinated, Gen2: 93.39% hallucinated
3. **Real is excellent**: 0.25% hallucinated, 89.69% beneficial
4. **Ratio: 373:1**: Gen2 is 373× more hallucinated than Real
5. **Failure at Gen1**: Catastrophic drop happened Real→Gen1, not Gen1→Gen2
6. **Positive class destroyed**: 0% of Gen1/Gen2 positive points create useful boundaries

**Implication:** Recursive training doesn't degrade quality further; the initial generation fundamentally failed to capture predictive patterns.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Author

**Aleksandar Babic**
Email: aleksandar.babic@gmail.com

## References

1. **Data Shapley: Equitable Valuation of Data for Machine Learning**
   Amirata Ghorbani and James Zou
   *International Conference on Machine Learning (ICML)*, 2019
   [Paper](https://arxiv.org/abs/1904.02868)

2. **Towards Efficient Data Valuation Based on the Shapley Value**
   Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas Spanos
   *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2019
   [Paper](https://arxiv.org/abs/1902.10275)

---

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/kontramind/sdvaluation).
