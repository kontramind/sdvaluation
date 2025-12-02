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

The valuation results are saved as a CSV file in the specified output directory. Each row represents one training sample with the following columns:

| Column | Description |
|--------|-------------|
| `{features}` | Original feature columns (if `include_features=True`) |
| `{target}` | Target column value |
| `shapley_value` | Estimated Data Shapley value |
| `shapley_std` | Standard deviation of the estimate |
| `shapley_se` | Standard error of the estimate |
| `shapley_ci_lower` | Lower bound of 95% confidence interval |
| `shapley_ci_upper` | Upper bound of 95% confidence interval |
| `sample_index` | Index of the sample in the training data |

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
