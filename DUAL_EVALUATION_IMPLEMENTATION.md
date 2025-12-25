# Dual Scenario Evaluation - Implementation Summary

## Overview

Implementation complete! The dual scenario evaluation system has been added to `sdvaluation` with zero dependencies on `sdpype`.

## What Was Implemented

### 1. New Modules

#### `sdvaluation/tuning.py`
- Self-contained LightGBM hyperparameter tuning using Optuna
- Bayesian optimization with Tree-structured Parzen Estimator (TPE)
- Stratified k-fold cross-validation
- No dependency on sdpype

**Key class**: `LGBMTuner`
- Optimizes for AUROC with comprehensive hyperparameter search space
- Handles class imbalance (scale_pos_weight, is_unbalance)
- Returns clean parameter dict ready for LGBMClassifier

#### `sdvaluation/leaf_alignment.py`
- Leaf co-occurrence analysis for hallucination detection
- Extracted and refactored from existing `detect_hallucinations_leaf_alignment.py`
- Reusable function-based API

**Key function**: `run_leaf_alignment()`
- Trains LGBM on synthetic data
- Computes utility scores based on leaf quality for real test data
- Returns summary statistics and optionally saves CSV results

#### `sdvaluation/dual_evaluation.py`
- Core orchestration for dual scenario comparison
- Manages entire workflow from data loading to result saving

**Key function**: `run_dual_evaluation()`
- Loads and encodes 4 datasets (tuning 40k, real train 10k, synth train 10k, real test 10k)
- Tunes hyperparameters on both 40k and 10k
- Evaluates Real vs Synth in both scenarios
- Runs leaf alignment with both parameter sets
- Generates comprehensive reports

#### `sdvaluation/cli.py`
- Extended with new `dual-eval` command
- Rich terminal output with tables, progress indicators
- Comprehensive help text and examples

### 2. Updated Files

#### `pyproject.toml`
- Added `optuna>=3.0.0` dependency

## Usage

### Installation

```bash
cd sdvaluation
pip install -e .
```

### Command Line

```bash
sdvaluation dual-eval \
  --tuning-data population_40k.csv \
  --real-train real_train_10k.csv \
  --synth-train synth_train_10k.csv \
  --real-test real_test_10k.csv \
  --encoding-config encoding.yaml \
  --n-trials 100 \
  --n-folds 5 \
  --output-dir experiments/dual_eval
```

### Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--tuning-data` | Yes | - | 40k population data for hyperparameter tuning |
| `--real-train` | Yes | - | 10k real training data |
| `--synth-train` | Yes | - | 10k synthetic training data |
| `--real-test` | Yes | - | 10k real test data |
| `--encoding-config` | Yes | - | RDT encoding config YAML |
| `--target-column`, `-c` | No | `IS_READMISSION_30D` | Target column name |
| `--n-trials`, `-n` | No | 100 | Number of Optuna trials |
| `--n-folds`, `-k` | No | 5 | Number of CV folds |
| `--output-dir`, `-o` | No | `experiments/dual_eval` | Output directory |
| `--no-leaf-alignment` | No | False | Skip leaf alignment analysis |
| `--seed`, `-s` | No | 42 | Random seed |

### Python API

```python
from pathlib import Path
from sdvaluation.dual_evaluation import run_dual_evaluation

results = run_dual_evaluation(
    tuning_data=Path("data/population_40k.csv"),
    real_train=Path("data/real_train_10k.csv"),
    synth_train=Path("data/synth_train_10k.csv"),
    real_test=Path("data/real_test_10k.csv"),
    target_column="IS_READMISSION_30D",
    encoding_config=Path("config/encoding.yaml"),
    output_dir=Path("experiments/dual_eval"),
    n_trials=100,
    n_folds=5,
    run_leaf_alignment=True,
    random_state=42,
)

print(f"Transfer gap: {results['transfer_gap']:.4f}")
print(f"Synth gap (10k params): {results['scenario_1_optimal']['gap']:.4f}")
print(f"Synth gap (40k params): {results['scenario_2_deployment']['gap']:.4f}")
```

## Output Structure

```
experiments/dual_eval/
├── params_40k_YYYYMMDD_HHMMSS.json
├── params_10k_YYYYMMDD_HHMMSS.json
├── scenario_1_optimal_YYYYMMDD_HHMMSS.json
├── scenario_2_deployment_YYYYMMDD_HHMMSS.json
├── leaf_alignment_10k_params_YYYYMMDD_HHMMSS.csv
├── leaf_alignment_40k_params_YYYYMMDD_HHMMSS.csv
└── summary_YYYYMMDD_HHMMSS.json
```

### File Contents

#### `params_40k_*.json` / `params_10k_*.json`
```json
{
  "params": {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.1,
    ...
  },
  "cv_score": 0.8234,
  "n_trials": 100,
  "n_folds": 5,
  "source": "tuning_data_40k"
}
```

#### `scenario_1_optimal_*.json` / `scenario_2_deployment_*.json`
```json
{
  "scenario": "optimal_10k_tuned",
  "params_source": "real_train_10k",
  "real_performance": {
    "auroc": 0.8012,
    "precision": 0.183,
    "recall": 0.400,
    "f1": 0.251,
    "logloss": -0.432,
    "confusion_matrix": {
      "tn": 7234,
      "fp": 1432,
      "fn": 234,
      "tp": 1100
    }
  },
  "synth_performance": {
    "auroc": 0.7234,
    "precision": 0.121,
    "recall": 0.223,
    "f1": 0.158,
    ...
  }
}
```

#### `leaf_alignment_*_params_*.csv`
```csv
synthetic_index,utility_score,utility_se,utility_ci_lower,utility_ci_upper,reliably_hallucinated
0,-0.0234,0.0012,-0.0258,-0.0210,True
1,0.0123,0.0015,0.0093,0.0153,False
2,-0.0456,0.0018,-0.0492,-0.0420,True
...
```

#### `summary_*.json`
```json
{
  "timestamp": "20251211_143022",
  "transfer_gap": 0.0089,
  "scenario_1_optimal": {
    "params_source": "real_train_10k",
    "real_auroc": 0.8012,
    "synth_auroc": 0.7234,
    "gap": -0.0778
  },
  "scenario_2_deployment": {
    "params_source": "tuning_data_40k",
    "real_auroc": 0.7923,
    "synth_auroc": 0.7102,
    "gap": -0.0821
  },
  "leaf_alignment": {
    "10k_params": {
      "n_total": 10000,
      "n_hallucinated": 9234,
      "pct_hallucinated": 92.34,
      ...
    },
    "40k_params": {
      "n_total": 10000,
      "n_hallucinated": 9401,
      "pct_hallucinated": 94.01,
      ...
    }
  }
}
```

## Workflow

The evaluation runs in 6 phases:

### Phase 1: Data Loading
- Loads all 4 datasets
- Applies RDT encoding consistently across all datasets
- Fits encoder on 40k tuning data only

### Phase 2: Hyperparameter Tuning
- **Tuning on 40k**: Population-level hyperparameters
  - Represents historical data / different distribution
  - Deployment scenario

- **Tuning on 10k**: Real training data hyperparameters
  - Represents optimal configuration for this specific dataset
  - Best-case scenario

### Phase 3: Scenario 1 - Optimal (10k-tuned params)
- Train Real (10k) → Test on Real (10k)
- Train Synth (10k) → Test on Real (10k)
- Compare performance
- **Interpretation**: Maximum quality synthetic data can achieve

### Phase 4: Scenario 2 - Deployment (40k-tuned params)
- Train Real (10k) → Test on Real (10k)
- Train Synth (10k) → Test on Real (10k)
- Compare performance
- **Interpretation**: Realistic deployment with parameter transfer

### Phase 5: Transfer Gap Analysis
- Compares Real performance with 10k vs 40k params
- Quantifies how well hyperparameters transfer
- Determines if scenario comparisons are valid

### Phase 6: Leaf Alignment (Optional)
- Runs harmful detection with both parameter sets
- Identifies specific hallucinated points
- Shows consistency across scenarios

## Interpretation Guide

### Transfer Gap

| Gap | Status | Interpretation |
|-----|--------|----------------|
| < 0.02 | Excellent ✓ | Hyperparameters transfer perfectly |
| 0.02-0.05 | Acceptable ⚠ | Minor mismatch, comparison still valid |
| > 0.05 | Poor ❌ | Significant mismatch, results confounded |

### Synthetic Quality Gap

| Scenario | Gap | Interpretation |
|----------|-----|----------------|
| Optimal (10k) | -0.08 | Synth 8% worse than Real (best case) |
| Deployment (40k) | -0.08 | Synth 8% worse than Real (realistic) |

**Key insight**: If gaps are similar in both scenarios → problem is fundamental synthetic quality, not hyperparameter sensitivity

### Leaf Alignment

| Hallucinated % | Quality |
|----------------|---------|
| < 5% | Excellent (comparable to real data) |
| 5-20% | Acceptable |
| 20-50% | Poor quality |
| > 50% | Fundamentally broken |

## Design Decisions

1. **No stratification**: Simulates realistic scenario where future data distribution is unknown
2. **Both parameter sets**: Tests robustness across configurations
3. **Threshold 0.5**: Simple default, can be made configurable later
4. **500 trees for leaf alignment**: Balance between CI tightness and runtime
5. **Rich terminal output**: Immediate visual feedback during long runs

## Next Steps for Testing

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Prepare your data**:
   - Split 60k population into 40k tuning + 10k real train + 10k real test
   - Generate 10k synthetic from real train
   - Ensure all have same columns and target

3. **Run evaluation**:
   ```bash
   sdvaluation dual-eval \
     --tuning-data data/tuning_40k.csv \
     --real-train data/real_train_10k.csv \
     --synth-train data/synth_train_10k.csv \
     --real-test data/real_test_10k.csv \
     --encoding-config config/encoding.yaml \
     --n-trials 100
   ```

4. **Review results** in `experiments/dual_eval/`

## Troubleshooting

### Import errors
Ensure all dependencies installed: `pip install -e .`

### Memory issues
Reduce `--n-trials` or `--n-folds` for faster tuning

### Leaf alignment too slow
Use `--no-leaf-alignment` to skip harmful detection phase

### Different column names
Use `--target-column` to specify your target column name

## Files Modified

- ✓ `sdvaluation/tuning.py` (NEW)
- ✓ `sdvaluation/leaf_alignment.py` (NEW)
- ✓ `sdvaluation/dual_evaluation.py` (NEW)
- ✓ `sdvaluation/cli.py` (EXTENDED)
- ✓ `pyproject.toml` (UPDATED - added optuna)

---

**Status**: Ready for testing! 🚀
