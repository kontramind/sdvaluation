"""
Hyperparameter tuning for LightGBM using Optuna (Bayesian Optimization).

Self-contained module for optimizing LGBM hyperparameters with cross-validation.
"""

from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class LGBMTuner:
    """
    Lightweight LGBM hyperparameter tuner using Bayesian Optimization.

    Optimizes hyperparameters using Optuna with Tree-structured Parzen Estimator (TPE)
    and stratified k-fold cross-validation with AUROC as the primary metric.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int = 5,
        n_trials: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize the tuner.

        Args:
            X_train: Training features
            y_train: Training labels (binary)
            n_folds: Number of cross-validation folds
            n_trials: Number of Bayesian optimization trials
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization.

        Args:
            trial: A trial object from Optuna

        Returns:
            Mean AUROC across folds
        """
        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 4, 60),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 2**(-10), 2**0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 60),
            'n_estimators': 1000,  # Large number, will use early stopping
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        # Regularization parameters
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 10.0)  # L1
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 10.0)  # L2

        # Feature sampling
        params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)

        # Leaf constraints
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 50)

        # Class imbalance handling
        imbalance_method = trial.suggest_categorical(
            'imbalance_method',
            ['none', 'scale_pos_weight', 'is_unbalance']
        )

        if imbalance_method == 'scale_pos_weight':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            params['is_unbalance'] = True

        # Early stopping
        early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 7, 30)

        # Perform k-fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        cv_scores = []

        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=0)  # Suppress output
                ]
            )

            # Predict on validation set
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            # Calculate AUROC
            auroc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auroc)

        return np.mean(cv_scores)

    def tune(self, timeout: int = None) -> Dict[str, Any]:
        """
        Run Bayesian Optimization to find best hyperparameters.

        Args:
            timeout: Maximum time in seconds for optimization (None = no limit)

        Returns:
            Best hyperparameters found
        """
        # Create Optuna study with TPE sampler
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=timeout,
            show_progress_bar=False
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params

    def get_lgbm_params(self) -> Dict[str, Any]:
        """
        Get the best parameters formatted for LGBMClassifier.

        Returns:
            Dictionary of hyperparameters ready for LGBMClassifier
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before getting parameters")

        # Format parameters for LGBMClassifier
        lgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': self.best_params['boosting_type'],
            'num_leaves': self.best_params['num_leaves'],
            'max_depth': self.best_params['max_depth'],
            'learning_rate': self.best_params['learning_rate'],
            'min_child_samples': self.best_params['min_child_samples'],
            'n_estimators': 1000,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1,
            'reg_alpha': self.best_params.get('reg_alpha', 0.0),
            'reg_lambda': self.best_params.get('reg_lambda', 0.0),
            'feature_fraction': self.best_params.get('feature_fraction', 1.0),
            'min_data_in_leaf': self.best_params.get('min_data_in_leaf', 20),
        }

        # Add class imbalance handling if selected
        imbalance_method = self.best_params.get('imbalance_method', 'none')
        if imbalance_method == 'scale_pos_weight':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            lgbm_params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            lgbm_params['is_unbalance'] = True

        # Store metadata separately
        lgbm_params['imbalance_method'] = imbalance_method
        lgbm_params['early_stopping_rounds'] = self.best_params.get('early_stopping_rounds', 10)

        return lgbm_params


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    timeout: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function to tune LGBM hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        timeout: Timeout in seconds (None = no limit)
        random_state: Random seed

    Returns:
        Dictionary containing best parameters and CV score
    """
    tuner = LGBMTuner(
        X_train=X_train,
        y_train=y_train,
        n_folds=n_folds,
        n_trials=n_trials,
        random_state=random_state,
    )

    best_params = tuner.tune(timeout=timeout)
    lgbm_params = tuner.get_lgbm_params()

    return {
        'best_params': lgbm_params,
        'cv_score': tuner.best_score,
        'n_trials': n_trials,
        'n_folds': n_folds,
    }
