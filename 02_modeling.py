"""
PHASE 2: Modeling and Interpretation
=====================================
Latino Trump Support ML Analysis

Steps 2.1-2.3: Data loading, train/test split, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Step 2.1: Load Data
# =============================================================================

print("=" * 60)
print("PHASE 2: Modeling and Interpretation")
print("Step 2.1: Load Data")
print("=" * 60)

# Load parquet files
X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet')['trump_vote']
weights = pd.read_parquet('cmps_2016_weights.parquet')['survey_wt']

print(f"\nData loaded successfully:")
print(f"  X shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
print(f"  y shape: {len(y):,} labels")
print(f"  weights shape: {len(weights):,} weights")

print(f"\nTarget distribution:")
print(f"  Trump voters (1): {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)")

# =============================================================================
# Step 2.2: Train/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.2: Train/Test Split")
print("=" * 60)

# 80/20 split, stratified on y, random_state=42
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"\nTrain/Test Split (80/20, stratified):")
print(f"  Training set: {len(X_train):,} observations")
print(f"  Test set: {len(X_test):,} observations")

print(f"\nTraining set target distribution:")
print(f"  Trump voters (1): {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y_train).sum():,} ({(1-y_train.mean())*100:.1f}%)")

print(f"\nTest set target distribution:")
print(f"  Trump voters (1): {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y_test).sum():,} ({(1-y_test.mean())*100:.1f}%)")

# =============================================================================
# Step 2.3: GridSearchCV for Hyperparameter Tuning
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.3: GridSearchCV (5-fold CV)")
print("=" * 60)

# Define parameter grid
param_grid = {
    'max_features': ['sqrt', 0.33],
    'min_samples_leaf': [1, 5]
}

# Fixed parameters
fixed_params = {
    'n_estimators': 500,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nFixed parameters:")
for k, v in fixed_params.items():
    print(f"  {k}: {v}")

print(f"\nParameter grid to search:")
for k, v in param_grid.items():
    print(f"  {k}: {v}")

print(f"\nTotal combinations: {len(param_grid['max_features']) * len(param_grid['min_samples_leaf'])}")

# Create base estimator
rf = RandomForestClassifier(**fixed_params)

# Create stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run GridSearchCV
print(f"\nRunning 5-fold cross-validation...")
print(f"Optimizing: ROC-AUC")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Fit with sample weights
grid_search.fit(X_train, y_train, sample_weight=weights_train)

# =============================================================================
# Results
# =============================================================================

print("\n" + "=" * 60)
print("GridSearchCV Results")
print("=" * 60)

# Best parameters
print(f"\nBest Parameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nBest CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# All results
print(f"\n" + "-" * 60)
print("All Parameter Combinations:")
print("-" * 60)

results_df = pd.DataFrame(grid_search.cv_results_)
results_summary = results_df[[
    'param_max_features',
    'param_min_samples_leaf',
    'mean_test_score',
    'std_test_score',
    'mean_train_score',
    'rank_test_score'
]].sort_values('rank_test_score')

results_summary.columns = ['max_features', 'min_samples_leaf', 'CV_AUC_mean', 'CV_AUC_std', 'Train_AUC', 'Rank']

print(results_summary.to_string(index=False))

# Check for overfitting
print(f"\n" + "-" * 60)
print("Overfitting Check (Train - CV difference):")
print("-" * 60)
for idx, row in results_summary.iterrows():
    overfit = row['Train_AUC'] - row['CV_AUC_mean']
    print(f"  max_features={row['max_features']}, min_samples_leaf={row['min_samples_leaf']}: "
          f"Train={row['Train_AUC']:.4f}, CV={row['CV_AUC_mean']:.4f}, Diff={overfit:.4f}")

print("\n" + "=" * 60)
print("Step 2.3 Complete: Ready to train final model")
print("=" * 60)
print(f"\nRecommended parameters for final model:")
print(f"  n_estimators: 500")
print(f"  max_features: {grid_search.best_params_['max_features']}")
print(f"  min_samples_leaf: {grid_search.best_params_['min_samples_leaf']}")
print(f"  class_weight: 'balanced'")
print(f"\nExpected CV ROC-AUC: {grid_search.best_score_:.4f}")
