#!/usr/bin/env python3
"""
Phase 2 Step 2.6: CORRECTED Model Interpretation
Latino Trump Support ML Analysis - CMPS 2016

This script runs TWO models:
1. Full Model: All available predictors
2. Non-Partisan Model: Excludes all tautological/partisan variables

CORRECTED EXCLUSION LIST based on actual column names in data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 2 Step 2.6: CORRECTED Model Interpretation")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

# Load the one-hot encoded feature matrix from parquet files
X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet')['trump_vote']
weights = pd.read_parquet('cmps_2016_weights.parquet')['survey_wt']

print(f"  X shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
print(f"  DV distribution: Trump=1: {y.sum():,} ({100*y.mean():.1f}%), Non-Trump=0: {(~y.astype(bool)).sum():,.0f} ({100*(1-y.mean()):.1f}%)")

# =============================================================================
# DEFINE COMPLETE EXCLUSION LIST FOR NON-PARTISAN MODEL
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING EXCLUSION LIST")
print("=" * 70)

# Prefixes to exclude (all candidate favorability, party ID, party evaluations)
exclude_prefixes = [
    # Candidate Favorability (C2-C11 series) - ALL candidates
    'C2_',    # Favorability variable
    'C3_',    # Favorability variable
    'C4_',    # Favorability variable (appears to be Trump based on predictive power)
    'C5_',    # Favorability variable
    'C8_',    # Favorability variable
    'C9_',    # Favorability variable
    'C10_',   # Favorability variable
    'C11_',   # Favorability variable

    # Party Identification
    'C25_',   # Party self-ID (Rep/Dem/Ind)
    'C26_',   # Party strength (strong partisan)
    'C27_',   # Party lean (for independents)
    'C31_',   # Ideology (liberal-conservative scale)

    # Party Evaluations - "Which party better on X"
    'L46_',   # Which party better on immigration
    'L266_',  # Which party better for Latinos
    'L267_',  # Which party better on values

    # Party Favorability Scales
    'L293_',  # Democratic Party favorability (0-10 scale)
    'L294_',  # Republican Party favorability (0-10 scale)

    # Derived Party ID
    'C242_HID_',  # Derived party identification

    # Party Support (LA204 asks about party support)
    'LA204_',  # Which party respondent's group supports
]

# Exact matches to exclude
exclude_exact = [
    'C242_HID',
]

def get_exclusion_columns(df_columns, exclude_prefixes, exclude_exact):
    """
    Identify all columns that should be excluded from non-partisan model.
    """
    exclude_cols = []

    for col in df_columns:
        # Check prefix matches
        for prefix in exclude_prefixes:
            if col.startswith(prefix):
                exclude_cols.append(col)
                break

        # Check exact matches
        if col in exclude_exact and col not in exclude_cols:
            exclude_cols.append(col)

    return sorted(list(set(exclude_cols)))

# Get columns to exclude
cols_to_exclude = get_exclusion_columns(X.columns, exclude_prefixes, exclude_exact)

print(f"\nTotal columns to exclude: {len(cols_to_exclude)}")
print("\nBreakdown by variable type:")

# Count by prefix
prefix_counts = {}
for col in cols_to_exclude:
    for prefix in exclude_prefixes:
        if col.startswith(prefix):
            prefix_key = prefix.rstrip('_')
            prefix_counts[prefix_key] = prefix_counts.get(prefix_key, 0) + 1
            break

for prefix, count in sorted(prefix_counts.items()):
    print(f"  {prefix}: {count} columns")

print("\n" + "-" * 70)
print("EXCLUDED COLUMNS (full list):")
print("-" * 70)
for col in cols_to_exclude:
    print(f"  {col}")

# Save excluded columns to file
pd.DataFrame({'excluded_column': cols_to_exclude}).to_csv('excluded_partisan_columns.csv', index=False)
print(f"\nSaved excluded columns to: excluded_partisan_columns.csv")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("TRAIN/TEST SPLIT")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.20, stratify=y, random_state=42
)

print(f"  Train: {len(y_train):,} ({100*y_train.mean():.1f}% Trump)")
print(f"  Test:  {len(y_test):,} ({100*y_test.mean():.1f}% Trump)")

# =============================================================================
# FULL MODEL
# =============================================================================
print("\n" + "=" * 70)
print("PART A: FULL MODEL (All Predictors)")
print("=" * 70)

rf_params = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print("\nTraining full model...")
rf_full = RandomForestClassifier(**rf_params)
rf_full.fit(X_train, y_train, sample_weight=weights_train)

# Evaluate
y_pred_full = rf_full.predict_proba(X_test)[:, 1]
auc_full = roc_auc_score(y_test, y_pred_full, sample_weight=weights_test)
print(f"  Full Model Test ROC-AUC: {auc_full:.4f}")

# Permutation importance for full model
print("\n" + "-" * 60)
print("Permutation Importance - Full Model (50 repeats)")
print("-" * 60)
print("  Computing permutation importance...")

perm_imp_full = permutation_importance(
    rf_full, X_test, y_test,
    n_repeats=50,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

# Create sorted importance DataFrame
imp_full_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_imp_full.importances_mean,
    'importance_std': perm_imp_full.importances_std
}).sort_values('importance_mean', ascending=False)

top30_full = imp_full_df.head(30).copy()
top30_full['rank'] = range(1, 31)
top30_full = top30_full[['rank', 'feature', 'importance_mean', 'importance_std']]

print("\n  Top 30 Features (Full Model):")
print("  " + "-" * 75)
print(f"  {'Rank':<6}{'Feature':<50}{'Importance':<12}{'Std':<10}")
print("  " + "-" * 75)
for _, row in top30_full.iterrows():
    print(f"  {int(row['rank']):<6}{row['feature'][:48]:<50}{row['importance_mean']:.4f}       {row['importance_std']:.4f}")

# Save full model results
top30_full.to_csv('top30_full_model_corrected.csv', index=False)
print(f"\n  Saved: top30_full_model_corrected.csv")

# =============================================================================
# NON-PARTISAN MODEL
# =============================================================================
print("\n" + "=" * 70)
print("PART B: NON-PARTISAN MODEL (Excluding Tautological Variables)")
print("=" * 70)

# Create non-partisan feature set
X_nonpartisan = X.drop(columns=cols_to_exclude, errors='ignore')
print(f"\n  Original features: {X.shape[1]:,}")
print(f"  After excluding partisan: {X_nonpartisan.shape[1]:,}")
print(f"  Features removed: {X.shape[1] - X_nonpartisan.shape[1]}")

# Split non-partisan data (same random state for comparability)
X_train_np = X_train.drop(columns=cols_to_exclude, errors='ignore')
X_test_np = X_test.drop(columns=cols_to_exclude, errors='ignore')

print("\nTraining non-partisan model...")
rf_nonpartisan = RandomForestClassifier(**rf_params)
rf_nonpartisan.fit(X_train_np, y_train, sample_weight=weights_train)

# Evaluate
y_pred_np = rf_nonpartisan.predict_proba(X_test_np)[:, 1]
auc_nonpartisan = roc_auc_score(y_test, y_pred_np, sample_weight=weights_test)
print(f"  Non-Partisan Model Test ROC-AUC: {auc_nonpartisan:.4f}")

# AUC comparison
auc_drop = auc_full - auc_nonpartisan
auc_drop_pct = 100 * auc_drop / auc_full

print("\n" + "-" * 60)
print("MODEL COMPARISON")
print("-" * 60)
print(f"  Full Model AUC:        {auc_full:.4f}")
print(f"  Non-Partisan AUC:      {auc_nonpartisan:.4f}")
print(f"  AUC Drop:              {auc_drop:.4f} ({auc_drop_pct:.1f}% relative decline)")

# Permutation importance for non-partisan model
print("\n" + "-" * 60)
print("Permutation Importance - Non-Partisan Model (50 repeats)")
print("-" * 60)
print("  Computing permutation importance...")

perm_imp_np = permutation_importance(
    rf_nonpartisan, X_test_np, y_test,
    n_repeats=50,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

# Create sorted importance DataFrame
imp_np_df = pd.DataFrame({
    'feature': X_nonpartisan.columns,
    'importance_mean': perm_imp_np.importances_mean,
    'importance_std': perm_imp_np.importances_std
}).sort_values('importance_mean', ascending=False)

top30_np = imp_np_df.head(30).copy()
top30_np['rank'] = range(1, 31)
top30_np = top30_np[['rank', 'feature', 'importance_mean', 'importance_std']]

print("\n  Top 30 Features (Non-Partisan Model):")
print("  " + "-" * 75)
print(f"  {'Rank':<6}{'Feature':<50}{'Importance':<12}{'Std':<10}")
print("  " + "-" * 75)
for _, row in top30_np.iterrows():
    print(f"  {int(row['rank']):<6}{row['feature'][:48]:<50}{row['importance_mean']:.4f}       {row['importance_std']:.4f}")

# Save non-partisan model results
top30_np.to_csv('top30_nonpartisan_model_corrected.csv', index=False)
print(f"\n  Saved: top30_nonpartisan_model_corrected.csv")

# =============================================================================
# VERIFICATION: Check for leakage
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Checking for Partisan Leakage in Non-Partisan Top 30")
print("=" * 70)

leaked_vars = []
for feature in top30_np['feature']:
    for prefix in exclude_prefixes:
        if feature.startswith(prefix):
            leaked_vars.append(feature)
            break

if leaked_vars:
    print("\n  WARNING: Partisan variables found in non-partisan top 30:")
    for var in leaked_vars:
        print(f"    - {var}")
else:
    print("\n  ✓ VERIFICATION PASSED: No partisan variables in non-partisan top 30")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
MODEL COMPARISON
----------------
                        Full Model      Non-Partisan Model
Features:               {X.shape[1]:,}             {X_nonpartisan.shape[1]:,}
Excluded:               0                 {len(cols_to_exclude)}
Test ROC-AUC:           {auc_full:.4f}            {auc_nonpartisan:.4f}
AUC Drop:               --                {auc_drop:.4f} ({auc_drop_pct:.1f}%)

EXCLUDED VARIABLE CATEGORIES
-----------------------------
- Candidate favorability (C2-C11): All response categories
- Party identification (C25, C26, C27): Self-ID, strength, lean
- Ideology (C31): Liberal-conservative scale
- Party evaluations (L46, L266, L267): Which party better on X
- Party favorability (L293, L294): 0-10 scales
- Derived party ID (C242_HID)
- Party support (LA204): Group party support

OUTPUT FILES
------------
- top30_full_model_corrected.csv
- top30_nonpartisan_model_corrected.csv
- excluded_partisan_columns.csv
""")

print("=" * 70)
print("Step 2.6 CORRECTED Analysis Complete")
print("=" * 70)
