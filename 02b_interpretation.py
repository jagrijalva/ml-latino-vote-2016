"""
PHASE 2 Step 2.6: Model Interpretation
======================================
Part A: Full Model Interpretation
Part B: Non-Partisan Model Interpretation

Compares predictive power with and without partisan variables.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load Data
# =============================================================================

print("=" * 70)
print("PHASE 2 Step 2.6: Model Interpretation")
print("=" * 70)

print("\nLoading data...")
X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet')['trump_vote']
weights = pd.read_parquet('cmps_2016_weights.parquet')['survey_wt']

print(f"  X shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")

# Recreate train/test split (same as original)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# =============================================================================
# PART A: Full Model Interpretation
# =============================================================================

print("\n" + "=" * 70)
print("PART A: Full Model Interpretation")
print("=" * 70)

# Train full model
print("\nTraining full model...")
full_model = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
full_model.fit(X_train, y_train, sample_weight=weights_train)

# Evaluate
y_test_pred_proba = full_model.predict_proba(X_test)[:, 1]
full_auc = roc_auc_score(y_test, y_test_pred_proba, sample_weight=weights_test)
print(f"  Full Model Test ROC-AUC: {full_auc:.4f}")

# --- Permutation Importance (50 repeats) ---
print("\n" + "-" * 60)
print("Permutation Importance (50 repeats)")
print("-" * 60)

print("  Computing permutation importance (this may take several minutes)...")
perm_imp_full = permutation_importance(
    full_model, X_test, y_test,
    n_repeats=50,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# Create DataFrame
perm_imp_full_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm_imp_full.importances_mean,
    'importance_std': perm_imp_full.importances_std
}).sort_values('importance_mean', ascending=False)

print("\n  Top 30 Features (Full Model):")
print(f"  {'Rank':<6} {'Feature':<55} {'Importance':<12} {'Std':<10}")
print("  " + "-" * 83)

top30_full = perm_imp_full_df.head(30).copy()
for i, (_, row) in enumerate(top30_full.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<55} {row['importance_mean']:<12.4f} {row['importance_std']:<10.4f}")

# --- SHAP Analysis (500 observations) ---
print("\n" + "-" * 60)
print("SHAP Analysis (500 test observations)")
print("-" * 60)

print("  Creating TreeExplainer...")
explainer_full = shap.TreeExplainer(full_model)

n_shap = min(500, len(X_test))
X_shap_full = X_test.iloc[:n_shap]

print(f"  Computing SHAP values for {n_shap} observations...")
shap_values_full = explainer_full.shap_values(X_shap_full)

# Handle different SHAP output formats
if isinstance(shap_values_full, list):
    shap_values_full_class1 = shap_values_full[1]
else:
    shap_values_full_class1 = shap_values_full

if len(shap_values_full_class1.shape) == 3:
    shap_values_full_class1 = shap_values_full_class1[:, :, 1]

# Mean absolute SHAP
mean_abs_shap_full = np.abs(shap_values_full_class1).mean(axis=0)
if len(mean_abs_shap_full.shape) > 1:
    mean_abs_shap_full = mean_abs_shap_full.flatten()

shap_full_df = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': mean_abs_shap_full
}).sort_values('mean_abs_shap', ascending=False)

print("\n  Top 30 Features by SHAP (Full Model):")
print(f"  {'Rank':<6} {'Feature':<55} {'Mean |SHAP|':<12}")
print("  " + "-" * 73)

for i, (_, row) in enumerate(shap_full_df.head(30).iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<55} {row['mean_abs_shap']:<12.4f}")

# Save SHAP beeswarm plot
print("\n  Generating SHAP summary plot...")
plt.figure(figsize=(14, 12))
shap.summary_plot(shap_values_full_class1, X_shap_full, show=False, max_display=30)
plt.tight_layout()
plt.savefig('shap_full_model_top30.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap_full_model_top30.png")

# Save top 30 full model
top30_full['rank'] = range(1, 31)
top30_full = top30_full[['rank', 'feature', 'importance_mean', 'importance_std']]
top30_full.to_csv('top30_full_model.csv', index=False)
print("  Saved: top30_full_model.csv")

# =============================================================================
# PART B: Non-Partisan Model
# =============================================================================

print("\n" + "=" * 70)
print("PART B: Non-Partisan Model")
print("=" * 70)

# Define partisan variable prefixes to remove
partisan_prefixes = [
    'C25_',   # Party registration
    'C26_',   # Strong partisan
    'C27_',   # Lean partisan
    'C31_',   # Ideology
    'L46_',   # Party better on immigration
    'L266_',  # Party better for Latinos
    'L267_',  # Party better on values
    'C2_',    # Clinton favorability
    'C3_',    # Sanders favorability
    'C4_',    # Trump favorability
    'C5_',    # Cruz favorability
    'C8_',    # Bill Clinton favorability
    'C9_',    # Obama favorability
]

# Also check for C242_HID (party identification) if present
partisan_prefixes.append('C242_HID_')

# Identify partisan columns
partisan_cols = []
for col in X.columns:
    for prefix in partisan_prefixes:
        if col.startswith(prefix):
            partisan_cols.append(col)
            break

partisan_cols = list(set(partisan_cols))  # Remove duplicates

print(f"\n  Identified {len(partisan_cols)} partisan columns to remove:")
print(f"  Prefixes: {partisan_prefixes}")

# Show breakdown by prefix
print("\n  Breakdown by variable:")
for prefix in partisan_prefixes:
    count = sum(1 for col in partisan_cols if col.startswith(prefix))
    if count > 0:
        print(f"    {prefix[:-1]}: {count} columns")

# Remove partisan columns
X_train_np = X_train.drop(columns=partisan_cols)
X_test_np = X_test.drop(columns=partisan_cols)

print(f"\n  Original features: {X_train.shape[1]:,}")
print(f"  After removing partisan: {X_train_np.shape[1]:,}")
print(f"  Features removed: {len(partisan_cols)}")

# --- Retrain RF on non-partisan features ---
print("\n" + "-" * 60)
print("Retraining RF on Non-Partisan Features")
print("-" * 60)

print("  Training non-partisan model...")
np_model = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
np_model.fit(X_train_np, y_train, sample_weight=weights_train)

# Evaluate
y_test_pred_proba_np = np_model.predict_proba(X_test_np)[:, 1]
np_auc = roc_auc_score(y_test, y_test_pred_proba_np, sample_weight=weights_test)

print(f"\n  Non-Partisan Model Test ROC-AUC: {np_auc:.4f}")

# AUC comparison
auc_drop = full_auc - np_auc
auc_drop_pct = (auc_drop / full_auc) * 100

print("\n" + "-" * 60)
print("AUC COMPARISON")
print("-" * 60)
print(f"  Full Model AUC:        {full_auc:.4f}")
print(f"  Non-Partisan AUC:      {np_auc:.4f}")
print(f"  AUC Drop:              {auc_drop:.4f} ({auc_drop_pct:.1f}% relative decline)")

# --- Permutation Importance (50 repeats) ---
print("\n" + "-" * 60)
print("Permutation Importance - Non-Partisan (50 repeats)")
print("-" * 60)

print("  Computing permutation importance...")
perm_imp_np = permutation_importance(
    np_model, X_test_np, y_test,
    n_repeats=50,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

perm_imp_np_df = pd.DataFrame({
    'feature': X_test_np.columns,
    'importance_mean': perm_imp_np.importances_mean,
    'importance_std': perm_imp_np.importances_std
}).sort_values('importance_mean', ascending=False)

print("\n  Top 30 Features (Non-Partisan Model):")
print(f"  {'Rank':<6} {'Feature':<55} {'Importance':<12} {'Std':<10}")
print("  " + "-" * 83)

top30_np = perm_imp_np_df.head(30).copy()
for i, (_, row) in enumerate(top30_np.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<55} {row['importance_mean']:<12.4f} {row['importance_std']:<10.4f}")

# --- SHAP Analysis (500 observations) ---
print("\n" + "-" * 60)
print("SHAP Analysis - Non-Partisan (500 test observations)")
print("-" * 60)

print("  Creating TreeExplainer...")
explainer_np = shap.TreeExplainer(np_model)

X_shap_np = X_test_np.iloc[:n_shap]

print(f"  Computing SHAP values for {n_shap} observations...")
shap_values_np = explainer_np.shap_values(X_shap_np)

if isinstance(shap_values_np, list):
    shap_values_np_class1 = shap_values_np[1]
else:
    shap_values_np_class1 = shap_values_np

if len(shap_values_np_class1.shape) == 3:
    shap_values_np_class1 = shap_values_np_class1[:, :, 1]

mean_abs_shap_np = np.abs(shap_values_np_class1).mean(axis=0)
if len(mean_abs_shap_np.shape) > 1:
    mean_abs_shap_np = mean_abs_shap_np.flatten()

shap_np_df = pd.DataFrame({
    'feature': X_test_np.columns,
    'mean_abs_shap': mean_abs_shap_np
}).sort_values('mean_abs_shap', ascending=False)

print("\n  Top 30 Features by SHAP (Non-Partisan Model):")
print(f"  {'Rank':<6} {'Feature':<55} {'Mean |SHAP|':<12}")
print("  " + "-" * 73)

for i, (_, row) in enumerate(shap_np_df.head(30).iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<55} {row['mean_abs_shap']:<12.4f}")

# Save SHAP beeswarm plot
print("\n  Generating SHAP summary plot...")
plt.figure(figsize=(14, 12))
shap.summary_plot(shap_values_np_class1, X_shap_np, show=False, max_display=30)
plt.tight_layout()
plt.savefig('shap_nonpartisan_model_top30.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap_nonpartisan_model_top30.png")

# Save top 30 non-partisan model
top30_np['rank'] = range(1, 31)
top30_np = top30_np[['rank', 'feature', 'importance_mean', 'importance_std']]
top30_np.to_csv('top30_nonpartisan_model.csv', index=False)
print("  Saved: top30_nonpartisan_model.csv")

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
Features:               {X_train.shape[1]:,}             {X_train_np.shape[1]:,}
Test ROC-AUC:           {full_auc:.4f}            {np_auc:.4f}
AUC Drop:               --                {auc_drop:.4f} ({auc_drop_pct:.1f}%)

INTERPRETATION
--------------
- Removing {len(partisan_cols)} partisan/favorability variables causes {auc_drop_pct:.1f}% relative AUC decline
- Non-partisan model still achieves {np_auc:.4f} AUC
- This indicates substantial predictive power from non-partisan variables

OUTPUT FILES
------------
- top30_full_model.csv           (Top 30 predictors, full model)
- top30_nonpartisan_model.csv    (Top 30 predictors, non-partisan model)
- shap_full_model_top30.png      (SHAP beeswarm, full model)
- shap_nonpartisan_model_top30.png (SHAP beeswarm, non-partisan model)
""")

# Save partisan columns list for reference
pd.DataFrame({'partisan_column': partisan_cols}).to_csv('partisan_columns_removed.csv', index=False)
print("  Saved: partisan_columns_removed.csv")

print("\n" + "=" * 70)
print("Step 2.6 Complete")
print("=" * 70)
