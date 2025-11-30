"""
CMPS 2016 Random Forest Model
Latino Trump Vote Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CMPS 2016 RANDOM FOREST MODEL")
print("Latino Trump Vote Prediction")
print("=" * 70)
print()

# ============================================================
# LOAD PREPARED DATA (from R pipeline)
# ============================================================
print("LOADING PREPARED DATA")
print("-" * 40)

# Load the R data using pyreadr
import pyreadr
result = pyreadr.read_r('cmps_2016_rf_prepared_data.rda')

# Get train and test data
train_data = result['train_data']
test_data = result['test_data']

print(f"Training set: {train_data.shape}")
print(f"Test set: {test_data.shape}")
print(f"Trump in training: {train_data['trump_vote'].sum()}")
print(f"Trump in test: {test_data['trump_vote'].sum()}")
print()

# ============================================================
# PREPARE DATA FOR RF
# ============================================================
print("PREPARING DATA FOR RF")
print("-" * 40)

# Separate features and target
X_train = train_data.drop('trump_vote', axis=1)
y_train = train_data['trump_vote']

X_test = test_data.drop('trump_vote', axis=1)
y_test = test_data['trump_vote']

print(f"Features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print()

# ============================================================
# FIT RANDOM FOREST WITH CLASS WEIGHTS
# ============================================================
print("FITTING RANDOM FOREST")
print("-" * 40)

# Calculate class weights
n_other = (y_train == 0).sum()
n_trump = (y_train == 1).sum()
weight_trump = n_other / n_trump

print(f"Class counts: Other={n_other}, Trump={n_trump}")
print(f"Class weight for Trump: {weight_trump:.3f}")
print()

print("Training Random Forest (this may take a few minutes)...")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

print("\nModel trained successfully!")
print()

# ============================================================
# PREDICTIONS
# ============================================================
print("PREDICTIONS")
print("-" * 40)

pred_class = rf_model.predict(X_test)
pred_prob = rf_model.predict_proba(X_test)[:, 1]

print(f"Predictions generated for {len(pred_class)} test samples")
print()

# ============================================================
# EVALUATION: CONFUSION MATRIX
# ============================================================
print("CONFUSION MATRIX")
print("-" * 40)

cm = confusion_matrix(y_test, pred_class)
print("             Predicted")
print("            Other  Trump")
print(f"Actual Other  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Trump  {cm[1,0]:5d}  {cm[1,1]:5d}")
print()

# Calculate metrics
accuracy = accuracy_score(y_test, pred_class)
precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)
f1 = f1_score(y_test, pred_class)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("Metrics:")
print(f"  Accuracy:    {accuracy:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f} (Trump correctly identified)")
print(f"  Specificity: {specificity:.4f}")
print(f"  F1 Score:    {f1:.4f}")
print()

# ============================================================
# EVALUATION: AUC-ROC
# ============================================================
print("AUC-ROC")
print("-" * 40)

auc_value = roc_auc_score(y_test, pred_prob)
print(f"AUC: {auc_value:.4f}")

# Save ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
roc_df = pd.DataFrame({
    'fpr': fpr,
    'tpr': tpr,
    'threshold': thresholds
})
roc_df.to_csv('cmps_2016_rf_roc_curve.csv', index=False)
print("ROC curve data saved to: cmps_2016_rf_roc_curve.csv")
print()

# ============================================================
# VARIABLE IMPORTANCE
# ============================================================
print("VARIABLE IMPORTANCE")
print("-" * 40)

importance_df = pd.DataFrame({
    'variable': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 variables by importance:")
print("-" * 70)
print(f"{'Variable':<55} {'Importance':>12}")
print("-" * 70)

for i, row in importance_df.head(30).iterrows():
    print(f"{row['variable'][:55]:<55} {row['importance']:>12.6f}")

# Save full importance
importance_df.to_csv('cmps_2016_rf_importance.csv', index=False)
print(f"\nFull importance saved to: cmps_2016_rf_importance.csv")
print()

# ============================================================
# SHAP VALUES
# ============================================================
print("SHAP VALUES")
print("-" * 40)

print("Computing SHAP values (this may take a while)...")

try:
    # Use a smaller sample for SHAP to speed up computation
    shap_sample_size = min(500, len(X_test))
    X_shap = X_test.iloc[:shap_sample_size]

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values is a list [class_0, class_1]
    # We want class 1 (Trump)
    if isinstance(shap_values, list):
        shap_values_trump = shap_values[1]
    else:
        # New SHAP versions return 3D array for multi-output
        if len(shap_values.shape) == 3:
            shap_values_trump = shap_values[:, :, 1]
        else:
            shap_values_trump = shap_values

    # Ensure 2D array
    if len(shap_values_trump.shape) == 1:
        shap_values_trump = shap_values_trump.reshape(-1, 1)

    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values_trump).mean(axis=0)

    # Create dataframe
    shap_importance = pd.DataFrame({
        'variable': X_train.columns.tolist(),
        'mean_abs_shap': mean_shap.flatten()
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 30 variables by mean |SHAP|:")
    print("-" * 70)
    print(f"{'Variable':<55} {'Mean |SHAP|':>12}")
    print("-" * 70)

    for _, row in shap_importance.head(30).iterrows():
        print(f"{row['variable'][:55]:<55} {row['mean_abs_shap']:>12.6f}")

    # Save SHAP importance
    shap_importance.to_csv('cmps_2016_rf_shap_importance.csv', index=False)
    print(f"\nSHAP importance saved to: cmps_2016_rf_shap_importance.csv")

    # Save raw SHAP values for the sample
    shap_df = pd.DataFrame(shap_values_trump, columns=X_train.columns)
    shap_df.to_csv('cmps_2016_rf_shap_values.csv', index=False)
    print(f"Raw SHAP values saved to: cmps_2016_rf_shap_values.csv")

except Exception as e:
    print(f"SHAP computation failed: {e}")
    print("Skipping SHAP values, using feature importance instead.")

print()

# ============================================================
# SAVE PREDICTIONS
# ============================================================
print("SAVING OUTPUTS")
print("-" * 40)

predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': pred_class,
    'prob_trump': pred_prob
})
predictions_df.to_csv('cmps_2016_rf_predictions.csv', index=False)
print("Predictions saved to: cmps_2016_rf_predictions.csv")

# Save model (using joblib)
import joblib
joblib.dump(rf_model, 'cmps_2016_rf_model.joblib')
print("Model saved to: cmps_2016_rf_model.joblib")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
print(f"Algorithm:      Random Forest")
print(f"Trees:          500")
print(f"Features:       {X_train.shape[1]}")
print(f"Class weights:  balanced")
print()
print("Test Set Performance:")
print(f"  AUC:          {auc_value:.4f}")
print(f"  Accuracy:     {accuracy:.4f}")
print(f"  Precision:    {precision:.4f}")
print(f"  Recall:       {recall:.4f}")
print(f"  F1:           {f1:.4f}")
print("=" * 70)
