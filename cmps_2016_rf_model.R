# ============================================================
# CMPS 2016 Random Forest Model
# Latino Trump Vote Prediction
# ============================================================

cat(strrep("=", 70), "\n")
cat("CMPS 2016 RANDOM FOREST MODEL\n")
cat("Latino Trump Vote Prediction\n")
cat(strrep("=", 70), "\n\n")

# Check for required packages
if (!require("randomForest", quietly = TRUE)) {
  cat("Installing randomForest package...\n")
  install.packages("randomForest", repos = "https://cloud.r-project.org")
  library(randomForest)
}

if (!require("pROC", quietly = TRUE)) {
  cat("Installing pROC package...\n")
  install.packages("pROC", repos = "https://cloud.r-project.org")
  library(pROC)
}

# ============================================================
# LOAD PREPARED DATA
# ============================================================
cat("LOADING PREPARED DATA\n")
cat(strrep("-", 40), "\n")

load("cmps_2016_rf_prepared_data.rda")
cat("Training set:", nrow(train_data), "x", ncol(train_data), "\n")
cat("Test set:", nrow(test_data), "x", ncol(test_data), "\n")
cat("Trump in training:", sum(train_data$trump_vote), "\n")
cat("Trump in test:", sum(test_data$trump_vote), "\n\n")

# ============================================================
# PREPARE DATA FOR RF
# ============================================================
cat("PREPARING DATA FOR RF\n")
cat(strrep("-", 40), "\n")

# Separate features and target
X_train <- train_data[, setdiff(names(train_data), "trump_vote")]
y_train <- as.factor(train_data$trump_vote)

X_test <- test_data[, setdiff(names(test_data), "trump_vote")]
y_test <- as.factor(test_data$trump_vote)

cat("Features:", ncol(X_train), "\n")
cat("Training samples:", nrow(X_train), "\n")
cat("Test samples:", nrow(X_test), "\n\n")

# ============================================================
# FIT RANDOM FOREST WITH CLASS WEIGHTS
# ============================================================
cat("FITTING RANDOM FOREST\n")
cat(strrep("-", 40), "\n")

# Calculate class weights (inverse of class frequency)
class_counts <- table(y_train)
class_weights <- max(class_counts) / class_counts
cat("Class weights:\n")
cat("  Class 0 (Other):", round(class_weights["0"], 3), "\n")
cat("  Class 1 (Trump):", round(class_weights["1"], 3), "\n\n")

# Fit RF with stratified sampling (sampsize) to handle imbalance
# Using classwt for weighting
set.seed(42)
cat("Training Random Forest (this may take a few minutes)...\n")

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train))),  # sqrt(p) for classification
  classwt = c("0" = 1, "1" = class_weights["1"]),  # Weight minority class
  importance = TRUE,
  do.trace = 100
)

cat("\nModel trained successfully!\n")
print(rf_model)

# ============================================================
# PREDICTIONS
# ============================================================
cat("\n\nPREDICTIONS\n")
cat(strrep("-", 40), "\n")

# Predict on test set
pred_class <- predict(rf_model, X_test, type = "response")
pred_prob <- predict(rf_model, X_test, type = "prob")[, "1"]

cat("Predictions generated for", length(pred_class), "test samples\n\n")

# ============================================================
# EVALUATION: CONFUSION MATRIX
# ============================================================
cat("CONFUSION MATRIX\n")
cat(strrep("-", 40), "\n")

conf_matrix <- table(Predicted = pred_class, Actual = y_test)
print(conf_matrix)

# Calculate metrics
TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrix)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)  # Sensitivity
specificity <- TN / (TN + FP)
f1 <- 2 * precision * recall / (precision + recall)

cat("\nMetrics:\n")
cat("  Accuracy:    ", round(accuracy, 4), "\n")
cat("  Precision:   ", round(precision, 4), "\n")
cat("  Recall:      ", round(recall, 4), "(Trump correctly identified)\n")
cat("  Specificity: ", round(specificity, 4), "\n")
cat("  F1 Score:    ", round(f1, 4), "\n")

# ============================================================
# EVALUATION: AUC-ROC
# ============================================================
cat("\n\nAUC-ROC\n")
cat(strrep("-", 40), "\n")

roc_obj <- roc(y_test, pred_prob, levels = c("0", "1"), direction = "<")
auc_value <- auc(roc_obj)
cat("AUC:", round(auc_value, 4), "\n")

# Save ROC curve data
roc_data <- data.frame(
  specificity = roc_obj$specificities,
  sensitivity = roc_obj$sensitivities
)
write.csv(roc_data, "cmps_2016_rf_roc_curve.csv", row.names = FALSE)
cat("ROC curve data saved to: cmps_2016_rf_roc_curve.csv\n")

# ============================================================
# VARIABLE IMPORTANCE
# ============================================================
cat("\n\nVARIABLE IMPORTANCE\n")
cat(strrep("-", 40), "\n")

# Get importance measures
importance_df <- data.frame(
  variable = rownames(rf_model$importance),
  MeanDecreaseAccuracy = rf_model$importance[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = rf_model$importance[, "MeanDecreaseGini"],
  stringsAsFactors = FALSE
)

# Sort by MeanDecreaseAccuracy
importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]

cat("Top 30 variables by MeanDecreaseAccuracy:\n")
cat(strrep("-", 70), "\n")
cat(sprintf("%-50s %15s %15s\n", "Variable", "DecAccuracy", "DecGini"))
cat(strrep("-", 70), "\n")

for (i in 1:min(30, nrow(importance_df))) {
  cat(sprintf("%-50s %15.4f %15.4f\n",
              substr(importance_df$variable[i], 1, 50),
              importance_df$MeanDecreaseAccuracy[i],
              importance_df$MeanDecreaseGini[i]))
}

# Save full importance
write.csv(importance_df, "cmps_2016_rf_importance.csv", row.names = FALSE)
cat("\nFull importance saved to: cmps_2016_rf_importance.csv\n")

# ============================================================
# SAVE MODEL
# ============================================================
cat("\n\nSAVING MODEL\n")
cat(strrep("-", 40), "\n")

save(rf_model, file = "cmps_2016_rf_model.rda")
cat("Model saved to: cmps_2016_rf_model.rda\n")

# Save predictions
predictions_df <- data.frame(
  actual = y_test,
  predicted = pred_class,
  prob_trump = pred_prob
)
write.csv(predictions_df, "cmps_2016_rf_predictions.csv", row.names = FALSE)
cat("Predictions saved to: cmps_2016_rf_predictions.csv\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 70), "\n")
cat("MODEL SUMMARY\n")
cat(strrep("=", 70), "\n")
cat("Algorithm:      Random Forest\n")
cat("Trees:          500\n")
cat("Features:      ", ncol(X_train), "\n")
cat("Class weights:  1.0 (Other), ", round(class_weights["1"], 2), " (Trump)\n")
cat("\n")
cat("Test Set Performance:\n")
cat("  AUC:          ", round(auc_value, 4), "\n")
cat("  Accuracy:     ", round(accuracy, 4), "\n")
cat("  Precision:    ", round(precision, 4), "\n")
cat("  Recall:       ", round(recall, 4), "\n")
cat("  F1:           ", round(f1, 4), "\n")
cat(strrep("=", 70), "\n")
