# ============================================================
# CMPS 2016 Random Forest Pipeline
# Latino Trump Vote Prediction
# ============================================================

cat(strrep("=", 70), "\n")
cat("CMPS 2016 RANDOM FOREST PIPELINE\n")
cat("Latino Trump Vote Prediction\n")
cat(strrep("=", 70), "\n\n")

# ============================================================
# STEP 0: LOAD DATA
# ============================================================
cat("STEP 0: LOAD DATA\n")
cat(strrep("-", 40), "\n")

load("CMPS_2016_raw.rda")
df <- da38040.0001
cat("Loaded:", nrow(df), "x", ncol(df), "\n\n")

# ============================================================
# STEP 1: FILTER TO LATINOS WITH VALID VOTE CHOICE
# ============================================================
cat("STEP 1: FILTER TO LATINOS WITH VALID VOTE CHOICE\n")
cat(strrep("-", 40), "\n")

# Filter to Latino respondents
latinos <- df[df$ETHNIC_QUOTA == "(2) Hispanic or Latino", ]
cat("Latino respondents:", nrow(latinos), "\n")

# Filter to valid vote choice (exclude NA)
latinos_voters <- latinos[!is.na(latinos$C14), ]
cat("With valid vote choice:", nrow(latinos_voters), "\n\n")

# ============================================================
# STEP 2: CREATE BINARY DV
# ============================================================
cat("STEP 2: CREATE BINARY DV\n")
cat(strrep("-", 40), "\n")

latinos_voters$trump_vote <- ifelse(latinos_voters$C14 == "(2) Donald Trump", 1, 0)
cat("Trump (1):", sum(latinos_voters$trump_vote), "\n")
cat("Other (0):", sum(latinos_voters$trump_vote == 0), "\n")
cat("Trump %:", round(100 * mean(latinos_voters$trump_vote), 2), "%\n\n")

# ============================================================
# STEP 3: DROP VARIABLES
# ============================================================
cat("STEP 3: DROP VARIABLES\n")
cat(strrep("-", 40), "\n")

# Start with all variables except trump_vote (which we just created)
all_vars <- setdiff(names(latinos_voters), "trump_vote")

# 3a. Identifiers, weights, timestamps
drop_identifiers <- c("RESPID", "INTERVIEW_START", "INTERVIEW_END", "DIFF_DATE",
                      "ZIPCODE", "CITY_NAME", "COUNTY_NAME",
                      "WEIGHT", "NAT_WEIGHT", "C14", "ETHNIC_QUOTA",
                      "FLAG_DESCRIPTION", "RESPONSE_MINUTES")
cat("Dropping identifiers/weights/DV:", length(drop_identifiers), "\n")

# 3b. Variables with >50% missing
missingness <- sapply(latinos_voters[, all_vars], function(x) 100 * sum(is.na(x)) / nrow(latinos_voters))
drop_high_missing <- names(missingness[missingness > 50])
cat("Dropping >50% missing:", length(drop_high_missing), "\n")

# 3c. Open-text factor variables (50+ levels)
# These are free-response fields, not structured predictors
drop_open_text <- c("LA201", "C242A", "C242B", "C50", "C391_14_OTHER",
                    "C259_OTHER", "C34_OTHER", "C129_8_OTHER", "C343",
                    "C373_7_OTHER", "LA201A", "C344", "C347_STATE",
                    "S10_21_OTHER", "C35_OTHER", "C375_6_OTHER",
                    "C237_7_OTHER", "S8_14_OTHER", "S9_15_OTHER",
                    "C85_8_OTHER", "TIME_304", "C347_CITY", "BA132")
cat("Dropping open-text factors:", length(drop_open_text), "\n")

# 3d. Split indicator variables
drop_splits <- all_vars[grepl("^SPLIT", all_vars)]
cat("Dropping split indicators:", length(drop_splits), "\n")

# 3e. Race-specific variables (not asked of Latinos - will be all NA)
drop_race_specific <- all_vars[grepl("^A[0-9]|^B[0-9]|^BW[0-9]|^BA[0-9]", all_vars)]
drop_race_specific <- drop_race_specific[sapply(latinos_voters[, drop_race_specific],
                                                 function(x) sum(is.na(x)) / length(x) > 0.95)]
cat("Dropping race-specific (>95% NA):", length(drop_race_specific), "\n")

# Combine all drops
drop_all <- unique(c(drop_identifiers, drop_high_missing, drop_open_text,
                     drop_splits, drop_race_specific))
cat("\nTotal variables to drop:", length(drop_all), "\n")

# Apply drops
keep_vars <- setdiff(all_vars, drop_all)
rf_data <- latinos_voters[, c(keep_vars, "trump_vote")]
cat("Remaining variables:", ncol(rf_data) - 1, "+ DV\n\n")

# ============================================================
# STEP 4: IDENTIFY VARIABLE TYPES
# ============================================================
cat("STEP 4: IDENTIFY VARIABLE TYPES\n")
cat(strrep("-", 40), "\n")

predictors <- setdiff(names(rf_data), "trump_vote")
factor_vars <- predictors[sapply(rf_data[, predictors], is.factor)]
numeric_vars <- predictors[sapply(rf_data[, predictors], is.numeric)]

cat("Factor variables:", length(factor_vars), "\n")
cat("Numeric variables:", length(numeric_vars), "\n\n")

# ============================================================
# STEP 5: IMPUTATION (before encoding)
# ============================================================
cat("STEP 5: IMPUTATION\n")
cat(strrep("-", 40), "\n")

# Mode function for factors
get_mode <- function(x) {
  tbl <- table(x)
  if (length(tbl) == 0) return(NA)
  names(tbl)[which.max(tbl)]
}

# Count missing before imputation
missing_before <- sum(sapply(rf_data[, predictors], function(x) sum(is.na(x))))
cat("Total missing values before:", missing_before, "\n")

# Impute numeric variables with median
for (v in numeric_vars) {
  if (any(is.na(rf_data[[v]]))) {
    med_val <- median(rf_data[[v]], na.rm = TRUE)
    rf_data[[v]][is.na(rf_data[[v]])] <- med_val
  }
}

# Impute factor variables with mode
for (v in factor_vars) {
  if (any(is.na(rf_data[[v]]))) {
    mode_val <- get_mode(rf_data[[v]])
    if (!is.na(mode_val)) {
      rf_data[[v]][is.na(rf_data[[v]])] <- mode_val
    }
  }
}

# Count missing after imputation
missing_after <- sum(sapply(rf_data[, predictors], function(x) sum(is.na(x))))
cat("Total missing values after:", missing_after, "\n\n")

# ============================================================
# STEP 5b: REMOVE CONSTANT/SINGLE-LEVEL FACTORS
# ============================================================
cat("STEP 5b: REMOVE CONSTANT/SINGLE-LEVEL FACTORS\n")
cat(strrep("-", 40), "\n")

# Find factors with <2 levels (constant after imputation)
constant_factors <- c()
for (v in factor_vars) {
  n_levels <- length(unique(na.omit(rf_data[[v]])))
  if (n_levels < 2) {
    constant_factors <- c(constant_factors, v)
  }
}

if (length(constant_factors) > 0) {
  cat("Removing", length(constant_factors), "constant factors:\n")
  for (v in constant_factors) {
    cat("  ", v, "\n")
  }
  rf_data <- rf_data[, !(names(rf_data) %in% constant_factors)]
  factor_vars <- setdiff(factor_vars, constant_factors)
  predictors <- setdiff(predictors, constant_factors)
}
cat("Remaining factors:", length(factor_vars), "\n\n")

# Also check for constant numeric variables
constant_numeric <- c()
for (v in numeric_vars) {
  if (length(unique(na.omit(rf_data[[v]]))) < 2) {
    constant_numeric <- c(constant_numeric, v)
  }
}

if (length(constant_numeric) > 0) {
  cat("Removing", length(constant_numeric), "constant numeric vars:\n")
  for (v in constant_numeric) {
    cat("  ", v, "\n")
  }
  rf_data <- rf_data[, !(names(rf_data) %in% constant_numeric)]
  numeric_vars <- setdiff(numeric_vars, constant_numeric)
  predictors <- setdiff(predictors, constant_numeric)
}

# ============================================================
# STEP 6: ONE-HOT ENCODING
# ============================================================
cat("STEP 6: ONE-HOT ENCODING\n")
cat(strrep("-", 40), "\n")

# Create model matrix (one-hot encoding)
# This automatically creates dummy variables for factors

# First, create a formula with all predictors
formula_str <- paste("~ 0 +", paste(predictors, collapse = " + "))

# Create the design matrix
X <- model.matrix(as.formula(formula_str), data = rf_data)

cat("Original predictors:", length(predictors), "\n")
cat("After one-hot encoding:", ncol(X), "columns\n")

# Combine with DV
rf_encoded <- data.frame(X)
rf_encoded$trump_vote <- rf_data$trump_vote

cat("Final encoded dataset:", nrow(rf_encoded), "x", ncol(rf_encoded), "\n\n")

# ============================================================
# STEP 7: TRAIN/TEST SPLIT (80/20 stratified)
# ============================================================
cat("STEP 7: TRAIN/TEST SPLIT\n")
cat(strrep("-", 40), "\n")

set.seed(42)

# Stratified split
trump_idx <- which(rf_encoded$trump_vote == 1)
other_idx <- which(rf_encoded$trump_vote == 0)

# 80% for training
train_trump <- sample(trump_idx, size = round(0.8 * length(trump_idx)))
train_other <- sample(other_idx, size = round(0.8 * length(other_idx)))
train_idx <- c(train_trump, train_other)

# Remaining 20% for test
test_idx <- setdiff(1:nrow(rf_encoded), train_idx)

train_data <- rf_encoded[train_idx, ]
test_data <- rf_encoded[test_idx, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("  Trump:", sum(train_data$trump_vote), "(", round(100*mean(train_data$trump_vote), 2), "%)\n")
cat("  Other:", sum(train_data$trump_vote == 0), "\n")
cat("Test set:", nrow(test_data), "observations\n")
cat("  Trump:", sum(test_data$trump_vote), "(", round(100*mean(test_data$trump_vote), 2), "%)\n")
cat("  Other:", sum(test_data$trump_vote == 0), "\n\n")

# ============================================================
# SAVE PREPARED DATA
# ============================================================
cat("SAVING PREPARED DATA\n")
cat(strrep("-", 40), "\n")

# Save for use with RF fitting
save(train_data, test_data, rf_encoded, file = "cmps_2016_rf_prepared_data.rda")
cat("Saved: cmps_2016_rf_prepared_data.rda\n")

# Save variable mapping (original names to encoded names)
var_mapping <- data.frame(
  encoded_name = colnames(X),
  stringsAsFactors = FALSE
)
write.csv(var_mapping, "cmps_2016_rf_variable_mapping.csv", row.names = FALSE)
cat("Saved: cmps_2016_rf_variable_mapping.csv\n")

# ============================================================
# SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 70), "\n")
cat("PIPELINE SUMMARY\n")
cat(strrep("=", 70), "\n")
cat("Original dataset:        ", nrow(df), "x", ncol(df), "\n")
cat("Latino voters:           ", nrow(latinos_voters), "\n")
cat("After variable drops:    ", length(keep_vars), "predictors\n")
cat("After one-hot encoding:  ", ncol(X), "features\n")
cat("\n")
cat("Training set:            ", nrow(train_data), "(Trump:", sum(train_data$trump_vote), ")\n")
cat("Test set:                ", nrow(test_data), "(Trump:", sum(test_data$trump_vote), ")\n")
cat("\n")
cat("Class balance:           ", round(100*mean(rf_encoded$trump_vote), 2), "% Trump\n")
cat(strrep("=", 70), "\n")

cat("\n\nNext: Run cmps_2016_rf_model.R to fit Random Forest\n")
