# ============================================================
# CMPS 2016 Factor Encoding Diagnostic
# Analyzes factor variables for one-hot encoding preparation
# ============================================================

# Load the RF-ready data (before >50% removal)
load("cmps_2016_latino_rf_data.rda")

cat(strrep("=", 70), "\n")
cat("FACTOR ENCODING DIAGNOSTIC FOR LATINO RF DATASET\n")
cat(strrep("=", 70), "\n\n")

# ============================================================
# STEP 1: IDENTIFY USABLE FACTOR VARIABLES
# ============================================================

# Get all predictors (exclude DV)
predictors <- setdiff(names(rf_data), "trump_vote")

# Calculate missingness
missingness <- sapply(rf_data[, predictors, drop=FALSE], function(x) 100 * sum(is.na(x)) / nrow(rf_data))

# Keep only variables with <=50% missing
usable_vars <- names(missingness[missingness <= 50])
cat("Total usable predictors (<=50% missing):", length(usable_vars), "\n")

# Identify factor variables among usable
factor_vars <- usable_vars[sapply(rf_data[, usable_vars, drop=FALSE], is.factor)]
numeric_vars <- usable_vars[sapply(rf_data[, usable_vars, drop=FALSE], is.numeric)]

cat("Factor variables:", length(factor_vars), "\n")
cat("Numeric variables:", length(numeric_vars), "\n\n")

# ============================================================
# STEP 2: ANALYZE FACTOR LEVELS
# ============================================================
cat(strrep("-", 70), "\n")
cat("FACTOR LEVEL ANALYSIS\n")
cat(strrep("-", 70), "\n\n")

# For each factor, count unique levels (excluding NA)
factor_info <- data.frame(
  variable = factor_vars,
  n_levels = sapply(rf_data[, factor_vars, drop=FALSE], function(x) length(unique(na.omit(x)))),
  pct_missing = sapply(rf_data[, factor_vars, drop=FALSE], function(x) round(100 * sum(is.na(x)) / length(x), 2)),
  stringsAsFactors = FALSE
)

# Sort by number of levels (descending)
factor_info <- factor_info[order(-factor_info$n_levels), ]

# ============================================================
# DISTRIBUTION OF LEVEL COUNTS
# ============================================================
cat("DISTRIBUTION OF FACTOR LEVELS\n")
cat(strrep("-", 45), "\n")

# Create bins
factor_info$level_bin <- cut(factor_info$n_levels,
                              breaks = c(0, 2, 5, 10, 20, 50, Inf),
                              labels = c("2 levels", "3-5 levels", "6-10 levels",
                                        "11-20 levels", "21-50 levels", "50+ levels"))

bin_counts <- table(factor_info$level_bin)
bin_pct <- round(100 * bin_counts / sum(bin_counts), 1)

cat(sprintf("%-15s %8s %10s %15s\n", "Level Range", "Count", "Percent", "One-Hot Cols"))
cat(strrep("-", 55), "\n")

# Calculate one-hot columns for each bin
onehot_cols <- c()
for (i in 1:length(bin_counts)) {
  bin_name <- names(bin_counts)[i]
  bin_vars <- factor_info$variable[factor_info$level_bin == bin_name]
  bin_levels <- factor_info$n_levels[factor_info$level_bin == bin_name]
  # One-hot creates (n_levels - 1) columns per factor
  total_cols <- sum(bin_levels - 1)
  onehot_cols <- c(onehot_cols, total_cols)
  cat(sprintf("%-15s %8d %9.1f%% %15d\n", bin_name, bin_counts[i], bin_pct[i], total_cols))
}
cat(strrep("-", 55), "\n")
cat(sprintf("%-15s %8d %9.1f%% %15d\n", "TOTAL", sum(bin_counts), 100, sum(onehot_cols)))

# ============================================================
# HIGH-LEVEL FACTORS (10+ levels)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("FACTORS WITH 10+ LEVELS (Candidates for Consolidation)\n")
cat(strrep("-", 70), "\n\n")

high_level <- factor_info[factor_info$n_levels >= 10, ]
cat("Total factors with 10+ levels:", nrow(high_level), "\n\n")

if (nrow(high_level) > 0) {
  cat(sprintf("%-40s %8s %10s\n", "Variable", "Levels", "Pct_Miss"))
  cat(strrep("-", 60), "\n")
  for (i in 1:nrow(high_level)) {
    cat(sprintf("%-40s %8d %9.1f%%\n",
                substr(high_level$variable[i], 1, 40),
                high_level$n_levels[i],
                high_level$pct_missing[i]))
  }
}

# ============================================================
# RARE LEVELS ANALYSIS (<1% of cases)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("RARE LEVELS ANALYSIS (<1% of cases = <30 observations)\n")
cat(strrep("-", 70), "\n\n")

# For each factor, count levels with <1% (30 cases)
threshold <- 0.01 * nrow(rf_data)  # 30 cases

rare_level_info <- data.frame(
  variable = character(),
  n_levels = integer(),
  n_rare_levels = integer(),
  rare_level_names = character(),
  stringsAsFactors = FALSE
)

for (v in factor_vars) {
  tbl <- table(rf_data[[v]])
  n_levels <- length(tbl)
  rare_levels <- names(tbl[tbl < threshold])
  n_rare <- length(rare_levels)

  if (n_rare > 0) {
    rare_level_info <- rbind(rare_level_info, data.frame(
      variable = v,
      n_levels = n_levels,
      n_rare_levels = n_rare,
      rare_level_names = paste(head(rare_levels, 5), collapse = "; "),
      stringsAsFactors = FALSE
    ))
  }
}

rare_level_info <- rare_level_info[order(-rare_level_info$n_rare_levels), ]

cat("Factors with rare levels (<30 cases):", nrow(rare_level_info), "of", length(factor_vars), "\n\n")

# Summary of rare levels
cat("Distribution of rare level counts:\n")
rare_bins <- cut(rare_level_info$n_rare_levels,
                 breaks = c(0, 1, 3, 5, 10, Inf),
                 labels = c("1 rare", "2-3 rare", "4-5 rare", "6-10 rare", "10+ rare"))
print(table(rare_bins))

cat("\n\nTop 30 factors by number of rare levels:\n")
cat(strrep("-", 70), "\n")
cat(sprintf("%-35s %8s %10s %s\n", "Variable", "Levels", "Rare", "Examples"))
cat(strrep("-", 70), "\n")

for (i in 1:min(30, nrow(rare_level_info))) {
  cat(sprintf("%-35s %8d %10d %s\n",
              substr(rare_level_info$variable[i], 1, 35),
              rare_level_info$n_levels[i],
              rare_level_info$n_rare_levels[i],
              substr(rare_level_info$rare_level_names[i], 1, 40)))
}

# ============================================================
# BINARY FACTORS (2 levels - most efficient)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("BINARY FACTORS (2 levels - most efficient)\n")
cat(strrep("-", 70), "\n\n")

binary_factors <- factor_info[factor_info$n_levels == 2, ]
cat("Total binary factors:", nrow(binary_factors), "\n\n")

if (nrow(binary_factors) > 0) {
  cat("First 50 binary factors:\n")
  cat(sprintf("%-45s %10s\n", "Variable", "Pct_Miss"))
  cat(strrep("-", 55), "\n")
  for (i in 1:min(50, nrow(binary_factors))) {
    cat(sprintf("%-45s %9.1f%%\n",
                binary_factors$variable[i],
                binary_factors$pct_missing[i]))
  }
}

# ============================================================
# SAVE OUTPUTS
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("SAVING OUTPUTS\n")
cat(strrep("-", 70), "\n\n")

write.csv(factor_info, "cmps_2016_latino_factor_levels.csv", row.names = FALSE)
cat("Saved: cmps_2016_latino_factor_levels.csv\n")

write.csv(rare_level_info, "cmps_2016_latino_rare_levels.csv", row.names = FALSE)
cat("Saved: cmps_2016_latino_rare_levels.csv\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 70), "\n")
cat("FACTOR ENCODING SUMMARY\n")
cat(strrep("=", 70), "\n")
cat("Total usable factors:        ", length(factor_vars), "\n")
cat("Total usable numeric:        ", length(numeric_vars), "\n")
cat("\n")
cat("Factor level distribution:\n")
for (i in 1:length(bin_counts)) {
  cat(sprintf("  %-15s: %3d factors -> %4d one-hot columns\n",
              names(bin_counts)[i], bin_counts[i], onehot_cols[i]))
}
cat("\n")
cat("TOTAL ONE-HOT COLUMNS:       ", sum(onehot_cols), "\n")
cat("Plus numeric variables:      ", length(numeric_vars), "\n")
cat("FINAL FEATURE COUNT:         ", sum(onehot_cols) + length(numeric_vars), "\n")
cat("\n")
cat("Factors needing attention:\n")
cat("  10+ levels (consolidate?): ", sum(factor_info$n_levels >= 10), "\n")
cat("  Has rare levels (<1%):     ", nrow(rare_level_info), "\n")
cat(strrep("=", 70), "\n")
