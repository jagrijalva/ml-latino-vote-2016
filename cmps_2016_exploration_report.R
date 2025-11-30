# ============================================================
# CMPS 2016 DATA EXPLORATION FOR RANDOM FOREST PREP
# Complete Report
# ============================================================

# Load the data
load("CMPS_2016_raw.rda")
df <- da38040.0001

cat("\n")
cat(strrep("=", 70), "\n")
cat("        CMPS 2016 DATA EXPLORATION REPORT FOR RF PREP\n")
cat(strrep("=", 70), "\n\n")

# ============================================================
# 1. LATINO FILTER
# ============================================================
cat(strrep("-", 70), "\n")
cat("1. LATINO FILTER\n")
cat(strrep("-", 70), "\n\n")

cat("VARIABLE NAME: ETHNIC_QUOTA\n")
cat("VARIABLE TYPE:", class(df$ETHNIC_QUOTA), "\n\n")

cat("ALL UNIQUE VALUES AND FREQUENCIES:\n")
eth_table <- table(df$ETHNIC_QUOTA, useNA = "always")
for (i in 1:length(eth_table)) {
  cat(sprintf("  %-45s %5d\n", names(eth_table)[i], eth_table[i]))
}

cat("\n>>> ANSWERS <<<\n")
cat("Variable identifying Latino/Hispanic: ETHNIC_QUOTA\n")
cat("Value indicating Latino ethnicity:    (2) Hispanic or Latino\n")
cat("Total respondents:                    ", nrow(df), "\n")
latino_count <- sum(df$ETHNIC_QUOTA == "(2) Hispanic or Latino", na.rm = TRUE)
cat("Latino respondents:                   ", latino_count, "\n")

# ============================================================
# 2. DEPENDENT VARIABLE (Vote Choice)
# ============================================================
cat("\n")
cat(strrep("-", 70), "\n")
cat("2. DEPENDENT VARIABLE (Vote Choice)\n")
cat(strrep("-", 70), "\n\n")

cat("VARIABLE NAME: C14\n")
cat("VARIABLE TYPE:", class(df$C14), "\n\n")

cat("ALL UNIQUE VALUES AND FREQUENCIES:\n")
vote_table <- table(df$C14, useNA = "always")
for (i in 1:length(vote_table)) {
  cat(sprintf("  %-25s %5d\n", names(vote_table)[i], vote_table[i]))
}

cat("\n>>> ANSWERS <<<\n")
cat("Variable capturing 2016 presidential vote: C14\n")
cat("Trump value:   (2) Donald Trump\n")
cat("Clinton value: (1) Hillary Clinton\n")

# Latino vote breakdown
latinos <- df[df$ETHNIC_QUOTA == "(2) Hispanic or Latino", ]
cat("\nLatino Vote (after filtering to Latinos only):\n")
latino_vote <- table(latinos$C14, useNA = "always")
for (i in 1:length(latino_vote)) {
  cat(sprintf("  %-25s %5d\n", names(latino_vote)[i], latino_vote[i]))
}

latino_trump <- sum(latinos$C14 == "(2) Donald Trump", na.rm = TRUE)
latino_clinton <- sum(latinos$C14 == "(1) Hillary Clinton", na.rm = TRUE)
cat("\nLATINOS WHO VOTED TRUMP:   ", latino_trump, "\n")
cat("LATINOS WHO VOTED CLINTON: ", latino_clinton, "\n")

# ============================================================
# 3. SAMPLE WEIGHTS
# ============================================================
cat("\n")
cat(strrep("-", 70), "\n")
cat("3. SAMPLE WEIGHTS\n")
cat(strrep("-", 70), "\n\n")

weight_cols <- grep("weight|wt|wgt", names(df), ignore.case = TRUE, value = TRUE)
cat("Weight-related variables found:\n")
for (w in weight_cols) {
  cat("  -", w, "\n")
}

for (v in weight_cols) {
  cat("\n--- Variable:", v, "---\n")
  cat("Type:", class(df[[v]]), "\n")
  cat("Summary stats:\n")
  stats <- summary(df[[v]])
  print(stats)
}

# ============================================================
# 4. ALL VARIABLE NAMES
# ============================================================
cat("\n")
cat(strrep("-", 70), "\n")
cat("4. ALL VARIABLE NAMES\n")
cat(strrep("-", 70), "\n\n")

var_info <- data.frame(
  variable = names(df),
  type = sapply(df, function(x) paste(class(x), collapse = "/")),
  n_missing = sapply(df, function(x) sum(is.na(x))),
  stringsAsFactors = FALSE
)

cat("TOTAL VARIABLES:", ncol(df), "\n\n")

# Type summary
cat("VARIABLE TYPES SUMMARY:\n")
type_summary <- table(var_info$type)
for (i in 1:length(type_summary)) {
  cat(sprintf("  %-20s %5d\n", names(type_summary)[i], type_summary[i]))
}

# Save full list
write.csv(var_info, "cmps_2016_all_variables.csv", row.names = FALSE)
cat("\nFull variable list saved to: cmps_2016_all_variables.csv\n")

# Print all variables
cat("\nCOMPLETE VARIABLE LIST:\n")
cat(sprintf("%-5s %-45s %-15s %-10s\n", "Num", "Variable", "Type", "Missing"))
cat(strrep("-", 80), "\n")
for (i in 1:nrow(var_info)) {
  cat(sprintf("%-5d %-45s %-15s %-10d\n",
              i,
              substr(var_info$variable[i], 1, 45),
              substr(var_info$type[i], 1, 15),
              var_info$n_missing[i]))
}

# ============================================================
# 5. CLASS BALANCE CHECK
# ============================================================
cat("\n")
cat(strrep("-", 70), "\n")
cat("5. CLASS BALANCE CHECK\n")
cat(strrep("-", 70), "\n\n")

# Filter to Latinos who voted for Trump or Clinton only
latino_voters <- latinos[latinos$C14 %in% c("(1) Hillary Clinton", "(2) Donald Trump"), ]

total_binary_voters <- nrow(latino_voters)
trump_voters <- sum(latino_voters$C14 == "(2) Donald Trump")
clinton_voters <- sum(latino_voters$C14 == "(1) Hillary Clinton")

pct_trump <- round(100 * trump_voters / total_binary_voters, 2)
pct_clinton <- round(100 * clinton_voters / total_binary_voters, 2)

cat("Among LATINO voters who chose TRUMP or CLINTON:\n\n")
cat("  Total Latino binary (Trump/Clinton) voters:", total_binary_voters, "\n")
cat("  Latinos who voted Trump:                   ", trump_voters, "\n")
cat("  Latinos who voted Clinton:                 ", clinton_voters, "\n\n")
cat("  >>> PERCENTAGE who chose TRUMP:  ", pct_trump, "%\n")
cat("  >>> PERCENTAGE who chose CLINTON:", pct_clinton, "%\n")

cat("\n")
cat(strrep("=", 70), "\n")
cat("                    END OF REPORT\n")
cat(strrep("=", 70), "\n")
