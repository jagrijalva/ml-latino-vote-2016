"""
PHASE 1: Data Preprocessing for Latino Trump Support ML Analysis
================================================================
Step 1.1: Load Data

This script loads the CMPS 2016 raw data and prepares it for analysis.

DV Definition:
- Trump voters = 1
- Non-Trump voters = 0
- Abstainers are excluded from analysis
"""

import pandas as pd
import numpy as np
import subprocess
import os

# =============================================================================
# Step 1.1: Load Data
# =============================================================================

print("=" * 60)
print("PHASE 1: Data Preprocessing")
print("Step 1.1: Load Data")
print("=" * 60)

# Load the RDA file using R to convert to CSV first (handles encoding better)
print("\nLoading CMPS_2016_raw.rda...")

# First, try using pyreadr with error handling
try:
    import pyreadr
    result = pyreadr.read_r('CMPS_2016_raw.rda')
except Exception as e:
    print(f"pyreadr failed with: {e}")
    print("Trying alternative method using R...")

    # Use R to convert RDA to CSV
    r_script = '''
    load("CMPS_2016_raw.rda")
    obj_names <- ls()
    for(name in obj_names) {
        obj <- get(name)
        if(is.data.frame(obj)) {
            write.csv(obj, "temp_cmps_2016.csv", row.names=FALSE, fileEncoding="UTF-8")
            cat(name)
            break
        }
    }
    '''

    with open('temp_convert.R', 'w') as f:
        f.write(r_script)

    result_proc = subprocess.run(['Rscript', 'temp_convert.R'], capture_output=True, text=True)
    if result_proc.returncode != 0:
        print(f"R conversion failed: {result_proc.stderr}")
        raise Exception("Could not load RDA file")

    df_name = result_proc.stdout.strip()
    df = pd.read_csv('temp_cmps_2016.csv', low_memory=False)

    # Clean up temp files
    os.remove('temp_convert.R')
    os.remove('temp_cmps_2016.csv')

    result = {df_name: df}

# Get the dataframe (pyreadr returns a dict with dataframe names as keys)
print(f"\nDataframes found in RDA file: {list(result.keys())}")

# Get the first (and likely only) dataframe
df_name = list(result.keys())[0]
df = result[df_name]

# Document initial dimensions
print("\n" + "-" * 60)
print("INITIAL DATA DIMENSIONS")
print("-" * 60)
print(f"Dataset name: {df_name}")
print(f"Number of rows (observations): {df.shape[0]:,}")
print(f"Number of columns (variables): {df.shape[1]:,}")

# Display basic info
print("\n" + "-" * 60)
print("DATA TYPES SUMMARY")
print("-" * 60)
print(df.dtypes.value_counts())

# Display first few column names
print("\n" + "-" * 60)
print("FIRST 20 COLUMN NAMES")
print("-" * 60)
for i, col in enumerate(df.columns[:20]):
    print(f"  {i+1}. {col}")

print("\n" + "-" * 60)
print("MEMORY USAGE")
print("-" * 60)
print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("Step 1.1 Complete: Data loaded successfully")
print("=" * 60)

# =============================================================================
# Step 1.1b: Filter to Latino Respondents Only
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.1b: Filter to Latino Respondents")
print("=" * 60)

print("\nETHNIC_QUOTA distribution (full sample):")
print(df['ETHNIC_QUOTA'].value_counts(dropna=False))

# Filter to Latino respondents only
n_before = len(df)
df = df[df['ETHNIC_QUOTA'] == '(2) Hispanic or Latino'].copy()
n_after = len(df)

print(f"\n" + "-" * 60)
print("FILTER STEP 1: Latino respondents only")
print("-" * 60)
print(f"  Before filter: {n_before:,}")
print(f"  After filter:  {n_after:,}")
print(f"  Excluded:      {n_before - n_after:,}")

# =============================================================================
# Step 1.2: Identify Presidential Vote Variable
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.2: Identify Presidential Vote Variable")
print("=" * 60)

# Search for vote-related columns
vote_keywords = ['vote', 'pres', 'trump', 'clinton', 'elect', 'ballot']
print("\nSearching for vote-related columns...")

vote_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in vote_keywords):
        vote_cols.append(col)

print(f"\nFound {len(vote_cols)} potential vote-related columns:")
for col in vote_cols[:30]:  # Show first 30
    print(f"  - {col}")

# Look at common presidential vote variable names
common_vote_vars = ['PRES_VOTE', 'VOTE_PRES', 'Q20', 'Q21', 'PRES16', 'VOTE2016']
print("\n" + "-" * 60)
print("Checking common presidential vote variable names...")
for var in common_vote_vars:
    if var in df.columns:
        print(f"\nFound: {var}")
        print(df[var].value_counts(dropna=False))

# Check for pattern like Q followed by numbers (survey questions)
print("\n" + "-" * 60)
print("Checking variables with 'PRES' in name:")
pres_vars = [c for c in df.columns if 'PRES' in c.upper()]
for var in pres_vars[:10]:
    print(f"\n{var}:")
    print(df[var].value_counts(dropna=False).head(10))

# List all columns to find vote choice
print("\n" + "-" * 60)
print("ALL COLUMN NAMES (searching for vote-related):")
print("-" * 60)
for i, col in enumerate(df.columns):
    # Only print columns that might be vote-related
    col_str = str(col).upper()
    if any(kw in col_str for kw in ['VOTE', 'PRES', 'TRUMP', 'CLINTON', 'CAND', 'BALLOT', 'ELECT']):
        print(f"  {col}")

# Check columns starting with specific patterns
print("\n" + "-" * 60)
print("Looking for columns with patterns like S1_, Q, or containing candidate names:")
for col in df.columns:
    # Check for Trump or Clinton in the actual values
    if df[col].dtype == 'object':
        values_str = ' '.join(df[col].dropna().astype(str).unique()[:20]).upper()
        if 'TRUMP' in values_str or 'CLINTON' in values_str:
            print(f"\nColumn '{col}' contains Trump/Clinton:")
            print(df[col].value_counts(dropna=False).head(15))

# =============================================================================
# Step 1.2b: Create Dependent Variable (DV)
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.2b: Create Dependent Variable")
print("=" * 60)

# The presidential vote variable is C14
print("\nPresidential Vote Variable (C14) - Full Distribution:")
print(df['C14'].value_counts(dropna=False))

# Create DV: Trump voters (1) vs Non-Trump voters (0)
# Exclude abstainers/non-voters (those who didn't vote are excluded)
print("\n" + "-" * 60)
print("Creating DV: Trump voters (1) vs Non-Trump voters (0)")
print("-" * 60)

# Identify voters only (those who chose a candidate)
# Values containing candidate names indicate actual voters
def create_trump_dv(vote_choice):
    if pd.isna(vote_choice):
        return np.nan  # Will be excluded
    vote_str = str(vote_choice).upper()
    if 'DONALD TRUMP' in vote_str:
        return 1  # Trump voter
    elif any(name in vote_str for name in ['HILLARY CLINTON', 'GARY JOHNSON', 'JILL STEIN', 'SOMEONE ELSE']):
        return 0  # Non-Trump voter (but still a voter)
    else:
        return np.nan  # Unknown/abstainer - exclude

df['trump_vote'] = df['C14'].apply(create_trump_dv)

print("\nDV Distribution (before dropping non-voters):")
print(df['trump_vote'].value_counts(dropna=False))

# Filter to voters only (exclude abstainers/missing)
n_before_vote_filter = len(df)
df_voters = df[df['trump_vote'].notna()].copy()
n_after_vote_filter = len(df_voters)

print(f"\n" + "-" * 60)
print("FILTER STEP 2: Valid vote choice (non-missing C14)")
print("-" * 60)
print(f"  Before filter: {n_before_vote_filter:,}")
print(f"  After filter:  {n_after_vote_filter:,}")
print(f"  Excluded:      {n_before_vote_filter - n_after_vote_filter:,}")

print("\nFinal DV Distribution:")
print(df_voters['trump_vote'].value_counts())

print(f"\nTrump voters: {df_voters['trump_vote'].sum():,.0f} ({df_voters['trump_vote'].mean()*100:.1f}%)")
print(f"Non-Trump voters: {len(df_voters) - df_voters['trump_vote'].sum():,.0f} ({(1-df_voters['trump_vote'].mean())*100:.1f}%)")

# =============================================================================
# Step 1.3: Preserve Survey Weights
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.3: Preserve Survey Weights")
print("=" * 60)

# Extract survey weights as separate vector
survey_wt = df_voters['WEIGHT'].copy()

print(f"\nSurvey weight variable: WEIGHT")
print(f"Weight statistics:")
print(f"  - N: {survey_wt.notna().sum():,}")
print(f"  - Missing: {survey_wt.isna().sum():,}")
print(f"  - Mean: {survey_wt.mean():.4f}")
print(f"  - Std: {survey_wt.std():.4f}")
print(f"  - Min: {survey_wt.min():.4f}")
print(f"  - Max: {survey_wt.max():.4f}")

# Save weights separately for modeling
survey_wt.to_csv('survey_weights.csv', index=False, header=['survey_wt'])
print(f"\nSaved: survey_weights.csv ({len(survey_wt):,} weights)")

# =============================================================================
# Step 1.4: Exclude Inappropriate Variables
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.4: Exclude Inappropriate Variables")
print("=" * 60)

cols_before = len(df_voters.columns)

# Define exclusion categories
exclude_vars = {
    'tautological': ['C6', 'C7', 'C15'],  # Trump/Pence favorability, congressional vote
    'identifiers': ['RESPID', 'ZIPCODE', 'CITY_NAME', 'COUNTY_NAME'],
    'metadata': ['INTERVIEW_START', 'INTERVIEW_END', 'DIFF_DATE', 'ETHNIC_QUOTA'],
    'weights': ['WEIGHT', 'NAT_WEIGHT'],  # Used separately, not as predictors
    'dv_source': ['C14'],  # Already encoded as trump_vote
}

# Identify race-specific variables (A*, B*, BW* items not asked of Latinos)
race_specific_prefixes = ['A', 'B', 'BW']
race_specific_vars = []
for col in df_voters.columns:
    for prefix in race_specific_prefixes:
        # Match columns that start with prefix followed by a number or underscore
        if col.startswith(prefix) and len(col) > len(prefix):
            next_char = col[len(prefix)]
            if next_char.isdigit() or next_char == '_':
                # Check if 100% missing (race-specific by design)
                if df_voters[col].isna().mean() > 0.99:
                    race_specific_vars.append(col)
                    break

exclude_vars['race_specific'] = race_specific_vars

# Identify high missingness variables (>50% missing)
high_missing_vars = []
for col in df_voters.columns:
    if col not in [v for vals in exclude_vars.values() for v in vals]:
        missing_pct = df_voters[col].isna().mean()
        if missing_pct > 0.50:
            high_missing_vars.append(col)

exclude_vars['high_missingness'] = high_missing_vars

# Identify open-text variables (>50 unique values for object type)
open_text_vars = []
for col in df_voters.columns:
    if col not in [v for vals in exclude_vars.values() for v in vals]:
        if df_voters[col].dtype == 'object':
            n_unique = df_voters[col].nunique()
            if n_unique > 50:
                open_text_vars.append(col)

exclude_vars['open_text'] = open_text_vars

# Print exclusion summary
print("\nVariables to exclude by category:")
total_excluded = 0
for category, vars_list in exclude_vars.items():
    # Filter to only variables that exist in the dataframe
    existing_vars = [v for v in vars_list if v in df_voters.columns]
    print(f"\n  {category.upper()} ({len(existing_vars)} variables):")
    if len(existing_vars) <= 10:
        for v in existing_vars:
            print(f"    - {v}")
    else:
        for v in existing_vars[:5]:
            print(f"    - {v}")
        print(f"    ... and {len(existing_vars) - 5} more")
    total_excluded += len(existing_vars)

# Create flat list of all variables to exclude
all_exclude = []
for vars_list in exclude_vars.values():
    all_exclude.extend([v for v in vars_list if v in df_voters.columns])
all_exclude = list(set(all_exclude))  # Remove duplicates

# Remove excluded variables
df_clean = df_voters.drop(columns=all_exclude, errors='ignore')

print(f"\n" + "-" * 60)
print("EXCLUSION SUMMARY")
print("-" * 60)
print(f"Variables before exclusion: {cols_before:,}")
print(f"Variables excluded: {len(all_exclude):,}")
print(f"Variables retained: {len(df_clean.columns):,}")

# Verify trump_vote is still present
if 'trump_vote' in df_clean.columns:
    print(f"\nDV 'trump_vote' retained: YES")
else:
    print(f"\nWARNING: DV 'trump_vote' was excluded!")

# Save the exclusion log
exclusion_log = pd.DataFrame([
    {'category': cat, 'variable': var}
    for cat, vars_list in exclude_vars.items()
    for var in vars_list if var in df_voters.columns
])
exclusion_log.to_csv('cmps_2016_excluded_vars.csv', index=False)
print(f"\nSaved: cmps_2016_excluded_vars.csv ({len(exclusion_log):,} exclusions logged)")

# =============================================================================
# Step 1.6: Pool Rare Factor Levels
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.6: Pool Rare Factor Levels (<5% -> Other)")
print("=" * 60)

# Separate DV from predictors for processing
y = df_clean['trump_vote'].copy()
X = df_clean.drop(columns=['trump_vote'])

# Track which variables were affected
pooled_vars = []
pooling_details = []

for col in X.columns:
    if X[col].dtype == 'object':
        # Calculate frequency of each level
        value_counts = X[col].value_counts(normalize=True, dropna=False)
        rare_levels = value_counts[value_counts < 0.05].index.tolist()

        # Remove NaN from rare_levels if present (handled separately)
        rare_levels = [lvl for lvl in rare_levels if pd.notna(lvl)]

        if len(rare_levels) > 0:
            n_rare = len(rare_levels)
            n_affected = X[col].isin(rare_levels).sum()

            # Recode rare levels to "Other"
            X[col] = X[col].apply(lambda x: 'Other' if x in rare_levels else x)

            pooled_vars.append(col)
            pooling_details.append({
                'variable': col,
                'n_rare_levels': n_rare,
                'n_obs_affected': n_affected,
                'pct_affected': n_affected / len(X) * 100
            })

print(f"\nVariables with pooled rare levels: {len(pooled_vars)}")
if len(pooled_vars) > 0:
    pooling_df = pd.DataFrame(pooling_details)
    print(f"\nTop 10 most affected variables:")
    print(pooling_df.nlargest(10, 'n_obs_affected')[['variable', 'n_rare_levels', 'pct_affected']].to_string(index=False))

    # Save pooling details
    pooling_df.to_csv('cmps_2016_pooling_log.csv', index=False)
    print(f"\nSaved: cmps_2016_pooling_log.csv")

# =============================================================================
# Step 1.7: Impute Remaining Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.7: Impute Remaining Missing Values")
print("=" * 60)

# Check missing before imputation
missing_before = X.isna().sum()
cols_with_missing = missing_before[missing_before > 0]
print(f"\nColumns with missing values before imputation: {len(cols_with_missing)}")
print(f"Total missing cells: {missing_before.sum():,}")

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Impute numeric with median
imputation_log = []
for col in numeric_cols:
    n_missing = X[col].isna().sum()
    if n_missing > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        imputation_log.append({
            'variable': col,
            'type': 'numeric',
            'method': 'median',
            'imputed_value': median_val,
            'n_imputed': n_missing
        })

# Impute categorical with mode
for col in categorical_cols:
    n_missing = X[col].isna().sum()
    if n_missing > 0:
        mode_val = X[col].mode()
        if len(mode_val) > 0:
            mode_val = mode_val[0]
        else:
            mode_val = 'Unknown'
        X[col] = X[col].fillna(mode_val)
        imputation_log.append({
            'variable': col,
            'type': 'categorical',
            'method': 'mode',
            'imputed_value': mode_val,
            'n_imputed': n_missing
        })

# Verify zero missing
missing_after = X.isna().sum().sum()
print(f"\n" + "-" * 60)
print("IMPUTATION SUMMARY")
print("-" * 60)
print(f"Variables imputed: {len(imputation_log)}")
print(f"Missing values remaining: {missing_after}")

if len(imputation_log) > 0:
    imputation_df = pd.DataFrame(imputation_log)
    print(f"\nImputation by type:")
    print(imputation_df.groupby('type')['n_imputed'].agg(['count', 'sum']))

    # Save imputation log
    imputation_df.to_csv('cmps_2016_imputation_log.csv', index=False)
    print(f"\nSaved: cmps_2016_imputation_log.csv")

if missing_after > 0:
    print(f"\nWARNING: {missing_after} missing values remain!")
else:
    print(f"\nConfirmed: Zero missing values after imputation")

# =============================================================================
# Step 1.8: One-Hot Encode Categorical Variables
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.8: One-Hot Encode Categorical Variables")
print("=" * 60)

n_features_before = len(X.columns)
n_categorical = len(categorical_cols)

print(f"\nFeatures before encoding: {n_features_before}")
print(f"Categorical columns to encode: {n_categorical}")

# One-hot encode with drop_first=True to avoid multicollinearity
X_encoded = pd.get_dummies(X, drop_first=True)

# Convert boolean columns to int (0/1)
bool_cols = X_encoded.select_dtypes(include=['bool']).columns
X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

n_features_after = len(X_encoded.columns)

print(f"\n" + "-" * 60)
print("ENCODING SUMMARY")
print("-" * 60)
print(f"Features before encoding: {n_features_before}")
print(f"Features after encoding: {n_features_after}")
print(f"New dummy features created: {n_features_after - n_features_before + n_categorical}")

# Verify all columns are numeric
non_numeric = X_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric) > 0:
    print(f"\nWARNING: {len(non_numeric)} non-numeric columns remain!")
else:
    print(f"\nConfirmed: All {n_features_after} features are numeric")

# =============================================================================
# Step 1.9: Save Final Outputs
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.9: Save Final Outputs")
print("=" * 60)

# Convert y to integer
y = y.astype(int)

# Create weights dataframe
weights = pd.DataFrame({'survey_wt': survey_wt.values})

# Save as parquet files
X_encoded.to_parquet('cmps_2016_X.parquet', index=False)
y.to_frame('trump_vote').to_parquet('cmps_2016_y.parquet', index=False)
weights.to_parquet('cmps_2016_weights.parquet', index=False)

print(f"\nSaved output files:")
print(f"  - cmps_2016_X.parquet: {X_encoded.shape[0]:,} rows x {X_encoded.shape[1]:,} features")
print(f"  - cmps_2016_y.parquet: {len(y):,} labels")
print(f"  - cmps_2016_weights.parquet: {len(weights):,} weights")

# Also save feature names for reference
feature_names = pd.DataFrame({'feature': X_encoded.columns})
feature_names.to_csv('cmps_2016_feature_names.csv', index=False)
print(f"  - cmps_2016_feature_names.csv: {len(feature_names):,} feature names")

# =============================================================================
# PHASE 1 COMPLETE
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE: Data Preprocessing")
print("=" * 60)
print(f"\nPipeline Summary:")
print(f"  Step 1.1: Loaded CMPS 2016 data (10,144 obs)")
print(f"  Step 1.1b: Filtered to Latinos ({n_after:,} obs)")
print(f"  Step 1.2: Created DV: trump_vote")
print(f"  Step 1.2b: Filtered to voters ({len(y):,} obs)")
print(f"  Step 1.3: Preserved survey weights")
print(f"  Step 1.4: Excluded {len(all_exclude):,} inappropriate variables")
print(f"  Step 1.6: Pooled rare factor levels ({len(pooled_vars)} vars)")
print(f"  Step 1.7: Imputed {len(imputation_log)} variables")
print(f"  Step 1.8: One-hot encoded → {n_features_after:,} features")
print(f"  Step 1.9: Saved parquet outputs")

print(f"\nFinal Dataset:")
print(f"  - Observations: {len(y):,}")
print(f"  - Features: {n_features_after:,}")
print(f"  - Trump voters: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"  - Non-Trump voters: {len(y) - y.sum():,} ({(1-y.mean())*100:.1f}%)")
