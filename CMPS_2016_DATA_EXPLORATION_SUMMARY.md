# CMPS 2016 Data Exploration Summary for Random Forest Prep

## Dataset Overview
- **File**: `CMPS_2016_raw.rda`
- **Object Name**: `da38040.0001`
- **Dimensions**: 10,144 rows x 1,006 columns

---

## 1. LATINO FILTER

| Item | Value |
|------|-------|
| **Variable Name** | `ETHNIC_QUOTA` |
| **Variable Type** | factor |
| **Latino Value** | `(2) Hispanic or Latino` |
| **Total Respondents** | 10,144 |
| **Latino Respondents** | 3,002 |

### All ETHNIC_QUOTA Values:
| Value | Count |
|-------|-------|
| (1) White, Not-Hispanic | 1,034 |
| (2) Hispanic or Latino | 3,002 |
| (3) Black or African American | 3,102 |
| (4) Asian American | 3,006 |
| (5) Middle Eastern or Arab | 0 |
| (6) American Indian/Native American | 0 |
| (7) Other | 0 |

---

## 2. DEPENDENT VARIABLE (Vote Choice)

| Item | Value |
|------|-------|
| **Variable Name** | `C14` |
| **Variable Type** | factor |
| **Trump Value** | `(2) Donald Trump` |
| **Clinton Value** | `(1) Hillary Clinton` |

### All C14 Values (Full Sample):
| Value | Count |
|-------|-------|
| (1) Hillary Clinton | 6,246 |
| (2) Donald Trump | 1,934 |
| (3) Gary Johnson | 258 |
| (4) Jill Stein | 206 |
| (5) Someone else | 1,500 |

### Latino Vote Breakdown (After Filtering):
| Value | Count |
|-------|-------|
| (1) Hillary Clinton | 1,796 |
| (2) Donald Trump | 594 |
| (3) Gary Johnson | 107 |
| (4) Jill Stein | 72 |
| (5) Someone else | 433 |

**Latinos who voted Trump**: 594
**Latinos who voted Clinton**: 1,796

---

## 3. SAMPLE WEIGHTS

### Weight Variables Found:
| Variable | Type |
|----------|------|
| `WEIGHT` | numeric |
| `NAT_WEIGHT` | numeric |

### WEIGHT Summary Stats:
| Statistic | Value |
|-----------|-------|
| Min | 0.05659 |
| 1st Quartile | 0.39771 |
| Median | 0.68815 |
| Mean | 0.97993 |
| 3rd Quartile | 1.20224 |
| Max | 9.72797 |

### NAT_WEIGHT Summary Stats:
| Statistic | Value |
|-----------|-------|
| Min | 0.01087 |
| 1st Quartile | 0.13853 |
| Median | 0.25274 |
| Mean | 0.92986 |
| 3rd Quartile | 0.50641 |
| Max | 45.95484 |

---

## 4. ALL VARIABLE NAMES

**Total Variables**: 1,006

### Variable Types Summary:
| Type | Count |
|------|-------|
| factor | 459 |
| numeric | 547 |

### Complete Variable List
See `cmps_2016_all_variables.csv` for the full listing with:
- Variable name
- Type (factor/numeric)
- Count of missing values

### Key Variables by Category:
- **Identifiers**: RESPID, INTERVIEW_START, INTERVIEW_END
- **Geography**: ZIPCODE, CITY_NAME, COUNTY_NAME
- **Weights**: WEIGHT, NAT_WEIGHT
- **Demographics**: AGE, S1-S10 (screener variables), ETHNIC_QUOTA
- **Vote Choice**: C14 (main 2016 presidential vote)
- **Census Data**: DP2000_*, DP2010_*, DP2015_*, EC*, SC*, HC* (contextual/geographic data)

---

## 5. CLASS BALANCE CHECK

### Among Latino Voters Who Chose Trump or Clinton:

| Metric | Value |
|--------|-------|
| Total Latino binary (Trump/Clinton) voters | 2,390 |
| Latinos who voted Trump | 594 |
| Latinos who voted Clinton | 1,796 |
| **% who chose TRUMP** | **24.85%** |
| **% who chose CLINTON** | **75.15%** |

### Class Imbalance Notes:
- The dataset has a **3:1 Clinton-to-Trump ratio** among Latino voters
- This imbalance should be considered when training Random Forest models
- Options to address:
  - Use class weights
  - Apply SMOTE or other oversampling techniques
  - Use stratified sampling

---

## Files Generated
1. `cmps_2016_all_variables.csv` - Complete variable list with types and missing value counts
2. `cmps_2016_exploration_report.R` - R script for data exploration
3. `CMPS_2016_DATA_EXPLORATION_SUMMARY.md` - This summary document

---

## Recommended Next Steps for RF Prep
1. Filter to Latino respondents using: `df[df$ETHNIC_QUOTA == "(2) Hispanic or Latino", ]`
2. Create binary outcome: Trump (1) vs Clinton (0) from C14
3. Handle the 3:1 class imbalance appropriately
4. Consider using `WEIGHT` or `NAT_WEIGHT` for weighted analysis
5. Select relevant predictor variables from the 1,006 available
