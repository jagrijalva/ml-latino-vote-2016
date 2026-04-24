# 2016 Tier-3 Top-20 Codebook (Direction-Aware Labels)

**Source:** CMPS 2016 (ICPSR 38040). Feature space: Tier 3 (Non-Partisan) after
`exclude_tautological` + `exclude_partisan`, post-L368 audit. Importance
metric: weighted mean |SHAP| on held-out test (train-frame unifier).

## Convention

Labels in this codebook name the **conservative (Trump-associated) pole** of
each item. The convention is **conservative = high** across tables, beeswarms,
dependence plots, and any figure that renders SHAP against the feature's
raw value.

Because the 2016 pipeline applies `recode_dir()` to some items but not all,
this document records, for each feature, whether the current pipeline
encoding places the conservative pole at the high end (`conv=high ✓`) or at
the low end (`conv=low ✗`). Items marked `✗` need a visualization-layer
axis flip for the convention to hold; no re-render is required.

The `recode_dir()` formula is `(max_val + 1) − x`, applied in the `preprocess`
chunk of `CMPS_2016_IFA_analysis_fixed_2.qmd` (≈ line 329). The codebook
column `Pipeline recode?` indicates whether an item is in `recode_specs`.

## Feature-by-feature

### 1. C228 — Black Lives Matter support
- **Question:** "From what you have heard about the Black Lives Matter
  movement, do you strongly support, somewhat support, somewhat oppose, or
  strongly oppose the Black Lives Matter movement activism?"
- **Scale (raw):** 1 = Strongly support ... 5 = Strongly oppose
- **Pipeline recode?** Yes (`max=5`). After recode: high = Strongly oppose.
- **Direction-aware label:** **Oppose BLM**
- **conv=high ✓**

### 2. C45 — Obamacare: amend vs repeal
- **Question:** "The health care reform law, sometimes called Obamacare,
  should be amended and improved, not repealed."
- **Scale (raw):** 1 = Strongly agree ... 5 = Strongly disagree
- **Pipeline recode?** Yes (`max=5`). After recode: high = Strongly disagree
  (i.e., repeal, don't improve).
- **Direction-aware label:** **Repeal Obamacare**
- **conv=high ✓**

### 3. C142 — Reaction to immigrant-rights flag displays
- **Question:** "At these rallies, demonstrators often waved the flags of
  Mexico [A] / waved the American flag as well as flags of other countries
  [B]. Do you support this or does this bother you?"
- **Scale (raw):** 1 = Strongly support ... 4 = Bothers me a lot
- **Pipeline recode?** Yes (`max=4`). After recode: high = Bothers me a lot.
- **Direction-aware label:** **Bothered by immigrant-rights flag displays**
- **conv=high ✓**

### 4. C141 — Pathway to citizenship vs deportation
- **Question:** "Do you think the millions of undocumented [Mexican]
  immigrants in the United States should be eligible for a pathway to
  citizenship, or do you think we should deport undocumented immigrants?"
- **Scale (raw):** 1 = Strongly support pathway ... 4 = Somewhat support
  deporting
- **Pipeline recode?** Yes (`max=4`). After recode: high = Strongly support
  deport.
- **Direction-aware label:** **Support deportation over citizenship**
- **conv=high ✓**

### 5. BLA207 — Immigrants take jobs, housing, health care
- **Question:** "Immigrants take jobs, housing, and healthcare away from
  people who were born in the U.S."
- **Scale (raw):** 1 = Strongly agree ... 4 = Strongly disagree
- **Pipeline recode?** Yes (spec `max=5`, but item is 4-point — shifts
  range to [2,5] but rank order is preserved). After recode: high = agree
  immigrants take jobs/housing/healthcare.
- **Direction-aware label:** **Immigrants take jobs/housing/healthcare**
- **conv=high ✓** (note minor scaling artifact; does not affect SHAP ranking)

### 6. C38 — Preferred policy for undocumented immigrants
- **Question:** "Which comes closest to your view about [undocumented /
  illegal] immigrants who are already living and working in the U.S.?"
- **Scale (raw):** 1 = Stay & apply for citizenship, 2 = Stay temporarily,
  3 = Required to leave
- **Pipeline recode?** Yes (spec `max=4`, but item is 3-point — shifts
  range but rank preserved). After recode: high = leave the U.S.
- **Direction-aware label:** **Support deportation of undocumented**
- **conv=high ✓** (note minor scaling artifact)

### 7. L366 — Worry about detention or deportation
- **Question:** "How worried are you that people you know might be detained
  or deported for immigration reasons?"
- **Scale (raw):** 1 = Extremely worried ... 5 = Not at all worried
- **Pipeline recode?** Yes (spec `max=4`, but item is 5-point — shifts range
  to [0,4] but rank preserved). After recode: high = Not at all worried.
- **Direction-aware label:** **Not worried about detention/deportation**
- **conv=high ✓** (note minor scaling artifact)

### 8. C158 — Federal apology for slavery
- **Question:** "Do you think the federal government should or should not
  apologize to African Americans for the slavery that once existed in this
  country?"
- **Scale (raw):** 1 = Should, 2 = Should not
- **Pipeline recode?** No. High = Should not apologize.
- **Direction-aware label:** **Oppose federal apology for slavery**
- **conv=high ✓**

### 9. C41 — Immigration hurts state economy
- **Question:** "Immigration has an overall negative impact on the economy
  here in {STATE}."
- **Scale (raw):** 1 = Strongly agree ... 5 = Strongly disagree
- **Pipeline recode?** Yes (`max=5`). After recode: high = Strongly agree.
- **Direction-aware label:** **Immigration hurts the economy**
- **conv=high ✓**

### 10. L241 — Gay/lesbian/bisexual rights activism
- **Question:** "How strongly do you support or oppose gay, lesbian, and
  bisexual rights activism?"
- **Scale (raw):** 1 = Strongly support ... 5 = Strongly oppose
- **Pipeline recode?** Yes (spec `max=4`, but item is 5-point — shifts
  range but rank preserved). After recode: high = Strongly oppose.
- **Direction-aware label:** **Oppose LGB rights activism**
- **conv=high ✓** (note minor scaling artifact)

### 11. BLA205 — Immigrants who break the law should leave
- **Question:** "Immigrants who break the law should be forced to leave the
  U.S. and return to their countries of origin."
- **Scale (raw):** 1 = Strongly agree ... 4 = Strongly disagree
- **Pipeline recode?** No. High = Strongly disagree.
- **Direction-aware label:** **Support removing lawbreaking immigrants**
- **conv=high ✗** (conservative pole is low; flip axis at viz layer, or
  flip SHAP x-axis sign, to render convention)

### 12. BL174 — Perceived arrest risk for people like me
- **Question:** "People like me are more likely to be arrested"
  (agree/disagree)
- **Scale (raw):** 1 = Strongly agree ... 5 = Strongly disagree
- **Pipeline recode?** No. High = Strongly disagree (deny arrest risk for
  people like me).
- **Direction-aware label:** **Deny personal arrest risk**
- **conv=high ✓** (denial of structural risk aligns with Trump-associated
  pole; if reviewer challenges this coding, revisit)

### 13. LA250 — Discrimination against own national-origin subgroup
- **Question:** "How much discrimination is there in the United States
  today against [own national-origin subgroup]?"
- **Scale (raw):** 1 = A lot, 2 = Some, 3 = A little, 4 = None at all,
  5 = Don't know (5 is NA in pipeline)
- **Pipeline recode?** No. High (excluding DK) = None at all.
- **Direction-aware label:** **Perceive no discrimination against own subgroup**
- **conv=high ✓**

### 14. L268 — Seriousness of discrimination against Latinos
- **Question:** "How much of a problem do you think discrimination against
  (Hispanics/Latinos) is in preventing (Hispanics/Latinos) in general from
  succeeding in America?"
- **Scale (raw):** 1 = Primary problem ... 5 = Not a problem at all
- **Pipeline recode?** No. High = Not a problem at all.
- **Direction-aware label:** **Minimize Latino discrimination**
- **conv=high ✓**

### 15. BL229 — Black Lives Matter effectiveness
- **Question:** "How effective do you think the Black Lives Matter movement
  will be in helping Blacks achieve equality in this country?"
- **Scale (raw):** 1 = Very effective ... 4 = Not at all effective,
  88 = Don't know (NA)
- **Pipeline recode?** No. High = Not at all effective.
- **Direction-aware label:** **Dismiss BLM effectiveness**
- **conv=high ✓**

### 16. L231 — Local police treatment of Latinos
- **Question:** "In your opinion, how do the local police generally treat
  Latinos?"
- **Scale (raw):** 1 = Often treated fairly ... 4 = Often treated unfairly
- **Pipeline recode?** No. High = Often treated unfairly.
- **Direction-aware label:** **Police treat Latinos fairly** (conservative
  pole is low; the label names the Trump-associated reading)
- **conv=high ✗** (perception of police unfairness is at the high end in
  the current pipeline; flip axis at viz layer, or flip SHAP x-axis sign,
  to render convention)

### 17. C247 — Perceived discrimination against Latinos (general)
- **Question:** "How much discrimination is there in the United States
  today against Latinos?"
- **Scale (raw):** 1 = A lot, 2 = Some, 3 = A little, 4 = None at all,
  5 = Don't know (NA)
- **Pipeline recode?** No. High (excluding DK) = None at all.
- **Direction-aware label:** **Perceive little Latino discrimination**
- **conv=high ✓**

### 18. L300 — Support for immigrant-rights activism
- **Question:** "How strongly do you support or oppose Immigrant Rights
  activism?"
- **Scale (raw):** 1 = Strongly support ... 5 = Strongly oppose
- **Pipeline recode?** Yes (spec `max=4`, but item is 5-point — shifts
  range but rank preserved). After recode: high = Strongly oppose.
- **Direction-aware label:** **Oppose immigrant-rights activism**
- **conv=high ✓** (note minor scaling artifact)

### 19. L195_3 — Identity rank: American
- **Question:** "If you had to rank different identities you may have, how
  would you rank [ethnic identity], [national-origin identity], or American?"
- **Scale (raw):** Rank 1 = most important ... 3 = least important
- **Pipeline recode?** No. High = American ranked least important.
- **Direction-aware label:** **American identity primary** (conservative
  pole is low: American ranked #1)
- **conv=high ✗** (flip axis at viz layer, or flip SHAP x-axis sign, to
  render convention)

### 20. L232 — Concern about police force on Latinos
- **Question:** "How concerned or worried are you about local police
  officers using excessive force on Latinos?"
- **Scale (raw):** 1 = Worried a lot ... 4 = Not at all worried
- **Pipeline recode?** No. High = Not at all worried.
- **Direction-aware label:** **Not concerned about police force on Latinos**
- **conv=high ✓**

## Summary: items needing viz-layer flip for conv=high convention

Three of twenty features have the conservative pole at the low end of the
current pipeline encoding:

- BLA205 (Support removing lawbreaking immigrants)
- L231 (Police treat Latinos fairly)
- L195_3 (American identity primary)

For these, flip the x-axis (or multiply SHAP values by −1 for dependence-style
plots) at render time so the conservative pole renders on the right. The
companion CSV (`codebook_2016_T3_top20.csv`) includes a `needs_viz_flip`
column to enable programmatic handling.

One item is ambiguous and flagged for reviewer scrutiny:

- BL174 (Deny personal arrest risk) — currently labeled conv=high; revisit
  if a reviewer pushes back on the coding direction.

## Pipeline scaling artifacts (flag for discussion, not for rerun)

Four items have `recode_specs` `max_val` entries that differ from the
item's true scale length: BLA207, C38, L366, L241, L300. In all cases the
rank order and hence the SHAP ranking is preserved — the reversal still
produces the intended directional polarity — but the numeric range is
shifted. This does not require a re-render; the codebook treats the
intended high pole as the effective high pole for labeling purposes.
