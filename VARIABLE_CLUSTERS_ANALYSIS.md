# CMPS 2016 Variable Clusters Analysis
## Top 30 SHAP Predictors for Latino Trump Vote

After removing tautological variables (C6=Trump fav, C7=Clinton fav, C15=Congressional vote), here are the emergent clusters:

---

## CLUSTER 1: PARTISAN IDENTITY (Most Important)

| Variable | SHAP | Description |
|----------|------|-------------|
| L46.Democrat | 0.024 | Party ID = Democrat |
| C242_HID.Democratic.Party | 0.021 | Party hidden = Democratic |
| C25.Democrat | 0.020 | Registration = Democrat |
| L266.Democrats | 0.019 | "Democrats fight harder for Latinos" |
| L267.Democrats | 0.009 | "Democrats better for community" |
| C31.Very.Conservative | 0.002 | Ideology |

**Index Proposal**: Partisan Identity Index
- Items: L46, C25, L266, L267, C31
- Scale: Strong Democrat → Strong Republican

---

## CLUSTER 2: FAVORABILITY TOWARD DEM FIGURES (C2-C5, C8-C11)

| Variable | SHAP | Description |
|----------|------|-------------|
| C4.Very.unfavorable | 0.029 | Unfavorable view of [figure] |
| C2.Very.unfavorable | 0.016 | Unfavorable view of [figure] |
| C9.Very.unfavorable | 0.014 | Unfavorable view of [figure] |
| C4.Somewhat.favorable | 0.013 | Favorable view |
| C3.Very.unfavorable | 0.010 | Unfavorable view |
| C5.Very.unfavorable | 0.003 | Unfavorable view |

**Note**: C2-C11 are favorability ratings for political figures. Based on distributions:
- C2, C3: Heavy favorable skew → likely Obama, Biden
- C4: More split → could be a contested figure
- C5: 44% "Not familiar" → less known figure
- C8, C9: Favorable-leaning

**Index Proposal**: Democratic Figure Favorability Index
- Average favorability toward C2, C3, C4, C5, C8, C9 (excluding C6=Trump, C7=Clinton already dropped)
- Higher = More favorable toward Democratic figures

---

## CLUSTER 3: IMMIGRATION POLICY

| Variable | SHAP | Description |
|----------|------|-------------|
| C337.Increase | 0.010 | Immigration levels should increase |
| C158.Should.not | 0.008 | [Something] should not be done |
| C38.Leave.immediately | 0.003 | Undocumented should be deported |

**C38 Full options**:
1. Stay + path to citizenship (78%)
2. Stay temporarily (12%)
3. Required to leave immediately (10%)

**Index Proposal**: Immigration Restrictionism Index
- Items: C38, C337, C158
- Higher = More restrictive/enforcement-oriented

---

## CLUSTER 4: RACIAL ATTITUDES / LINKED FATE

| Variable | SHAP | Description |
|----------|------|-------------|
| BLA207.Strongly.disagree | 0.007 | Disagree with [solidarity statement] |
| BL229.Not.effective | 0.006 | [Something] not effective for racial issues |
| BLA206.Strongly.disagree | 0.004 | Disagree with [solidarity statement] |
| BL155.Not.major.problem | 0.004 | "Racism exists but not major problem" |
| C142.Bothers.a.lot | 0.005 | [Something] bothers respondent |

**BL155 Options**:
1. Racism is a major problem (73%)
2. Racism exists but not major (23%)
3. Racism no longer exists (4%)

**Index Proposal**: Racial Consciousness Index
- Items: BL155, BL229, BLA206, BLA207
- Higher = More awareness of racism as ongoing problem

---

## CLUSTER 5: POLICY CONSERVATISM

| Variable | SHAP | Description |
|----------|------|-------------|
| C41.Strongly.disagree | 0.008 | Disagree with [liberal policy] |
| C228.Strongly.oppose | 0.007 | Oppose [policy] |
| C40.Strongly.disagree | 0.005 | Disagree with [liberal policy] |
| C111.Never | 0.004 | Never [political activity] |
| C45.Strongly.disagree | 0.003 | Disagree with [statement] |

**Need to verify**: C40, C41, C45, C228 content

**Index Proposal**: Policy Conservatism Index
- Need to identify specific policy content
- Higher = More conservative policy positions

---

## CLUSTER 6: POLITICAL ENGAGEMENT / ATTENTION

| Variable | SHAP | Description |
|----------|------|-------------|
| C247.A.little | 0.002 | [Low attention/trust] |
| C248.A.little | 0.002 | [Low attention/trust] |
| C111, C114, C115.Never | 0.003-0.004 | Never [political activity] |

**C247/C248**: Likely political attention or trust items
- "A lot / Some / A little / None at all"

**Index Proposal**: Political Engagement Index
- Items: C111, C114, C115, C247, C248
- Higher = More engaged

---

## SUMMARY: PROPOSED INDICES FOR LATINO TRUMP VOTE

1. **Partisan Identity Index** (L46, C25, L266, L267, C31)
   - *Strongest predictor cluster*

2. **Democratic Figure Favorability** (C2, C3, C4, C5, C8, C9)
   - *Favorable = lower Trump probability*

3. **Immigration Restrictionism** (C38, C337, C158)
   - *Restrictive = higher Trump probability*

4. **Racial Consciousness** (BL155, BL229, BLA206, BLA207)
   - *Low consciousness = higher Trump probability*

5. **Policy Conservatism** (C40, C41, C45, C228)
   - *Conservative = higher Trump probability*

6. **Political Engagement** (C111, C114, C115, C247, C248)
   - *Low engagement = different pattern?*

---

## NEXT STEPS

1. Verify C40, C41, C45, C228 question content
2. Verify C2-C11 which figure each rates
3. Create composite indices
4. Test index-level predictive power
5. Examine SHAP direction (+ or - for Trump)
