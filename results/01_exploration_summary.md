# Data Exploration Summary

**Notebook:** `01_exploration.ipynb`  


## 1. Dataset Overview

| Item | Value |
|------|-------|
| Total respondents | 36,908 |
| After dropping QB2C=99 | 36,125 |
| Total raw features | 654 |
| Columns dropped (>50% NaN) | 29: 23 restricted-use file variables (100% NaN in public dataset) + 6 survey-filter variables (KIDACT_A/B/C, PARTYLN, CONGRACE, LEADRACE) missing in >50% of cases because of hierarchical survey structure |
| Climate-related variables leakage check (CLIM1A) | r = -0.092 , so CLIM1A is safe to use as predictor feature |


## 2. Initial Class Variable Distribution (QB2C)

| Class | Value |
|-------|-------|
|2 |     23364 |
|1 |    12761 |
|99 |     783 |


## 3. Initial Religion Findings

| Variable | Finding |
|----------|---------|
| **RELTRAD** | Evangelical Protestant most anti-regulation (~53%); Unaffiliated, Buddhist, Jewish most pro (~75–80%) |
| **ATTNDPERRLS** | Clear gradient: higher attendance → more anti-regulation (53% anti at weekly+ → 25% anti at never) |
| **RELIMP** | Clear gradient: higher importance of religion → more anti-regulation |
| **GOD** | Non-believers ~85% pro-regulation vs. believers ~60% pro - strong binary split |
| **RELIGIOSITY index** | Anti-reg respondents score consistently higher across all 5 components |

## 4. Initial demographic Findings

| Variable | Finding |
|----------|---------|
| **PARTY** | Strongest predictor overall: Democrats ~90% pro-regulation, Republicans ~67% anti |
| **EDUCREC** | Clear gradient: higher education → more pro-regulation (HS or less ~50% anti vs. Master's+ ~25% anti) |
| **INC_SDT1** | Weak, non-linear effect — income alone does not predict climate views well; $150k+ slightly more anti |
| **RACECMB** | Moderate effect: White most anti (~36%), Asian most pro (~25% anti), Black ~30% anti |
| **BIRTHDECADE** | Younger cohorts slightly more pro-regulation |
| **GENDER** | To be confirmed in preprocessing |

