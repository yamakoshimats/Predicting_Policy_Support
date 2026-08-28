# Data Preprocessing Summary

**Notebook:** `02_preprocessing.ipynb`


## Preprocessing Pipeline Overview

**Phase 1: Data Cleaning (Before Train/Test Split)**
1. Missing Values: Drop structural/restricted columns (23 vars)
2. Hierarchical Variables: Merge UNAFFILDETAIL→CURREL_NEW, PARTYLN→POLITICAL_ATTITUDE
3. Duplicates & Straightlining: Check for duplicate respondents
4. Special Codes: Recode 99s, 900000s to NaN
5. Ordinal Reversals: Ensure consistent directionality (higher = more)
6. Binary Recoding: GOD → GOD_binary, QB2C → QB2C_binary
7. Composite Indices: Build 4 indices from 16 variables

**Phase 2: Overall Modeling Preparation (After Train/Test Split)**
8. Train/Test Split: 80/20 stratified by QB2C_binary
9. Imputation: Drop high-missingness variables (CLIM1A, RELCON_A-E, INC_SDT1, FRMREL), then median (ordinal) / mode (nominal) imputation (fit on train, transform both)

**Phase 3: Model-Specific Preparation (see `03_modeling.ipynb`)**
10. Nominal Encoding: Dummy encoding with drop='first' (fit on train)
11. Correlation Check: Remove highly correlated features (|r| > 0.7, drop less predictive)
12. Class Imbalance: SMOTE / oversampling on train only (on encoded data)
13. Feature Scaling: Standardization (fit on train)
14. Feature Selection: SelectKBest with f_classif




## 1. Missing Values:
#### 1.1 Detecting missing values
> 23 variables missing in public data set
> 14 variables with missing values due to hierarchical structure (PARTYLN, LEADRACE, CONGRACE, KIDACT\_A, KIDACT\_B, KIDACT\_C, MARWHENREC, RELMATCH, SPERLIMP, SPERLTALK, SPERLSIM, BORNFINAL, CHBORNFINAL, CHCONGRACE)
> 4 variables with missing values due to different reasons 

| Variable(s) | Reason for missingness |
|-------------|---------------------|
| 23 restricted-use vars (REGION, STATE, FAMILY, DENOMREC2, PROTFAM, CITIZEN, ORIENTMOD, etc.) | 100% missing in public-use file by design |
| BORNFINAL, CHBORNFINAL | Filter vars: only answered by Christians |
| PROTFAM | Filter var: only answered by Protestants |
| RELMATCH, SPBORNFINAL, SPRELIMP, SPRELTALK, SPRELSIM, MARWHENREC | Filter vars: only answered by married/partnered respondents |
| KIDACT_A, KIDACT_B, KIDACT_C | Filter vars: only answered by respondents with children |
| CONGRACE, LEADRACE, CHCONGRACE | Nominal with 7 categories, >50% missing, would inflate feature space |
| PAR2CHILDA, HH3REC, HUMNATR_A, HUMANTR_B| unknown|
#### 1.2 Dropping variables

## 2. Recoding Hierarchical Variables:

### 2.1 CURREL_NEW - Religious Affiliation with Unaffiliated Detail
**Goal:** Merge `UNAFFILDETAIL` into `CURREL` to capture atheist/agnostic distinction

**Original variables:**
- `CURREL`: Religious affiliation (Protestant, Catholic, Jewish, etc., Unaffiliated=100000)
- `UNAFFILDETAIL`: Only for unaffiliated (1=Atheist, 2=Agnostic, 3=Nothing in particular)

**New variable: `CURREL_NEW`**
- Keeps all original CURREL categories
- Splits unaffiliated (100000) into 3 subcategories:
  - 100001: Religiously unaffiliated (not specified)
  - 100002: Atheist (unaffiliated)
  - 100003: Agnostic (unaffiliated)
- 900000: Refused / uninterpretable

**Distribution:**
- Protestant (1000): 15,099
- Catholic (10000): 6,958
- Unaffiliated-unspecified (100001): 6,329
- Agnostic (100003): 2,401
- Atheist (100002): 1,999
- Others: Jewish (850), Mormon (565), Buddhist (348), etc.

**Variables dropped:** `CURREL`, `RELTRAD`, `UNAFFILDETAIL` (info captured in CURREL_NEW)



### 2.2 POLITICAL_ATTITUDE - Party Affiliation with Leaning
**Goal:** Merge `PARTYLN` into `PARTY` to capture independent leanings

**Original variables:**
- `PARTY`: Party affiliation (1=Republican, 2=Democrat, 3=Independent, 4=Other, 99=Refuse)
- `PARTYLN`: Leaning for non-major party (1=Lean Republican, 2=Lean Democrat, 99=No lean/Refuse)

**New variable: `POLITICAL_ATTITUDE` (5-point scale)**
- 1: Republican (10,530)
- 2: Lean Republican (5,171)
- 3: No Lean / Independent (1,932)
- 4: Lean Democrat (6,114)
- 5: Democrat (12,124)
- 99: Don't know / Refused (1,037)

**Logic:**
- Strong identifiers (PARTY=1,2) → keep as-is (1, 5)
- Independents/Others (PARTY=3,4) → check PARTYLN for leaning (2, 4, or 3)
- Refusers (PARTY=99) → check PARTYLN (2, 4, or 99)

**Variables dropped:** `PARTY`, `PARTYLN` (merged into POLITICAL_ATTITUDE)

## 3. Special Code Recoding: 
**Action:** Convert all special codes (99, 900000, 77, etc.) to NaN
- Variable-specific codes: CURREL (900000), CLIM1A (77, 99), POLITICAL_ATTITUDE (99), etc.
- Global pass: All remaining 99s → NaN across 104 variables
- **Result:** 0 columns contain special codes after recoding

## 4. Duplicates and Patterns in Question Answering:
#### 4.1 Duplicates
- No Duplicates
#### 4.2 Patterns in Question Answering (Straightlining, Attentiveness)
- No units with high shares of only refused/don't know answers
- No units with high shares of only maximum answers
- 956 units with high share of only minimum numbers (965 > 0.5)
  - Using RELCON variables to check straightlining
  - 15 found, deleted from dataset


## 5. Ordinal Reversals & Binary Recoding:

### 5.1 Ordinal Variables Reversed
**Goal:** Consistent direction (higher value = more of the concept)

**Variables reversed:**
- `ATTNDPERRLS` → `ATTND_R`: Service attendance (now: 1=never → 6=more than weekly)
- `PRAY` → `PRAY_R`: Prayer frequency (now: 1=never → 7=multiple times daily)
- `RELIMP` → `RELIMP_R`: Importance of religion (now: 1=not at all → 4=very important)
- `BIBIMP` → `BIBIMP_R`: Importance of Bible (now: 1=not at all → 5=word of God)
- `SPIRPER` → `SPIRPER_R`: How spiritual (now: 1=not at all → 4=very spiritual)

**Formula:** `new_value = (max + 1) - old_value`

**Verification:** All ordinals now have consistent direction (higher = more religious/important)

### 5.2 Binary Recoding
- `GOD` (1=Yes, 2=No) → `GOD_binary` (1=Believes, 0=Doesn't believe)
- `QB2C` (1=Anti, 2=Pro) → `QB2C_binary` (0=Anti, 1=Pro-regulation)

**Original variables dropped** after reversal/recoding



## 6. Composite Indices Building:
**Goal:** Create indices to reduce dimensionality, prevent multicollinearity

**Approach:** Build 4 indices from 16 raw variables, then drop components 



### 6.1 RELIGIOSITY Index (r = -0.258 with QB2C_binary)

**Components (5 variables):**
- `ATTND_R`: Service attendance frequency (reversed)
- `PRAY_R`: Prayer frequency (reversed)
- `RELIMP_R`: Importance of religion (reversed)
- `BIBIMP_R`: Importance of Bible (reversed)
- `SPIRPER_R`: How spiritual (reversed)

**Formula:** Mean of 5 components (row-wise average)

**Scale:** 1.0 (low religiosity) → 7.0 (high religiosity)

**Statistics:**
- Mean: 3.223, Std: 1.301
- Coverage: 100% (36,908 / 36,908)

**Findings:**
- Anti-regulation mean: 3.668 (more religious)
- Pro-regulation mean: 2.966 (less religious)
- **Correlation:** r = -0.258 (moderate negative - more religious → anti-regulation)

**Multicollinearity check:**
- Component-index correlations: r = 0.697 to 0.914
- **Action:** Dropped all 5 components after index creation

**Interpretation:** Captures overall religious commitment through behavioral measures (attendance, prayer) and attitudinal measures (importance, spirituality).



### 6.2 GOD_CERTAINTY Index (r = -0.238 with QB2C_binary)

**Components (2 variables):**
- `GOD_binary`: Belief in God (1=believes, 0=doesn't believe)
- `GOD2`: Certainty about belief (1=absolutely certain, 4=not at all certain)

**Formula:** Unified 7-point scale combining belief direction + certainty
- Non-believers: `GOD_CERTAINTY = GOD2` directly (1-4)
- Believers: `GOD_CERTAINTY = 8 - GOD2` (7-4)

**Scale:**
- **1-3:** Non-believers (1=absolutely certain atheist, 3=not too certain non-believer)
- **4:** Uncertain/agnostic (middle point)
- **5-7:** Believers (5=not too certain believer, 7=absolutely certain believer)

**Statistics:**
- Coverage: 98.2% (36,231 / 36,908)

**Distribution:**
- Value 1 (certain atheist): 2,283 (6.3%)
- Value 2: 2,695 (7.4%)
- Value 3: 775 (2.1%)
- Value 4 (uncertain): 1,005 (2.8%)
- Value 5: 2,378 (6.6%)
- Value 6: 7,556 (20.9%)
- Value 7 (certain believer): 19,539 (53.9%)

**Findings:**
- Anti-regulation mean: 6.342 (more certain believers)
- Pro-regulation mean: 5.392 (less certain/more atheist)
- **Correlation:** r = -0.238 (moderate negative - stronger belief → anti-regulation)

**Action:** Dropped `GOD2` (info captured in GOD_CERTAINTY), kept `GOD_binary` as separate binary feature

**Interpretation:** Captures belief strength on a continuous scale from certain atheism to devout belief, richer than binary belief measure.



### 6.3 SCIENCE_VS_RELIGION Index (r = +0.352 with QB2C_binary) 

**Components (2 variables):**
- `GUIDE_A`: How important are RELIGIOUS TEACHINGS when making moral decisions?
- `GUIDE_D`: How important is SCIENTIFIC INFORMATION when making moral decisions?
- Both: 1=extremely important → 5=not at all important

**Formula:** Bipolar difference score
1. Reverse: `GUIDE_RELIGION_R = 5 - GUIDE_A`, `GUIDE_SCIENCE_R = 5 - GUIDE_D` (scale 0–4)
2. Difference: `SCIENCE_VS_RELIGION = GUIDE_SCIENCE_R - GUIDE_RELIGION_R`

**Scale:** -4 (faith-oriented) ← 0 (balanced) → +4 (science-oriented)

**Statistics:**
- Mean: 0.713 (slightly science-oriented overall), Std: 1.733
- Coverage: 99.3% (36,661 / 36,908)

**Findings:**
- Anti-regulation mean: **-0.100** (faith-oriented, rely more on religion)
- Pro-regulation mean: **+1.177** (science-oriented, rely more on science)
- **Correlation:** r = **+0.352** (STRONGEST PREDICTOR! - science orientation → pro-regulation)

**Action:** Dropped all GUIDE variables (A, B, C, D) 

**Interpretation:** Captures **epistemological orientation** - whether respondents rely more on scientific evidence vs. religious teachings when making moral decisions. This is the best predictor so far of climate regulation attitudes

**INSIGHT:** HOW people make decisions (evidence-based vs faith-based) predicts climate attitudes better than how religious they are



### 6.4 STEWARDSHIP_BALANCE Index (r = +0.264 with QB2C_binary, believers only)

**Components (2 variables):**
- `HUMNTR_A`: "God gave humans the RIGHT TO USE the Earth" (1=completely agree → 5=not at all)
- `HUMNTR_B`: "God gave humans a DUTY TO PROTECT the Earth" (1=completely agree → 5=not at all)
- **Note:** Only asked to believers (GOD_binary=1)

**Formula:** Bipolar difference score
1. Reverse: `HUMNTR_A_R = 6 - HUMNTR_A`, `HUMNTR_B_R = 6 - HUMNTR_B`
2. Difference: `STEWARDSHIP_BALANCE = HUMNTR_B_R - HUMNTR_A_R`
3. Non-believers: `STEWARDSHIP_BALANCE = 0` (no theological view)

**Scale:** -4 (dominion theology) ← 0 (balanced/neutral) → +4 (stewardship theology)

**Statistics:**
- Mean: 0.369 (slightly stewardship-oriented), Std: 1.029
- Coverage: 97.0% (35,816 / 36,908)

**Findings (believers only, N=29,406):**
- Anti-regulation mean: **0.086** (balanced, slight dominion)
- Pro-regulation mean: **0.691** (stewardship-oriented)
- **Correlation (believers):** r = +0.264 (moderate positive - stewardship → pro-regulation)
- **Correlation (all):** r = +0.209 (diluted by non-believers at 0)

**Action:** Dropped `HUMNTR_A` and `HUMNTR_B` after index creation

**Interpretation:** Captures **theological view of nature** among believers. Dominion theology (right to use Earth's resources) vs. Stewardship theology (duty to protect Earth) predicts climate attitudes. Non-believers assigned neutral 0 (no theological position).




### 6.5 Variable Transformation Summary Table

| Original Variables | Arrow | New Variables | Reason |
|-------------------|-------|---------------|---------|
| ATTND_R, PRAY_R, RELIMP_R, BIBIMP_R, SPIRPER_R (5) | → | **RELIGIOSITY** | Composite index, prevent multicollinearity |
| GOD, GOD2 (2) | → | **GOD_CERTAINTY**, GOD_binary | Unified 7-point scale + binary flag |
| GUIDE_A, GUIDE_D (2) | → | **SCIENCE_VS_RELIGION** | Epistemological orientation |
| GUIDE_B, GUIDE_C (2) | → | (dropped) | Correlated with GUIDE_D, redundant |
| HUMNTR_A, HUMNTR_B (2) | → | **STEWARDSHIP_BALANCE** | Theological view of nature |
| CURREL, UNAFFILDETAIL (2) | → | **CURREL_NEW** | Merge atheist/agnostic detail |
| PARTY, PARTYLN (2) | → | **POLITICAL_ATTITUDE** | Merge independent leanings |
| RELTRAD (1) | → | (dropped) | Redundant with CURREL_NEW |
| **Total: 18 variables** | → | **7 features (4 indices + 3 recoded)** | **61% reduction** |

## 7. Train-Test-Split: 
- 80/20 split, stratified by QB2C_binary (random_state=68169)
- Training set: 28,888 rows (64.7% pro-regulation, 35.3% anti-regulation)
- Test set: 7,222 rows (same class distribution)

## 8. Imputation: 
#### 8.1 Dropped high-missingness variables from train & test
| Variable(s) | Missing % | Reason for dropping |
|-------------|-----------|-------------------|
| CLIM1A | 16.45% | Ambiguous categories, thematic closeness to class variable |
| RELCON_A–E | 7–13% | Confusing question, redundant with CURREL_NEW |
| INC_SDT1 | 7.40% | MNAR (income sensitivity), cannot reliably impute |
| FRMREL | 0.87% | Redundant with CURREL_NEW (64% exact match) |

#### 8.2 Median/mode imputation (fit on train, transform both)
- Remaining variables: 0.18%–3.91% missing, assumed MAR
- Ordinal variables: median imputation
- Nominal variables: mode imputation
- **Result:** 0 NaNs remaining, df_train (28,888 × 94), df_test (7,222 × 94)

---

**Steps 10–14 (encoding, correlation check, class imbalance, scaling, feature selection) moved to `03_modeling.ipynb` — they are model-dependent and must run after encoding.**
