# Data Modeling Summary

**Notebook:** `03_modeling.ipynb`


## Part 0: Model-Dependent Preprocessing

These steps pick up where `02_preprocessing.ipynb` left off. Steps 1–8 (data cleaning, composite indices, train/test split, imputation) are replayed in a compact loading cell; Steps 0.1–0.5 below handle encoding, feature reduction, class balancing, and scaling.

### Pipeline Overview

| Step | Description | Output Shape (train) |
|------|-------------|---------------------|
| 0.0 | Compact replay of preprocessing steps 1–8 | df_train (28,887 × 94) |
| 0.1a | Separate X / y | X_train (28,887 × 93) |
| 0.1b | Dummy-encode 19 nominal variables (drop_first) | X_train (28,887 × 135) |
| 0.2 | Drop highly correlated features (\|Spearman r\| > 0.7) | X_train (28,887 × 121) |
| 0.3 | SMOTE oversampling (train only) | X_train (37,372 × 121) |
| 0.4 | StandardScaler (fit on train, transform both) | X_train_scaled (37,372 × 121) |
| 0.5 | ANOVA F-scores for all features (reference) | FEATURE_SCORES table |


### Step 0.0: Data Loading
Compact replay of `02_preprocessing.ipynb` steps 1–8 in a single cell. Produces:
- **df_train**: 28,887 rows × 94 columns (80% stratified split)
- **df_test**: 7,222 rows × 94 columns (20%)
- 0 NaNs in both sets
- 15 straightliners + 1 duplicate removed
- Class distribution (train): 64.7% pro-regulation (1), 35.3% anti-regulation (0)


### Step 0.1: Separate X/y and Nominal Encoding

**Step 0.1a — Separate features and target**
- Target: `QB2C_binary` (1 = pro-regulation, 0 = anti-regulation)
- X_train: 28,887 × 93 features, X_test: 7,222 × 93

**Step 0.1b — Dummy encoding**
- 19 nominal variables encoded with `pd.get_dummies(drop_first=True)`
- Nominal vars cast to `int` before encoding (prevents float column names like `CURREL_NEW_10000.0`)
- Test columns aligned to train via `reindex(columns=..., fill_value=0)` (handles categories present in one set but not the other)
- Result: 135 features after encoding

**19 nominal variables:**
CHNG_A, CHNG_B, CHNG_C, DIVRELPOP, DIVRACPOP, POORASSIST, SCIMPACT, RELIMPACT, EVOL, CURREL_NEW, POLITICAL_ATTITUDE, SECMEMB, LIFEDIR, LIFESPIR, MARITAL, RACECMB, EMPLSIT, USGEN, GENDER


### Step 0.2: Correlation-Based Feature Dropping
- Method: Spearman rank correlation, threshold |r| > 0.7
- For each highly correlated pair, drops the feature **less** correlated with the target (QB2C_binary)
- **14 features dropped** → 121 features remaining
- Same features dropped from both train and test


### Step 0.3: SMOTE (Class Imbalance Handling)
- Applied to **training set only** (test set unchanged)
- `SMOTE(random_state=68169)`
- Before: Class 0 = 10,201, Class 1 = 18,686 (ratio 1.83:1)
- After: Class 0 = 18,686, Class 1 = 18,686 (balanced)
- Total training rows: 37,372


### Step 0.4: Feature Scaling
- `StandardScaler` fit on train, applied to both train and test
- Creates **separate** scaled datasets for linear models:
  - `X_train_scaled` / `X_test_scaled` → for Logistic Regression, SVM
  - `X_train` / `X_test` (unscaled) → for Decision Tree, Random Forest, XGBoost
- Verification: train mean ≈ 0, train std ≈ 1


### Step 0.5: Feature Importance Reference
- `SelectKBest(f_classif, k='all')` — ANOVA F-statistics for all 121 features
- Fit on SMOTE-resampled training data (balanced classes)
- **113 / 121 features significant** (p < 0.05)
- Stored in `FEATURE_SCORES` DataFrame (sorted by F-score descending)
- Top 5 features: IDEO, POORASSIST_2, CHNG_C_2, GOVSIZE1, CHNG_A_2
- Actual feature selection (choosing k) deferred to each model section


### Data Leakage Safeguards
- Imputation: fit on train, transform both (Step 0.0)
- SMOTE: train only, after train/test split (Step 0.3)
- Scaling: fit on train, transform both (Step 0.4)
- Correlation analysis: train only (Step 0.2)
- Feature scores: fit on train only (Step 0.5)


### Datasets Ready for Modeling

| Variable | Shape | Use | Models |
|----------|-------|-----|--------|
| `X_train` | (37,372 × 121) | Unscaled features (SMOTE-balanced) | DT, RF, XGBoost |
| `X_test` | (7,222 × 121) | Unscaled features (original distribution) | DT, RF, XGBoost |
| `X_train_scaled` | (37,372 × 121) | Scaled features (SMOTE-balanced) | LR, SVM |
| `X_test_scaled` | (7,222 × 121) | Scaled features (original distribution) | LR, SVM |
| `y_train` | (37,372,) | Target (balanced via SMOTE) | All |
| `y_test` | (7,222,) | Target (original 64.7/35.3 split) | All |
| `FEATURE_SCORES` | (121 × 3) | Feature selection reference | All |



## Part 1: Model Implementation

For each model, a default model is trained, fitted and these baseline metrics are reported. Next, hyperparameter tuning is performed (either with Grid Search or Random Search and paired with stratified 10-fold cross-validation). A final model is fitted with the best parameters resulting from the tuning step.

For evaluation, the classification report (precision, recall, f1-score, accuracy and support) is evaluated as well as confusion matrices, ROC-AUC curves and feature importance.


### 1.1 Decision Tree Classifier

Step 1: decision tree ```dt_default``` with default parameters

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.62      | 0.60   | 0.61     | 2550    |
| Pro Regulation  | 0.78      | 0.80   | 0.79     | 4672    |
| Accuracy        |           |        | 0.73     | 7222    |
| Macro Avg       | 0.70      | 0.70   | 0.70     | 7222    |
| Weighted Avg    | 0.73      | 0.73   | 0.73     | 7222    |

Step 2: hyperparameter tuning using ```RandomizedSearchCV```

- Best CV Score (F1-Macro): 0.7569
- Best Hyperparameters: {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 9}
- Test Set Score: 0.7735

Step 3: final model ```dt``` fitted with results from hyperparameter tuning

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.67      | 0.70   | 0.69     | 2550    |
| Pro Regulation  | 0.83      | 0.81   | 0.82     | 4672    |
| Accuracy        |           |        | 0.77     | 7222    |
| Macro Avg       | 0.75      | 0.76   | 0.75     | 7222    |
| Weighted Avg    | 0.78      | 0.77   | 0.77     | 7222    |


### 1.2 Logistic Regression

Step 1: logistic classifier ```lr_default``` with default parameters

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.67      | 0.83   | 0.74     | 2550    |
| Pro Regulation  | 0.89      | 0.78   | 0.83     | 4672    |
| Accuracy        |           |        | 0.80     | 7222    |
| Macro Avg       | 0.78      | 0.80   | 0.79     | 7222    |
| Weighted Avg    | 0.81      | 0.80   | 0.80     | 7222    |

Step 2: hyperparameter tuning using ```RandomizedSearchCV```

- Best CV Score (F1-Macro): 0.7737
- Best Hyperparameters: {'C': np.float64(0.050737814374886406), 'l1_ratio': np.float64(0.11347352124058907)}
- Test Set Score: 0.7955

Step 3: final model ```lr``` fitted with results from hyperparameter tuning

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.67      | 0.83   | 0.74     | 2550    |
| Pro Regulation  | 0.89      | 0.78   | 0.83     | 4672    |
| Accuracy        |           |        | 0.80     | 7222    |
| Macro Avg       | 0.78      | 0.80   | 0.79     | 7222    |
| Weighted Avg    | 0.81      | 0.80   | 0.80     | 7222    |


### 1.3 Random Forest

Step 1: random forest ```rf_default``` with default parameters

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.72      | 0.71   | 0.71     | 2550    |
| Pro Regulation  | 0.84      | 0.85   | 0.85     | 4672    |
| Accuracy        |           |        | 0.80     | 7222    |
| Macro Avg       | 0.78      | 0.78   | 0.78     | 7222    |
| Weighted Avg    | 0.80      | 0.80   | 0.80     | 7222    |

Step 2: hyperparameter tuning using ```RandomizedSearchCV```

- Best CV Score (F1-Macro): 0.7779
- Best Hyperparameters: {'max_depth': 26, 'min_samples_leaf': 3, 'n_estimators': 150}
- Test Set Score: 0.8012

Step 3: final model ```rf``` fitted with results from hyperparameter tuning

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.71      | 0.74   | 0.73     | 2550    |
| Pro Regulation  | 0.86      | 0.83   | 0.84     | 4672    |
| Accuracy        |           |        | 0.80     | 7222    |
| Macro Avg       | 0.78      | 0.79   | 0.78     | 7222    |
| Weighted Avg    | 0.80      | 0.80   | 0.80     | 7222    |


### 1.4 XGBoost

Step 1: XGBoost classifier ```xg_default``` with default parameters

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.68      | 0.76   | 0.72     | 2550    |
| Pro Regulation  | 0.86      | 0.80   | 0.83     | 4672    |
| Accuracy        |           |        | 0.79     | 7222    |
| Macro Avg       | 0.77      | 0.78   | 0.78     | 7222    |
| Weighted Avg    | 0.80      | 0.79   | 0.79     | 7222    |

Step 2: hyperparameter tuning using ```RandomizedSearchCV```

- Best CV Score (F1-Macro): 0.7772
- Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}
- Test Set Score: 0.7967

Step 3: final model ```xg``` fitted with results from hyperparameter tuning

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.68      | 0.82   | 0.74     | 2550    |
| Pro Regulation  | 0.89      | 0.79   | 0.83     | 4672    |
| Accuracy        |           |        | 0.80     | 7222    |
| Macro Avg       | 0.78      | 0.80   | 0.79     | 7222    |
| Weighted Avg    | 0.81      | 0.80   | 0.80     | 7222    |


### 1.5 Support Vector Machine (SVM)

Step 1: SVM classifier ```svm_default``` with default parameters

&rarr; Classification report:

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.68      | 0.80   | 0.73     | 2550    |
| Pro Regulation  | 0.88      | 0.79   | 0.83     | 4672    |
| Accuracy        |           |        | 0.79     | 7222    |
| Macro Avg       | 0.78      | 0.80   | 0.78     | 7222    |
| Weighted Avg    | 0.81      | 0.79   | 0.80     | 7222    |

Step 2: hyperparameter tuning using ```RandomizedSearchCV```

- Best CV Score (F1-Macro): 0.7772
- Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}
- Test Set Score: 0.7967

Step 3: final model ```svm``` fitted with results from hyperparameter tuning

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Anti Regulation | 0.66      | 0.84   | 0.74     | 2550    |
| Pro Regulation  | 0.90      | 0.77   | 0.83     | 4672    |
| Accuracy        |           |        | 0.79     | 7222    |
| Macro Avg       | 0.78      | 0.80   | 0.78     | 7222    |
| Weighted Avg    | 0.81      | 0.79   | 0.80     | 7222    |



## Part 2: Evaluation Methods / Comparing Results

Planned evaluation:
- Accuracy, Precision, Recall, F1-Score (per class and macro/weighted)
- Confusion matrices
- ROC-AUC curves
- Feature importance comparison across models
- Cross-validation scores

### Step 2.0 results comparison

Before evaluating complex models, a baseline is established by always predicting the majority class (**Anti Regulation**).

&rarr; Baseline Classification Report:

| Class           | Precision | Recall | F1-Score | Support |
| :-------------- | :-------- | :----- | :------- | :------ |
| Anti Regulation | 0.35      | 1.00   | 0.52     | 2550    |
| Pro Regulation  | 0.00      | 0.00   | 0.00     | 4672    |
| Accuracy        |           |        | 0.35     | 7222    |
| Macro Avg       | 0.18      | 0.50   | 0.26     | 7222    |
| Weighted Avg    | 0.12      | 0.35   | 0.18     | 7222    |

---

### Step 2.1 results comparison

The following table summarizes the performance of all tuned models, sorted by their **Macro F1-Score** on the test set.

| Model                  | Accuracy | ROC-AUC  | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| :--------------------- | :------- | :------- | :--------- | :---------------- | :------------- |
| XGBoost                | 0.7967   | 0.8853   | **0.7864** | 0.7812            | 0.8012         |
| Logistic Regression    | 0.7955   | 0.8792   | **0.7859** | 0.7811            | 0.8025         |
| Random Forest          | 0.8012   | 0.8837   | **0.7848** | 0.7820            | 0.7881         |
| Support Vector Machine | 0.7923   | 0.8795   | **0.7837** | 0.7800            | 0.8031         |
| Decision Tree          | 0.7735   | 0.8368   | **0.7542** | 0.7522            | 0.7566         |
| Baseline (Majority)    | 0.3531   | 0.5000   | **0.2610** | 0.1765            | 0.5000         |

---

### Step 2.2 Confusion Matrix

*See results folder 03_evaluation_confusion_matrices.svg*

---

### Step 2.3 ROC Curves

*See results folder 03_evaluation_roc_curves.svg*

---

### Step 2.4 Feature Importance

The top 10 most influential features for each model are ranked below (derived from visual analysis):

| Rank | Decision Tree        | Logistic Regression  | Random Forest        | XGBoost              |
| :--- | :------------------- | :------------------- | :------------------- | :------------------- |
| 1    | IDEO                 | POLITICAL_ATTITUDE_5 | IDEO                 | IDEO                 |
| 2    | OPENIDEN             | GOVSIZE1             | GOVSIZE1             | GOVSIZE1             |
| 3    | GOVSIZE1             | IDEO                 | OPENIDEN             | CHRNAT               |
| 4    | SCIMPACT_2           | POLITICAL_ATTITUDE_4 | POORASSIST_2         | OPENIDEN             |
| 5    | POLITICAL_ATTITUDE_4 | STEWARDSHIP_BALANCE  | ABRTLGL              | POLITICAL_ATTITUDE_4 |
| 6    | EDUCREC              | SCIENCE_VS_RELIGION  | CHNG_A_2             | SCIMPACT_2           |
| 7    | BIRTHDECADE          | OPENIDEN             | CHRNAT               | EDUCREC              |
| 8    | EXP_G                | SCIMPACT_2           | POLITICAL_ATTITUDE_5 | POLITICAL_ATTITUDE_5 |
| 9    | QB2D                 | CHNG_A_2             | SCIENCE_VS_RELIGION  | CHNG_C_2             |
| 10   | GAYMARR              | CHNG_C_2             | CHNG_C_2             | POORASSIST_2         |
| 11   | PAR2CHILD            | EDUCREC              | GAYMARR              | POLITICAL_ATTITUDE_2 |
| 12   | MEMB                 | POORASSIST_2         | EDUCREC              | ABRTLGL              |
| 13   | ATTNDONRLS           | RELIMPACT_2          | SCIMPACT_2           | SECBEL2              |
| 14   | HLL                  | CHRNAT               | POLITICAL_ATTITUDE_4 | CHNG_A_2             |
| 15   | SOUL                 | EXP_F                | BIRTHDECADE          | SCIENCE_VS_RELIGION  |
---

### Step 2.5 CV score

The stability of model performance was evaluated using 10-fold cross-validation.

| Model                  | Mean F1 (Macro) | Std Dev  | Min Score | Max Score |
| :--------------------- | :-------------- | :------- | :-------- | :-------- |
| Random Forest          | 0.7779          | 0.0052   | 0.7716    | 0.7876    |
| XGBoost                | 0.7772          | 0.0056   | 0.7657    | 0.7866    |
| Support Vector Machine | 0.7741          | 0.0064   | 0.7615    | 0.7834    |
| Logistic Regression    | 0.7737          | 0.0042   | 0.7654    | 0.7806    |
| Decision Tree          | 0.7569          | 0.0090   | 0.7404    | 0.7690    |
| Baseline (Majority)    | 0.2610          | 0.0001   | 0.2609    | 0.2611    |