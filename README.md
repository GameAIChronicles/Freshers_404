# CPRI State-Level Hackathon — Screening Round Submission

## Overview

This project predicts two hidden outputs for CPRI screening test data:

- **Reference_Parameter** — a continuous value (regression task)
- **Validity_Label** — Valid / Invalid (binary classification task)

from 8 raw input features (voltage, current, temperature, duration, and 4 sensor
readings), on a dataset explicitly engineered with noise, missing values, sensor
faults, duplicate measurement pairs, and a genuine embedded operating-regime
change.

**Approach:** two separate models (not a joint multitask network) sharing one
feature-engineering pipeline — a classifier for Validity_Label and a regressor
for Reference_Parameter. A multitask deep-learning approach was considered and
rejected early on: with only 1000 training rows, a shared-representation neural
net would be harder to tune and more overfitting-prone than tree ensembles,
which are the standard, reproducible choice for small tabular problems like
this one.

---

## Dataset

| | |
|---|---|
| Training records | 1000 |
| Test records | 350 |
| Raw input features | 8 |
| Hidden targets | Reference_Parameter (regression), Validity_Label (classification) |
| Class balance | 86.6% Valid / 13.4% Invalid (imbalanced) |

---

## Data Cleaning & Feature Engineering

Cleaning was **not** about deleting anomalies — several of them turned out to
be the actual signal the task is built around. Every decision below was
verified against the data before being applied, not assumed.

1. **Fixed physically impossible values.** One row each in train and test had
   a negative sensor reading (`Sensor_S2`), which isn't physically meaningful.
   Clipped to 0 rather than dropped — the rest of each row's data is valid,
   and dropping it would remove a legitimate Invalid example.

2. **Missingness indicator flags — the single most important discovery in the
   whole pipeline.** Every row with a missing `Sensor_S1`, `S2`, or `S3` value
   is **100% Invalid**; every row with a missing `Sensor_S4` value is **100%
   Valid**. This is almost certainly deliberately engineered. Flags
   (`S1_missing`, `S2_missing`, `S3_missing`, `S4_missing`) were created
   *before* imputing anything, so this signal wasn't destroyed.

3. **Duplicate measurement pairs.** 24 rows (12 pairs) share identical input
   features but different `Reference_Parameter` values — a separate fault
   mechanism from the missing-value one (no overlap between the two). All 24
   are Invalid. Flagged as `is_duplicate_input` rather than deduplicated,
   since removing them would remove exactly the examples that teach a model
   what this fault looks like.

4. **Median imputation for Sensor_S1/S2/S3**, fit on the training set only and
   applied identically to test, to avoid leakage.

5. **Dropped the raw `Sensor_S4` value** (correlation with Reference_Parameter
   ≈ 0.002 — essentially noise, matching the dataset's "not every sensor is
   useful" hint) but **kept `S4_missing`** as a feature, since its missingness
   carries signal even though its value doesn't.

6. **Outliers were deliberately left untouched.** Outliers in S1/S2/S3
   correlate almost entirely with Invalid rows (sensor faults); outliers in
   Reference_Parameter and Ambient_Temperature correlate almost entirely with
   Valid rows — that's the dataset's genuine embedded operating-regime change.
   Removing either would destroy real signal in one direction or the other.

7. **Target encoding:** Validity_Label → 0 (Valid) / 1 (Invalid). Invalid was
   deliberately made the positive class, standard practice for fault/anomaly
   detection. Reference_Parameter kept as-is (checked for skew; distribution
   is bimodal, not simply skewed — see Modeling notes).

**Final feature set (12 features from 8 raw columns):**
`Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min,
Sensor_S1, Sensor_S2, Sensor_S3, S1_missing, S2_missing, S3_missing,
S4_missing, is_duplicate_input`

All transformations (imputation values, flags, duplicate detection) were fit
on train and applied identically to test, with row counts, dtypes, and
missing-value counts verified to match at every step.

---

## Exploratory Findings Worth Noting

- `Reference_Parameter`'s distribution is **bimodal**: a tight main cluster
  around 15–22, and a broader second hump from ~25–45 — direct visual evidence
  of the "genuine operating-regime change embedded in valid data" mentioned in
  the task description.
- A hand-written rule using only the missingness/duplicate flags explains just
  39 of 134 Invalid rows (F1 ≈ 0.45) — the remaining 95 Invalid rows have no
  obviously-wrong values at a glance, implying a subtler numeric
  threshold/combination rule buried in the continuous sensor readings, which
  tree-based models were able to recover almost fully (see Classifier results).

---

## Models Tried

Six models were evaluated across the two tasks, using consistent
cross-validation (`StratifiedKFold` for classification, `KFold` for
regression) so comparisons are fair.

### Classifier (Validity_Label)

| Model | Mean tuned F1 | AUC | Threshold stability |
|---|---|---|---|
| Logistic Regression (baseline) | — | 0.647 | — |
| RandomForest | 0.9484 (single seed) / 0.731 avg (5-seed, default threshold) | 0.9977 | Tight (0.137–0.203) |
| LightGBM | 0.9282 | 0.9813 (±0.019) | Erratic (0.006–0.138) |
| XGBoost | 0.9501 | 0.9940 (±0.004) | Wide (0.080–0.502) |
| **ExtraTrees (final choice)** | **0.763 avg (5-seed, default threshold)** | **0.9997** | Tight (mean 0.195, median 0.197) |

**Selection process:**
- Logistic Regression scored AUC 0.647 — proved the classification boundary is
  genuinely nonlinear, justifying tree-based models from the start.
- LightGBM was ruled out: F1 competitive on average but thresholds swung
  wildly fold-to-fold (0.006–0.138), signaling instability on this small,
  imbalanced dataset.
- XGBoost was a near-tie with RandomForest on F1 (0.9501 vs 0.9484, a gap
  smaller than fold-to-fold noise) but lost on every stability metric (AUC
  variance, threshold range, F1 variance) — RandomForest was preferred for
  reliability.
- ExtraTrees initially appeared to hit a perfect F1/AUC of 1.0000 on one seed.
  This was investigated rather than trusted: a hand-written rule using only
  the obvious flags couldn't explain it, and results were re-tested across 5
  different fold-split seeds. The perfect score did not hold up (F1 ranged
  0.72–0.78 across seeds) — but a real, modest, reproducible improvement over
  RandomForest did (5-seed average F1 ~0.763 vs ~0.731, essentially tied AUC).
  **ExtraTrees was selected as the final classifier**, with the honest
  average reported rather than the misleading single-seed peak.
- Final decision threshold: **0.1967** (median of tuned thresholds across all
  5 seeds × 5 folds), used instead of the default 0.5 — at 0.5, F1 was
  understated by a large margin due to the class imbalance.

### Regressor (Reference_Parameter)

| Model | Mean RMSE | Mean MAE |
|---|---|---|
| Linear Regression (baseline) | 4.233 | 3.455 |
| **RandomForest (final choice)** | **2.2166 (±0.695)** | **0.9588 (±0.159)** |
| LightGBM | 2.2802 (±0.689) | 0.9813 (±0.170) |
| XGBoost | 2.4193 (±0.742) | 0.9835 (±0.161) |
| ExtraTrees | 2.4035 | — |

RandomForest won outright on every metric — not a close call. Linear
Regression's much larger error (~2x RandomForest's) confirms the bimodal
target distribution needs a model that can bend to fit two regimes, which
linear models structurally cannot do. Residual analysis (actual vs. predicted)
showed no drift or fanning across the full range of Reference_Parameter,
confirming RandomForest handles both regimes natively without special
treatment.

**Known limitation, not a modeling flaw:** the 3 largest single-row residuals
in validation all belonged to Invalid, duplicate-pair rows — rows where
identical input features map to two contradictory target values. No model can
get both members of such a pair right; this is an inherent property of the
injected fault, not a sign of poor fit. An experiment training the regressor
on Valid rows only vs. all rows showed no meaningful difference (RMSE 3.5473
vs 3.5797 on a held-out split — within noise), so the final model is trained
on **all rows**, using the full 1000-row training set rather than discarding
13% of it for no proven benefit.

---

## Final Models

| Task | Model | Configuration |
|---|---|---|
| Classification (Validity_Label) | `ExtraTreesClassifier` | `n_estimators=300, class_weight="balanced", random_state=42` |
| Regression (Reference_Parameter) | `RandomForestRegressor` | `n_estimators=300, random_state=42` |
| Decision threshold | 0.1967 | Applied to ExtraTrees' `predict_proba` output instead of the default 0.5 |

Both final models are trained on the **full cleaned training set** (not a
held-out split — the split was used only for evaluation).

---

## Validation Methodology

- 5-fold cross-validation, repeated across 5 random seeds (42, 7, 1, 100,
  2024) for the classifier, to avoid trusting a single lucky fold split.
- Metrics: F1 and ROC-AUC for classification (not accuracy — a model
  predicting "Valid" for everything would already score 86.6% accuracy while
  being useless), RMSE and MAE for regression.
- Decision thresholds tuned via precision-recall curves per fold, then
  averaged (median) across all 25 fold results for the final production
  threshold — not picked from a single run.
- Feature importance checked with **permutation importance**, not the
  default impurity-based importance, since impurity-based importance is
  known to under-rate sparse binary flags (like the missingness features)
  relative to continuous variables — permutation importance confirmed the
  missingness flags do carry real predictive weight, just less than the raw
  sensor outlier patterns which capture overlapping information.

---

## Results Summary

- **Classifier:** 5-seed average F1 ≈ 0.763 (default threshold), ≈ 0.94–1.00
  (tuned threshold, depending on fold split), AUC ≈ 0.9997
- **Regressor:** RMSE ≈ 2.22 (~21% of the target's standard deviation of
  10.69), MAE ≈ 0.96
- **Test predictions:** class balance of 87.1% Valid / 12.9% Invalid — closely
  matching training data's 86.6%/13.4% split, a good sign the model
  generalizes rather than overfitting to train-specific quirks.

---

## Reproducing This Work

1. Run the cleaning steps in order (see `Data Cleaning & Feature Engineering`
   above) on `Training_Data` and `Test_Data`, saving to `train_cleaned.csv`
   and `test_cleaned.csv`.
2. Train `ExtraTreesClassifier` and `RandomForestRegressor` with the
   configurations listed under `Final Models`, using the 12-column feature
   set listed above.
3. Predict on the cleaned test set; apply threshold 0.1967 to the
   classifier's `predict_proba` output (not `.predict()`, which defaults to
   0.5).
4. Assemble the submission as `Test_ID, Predicted_Reference_Parameter,
   Validity_Label`, matching `Sample_Submission`'s exact column names.

### Saving / loading trained models

```python
import joblib

joblib.dump(final_clf_et, "final_classifier_extratrees.pkl")
joblib.dump(final_reg, "final_regressor_randomforest.pkl")

# later, or in a new session
loaded_clf = joblib.load("final_classifier_extratrees.pkl")
loaded_reg = joblib.load("final_regressor_randomforest.pkl")
```

The decision threshold (0.1967) and the exact `feature_cols` order are not
part of the saved model object — store them separately (e.g. a small JSON
config) alongside the `.pkl` files so they aren't lost or forgotten.

---

## Key Takeaways

- The strongest predictive signal in this dataset was not a raw sensor
  reading but the *pattern of which values were missing* — verified
  statistically before being trusted, not assumed.
- Outliers were not noise to be cleaned away in one direction; they encoded
  two different real phenomena (sensor faults vs. a genuine operating-regime
  change) depending on which variable and direction they appeared in.
- A suspiciously perfect model score (ExtraTrees' single-seed F1/AUC of
  1.0000) was investigated rather than accepted, and the honest, multi-seed
  result was reported instead — a small but real improvement, not a
  breakthrough.
- Model selection was decided by cross-validated evidence at every step
  (including two close calls — XGBoost vs. RandomForest, and RandomForest vs.
  ExtraTrees) rather than by algorithm popularity.
