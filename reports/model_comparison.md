# Scientific Model Comparison & Empirical Benchmark Report

**Dataset:** Diabetes 130-US Hospitals (UCI ID 296)  
**Reference Study:** Emi-Johnson & Nkrumah (2025), *Cureus*, 17(4), e82437  
**Date of Audit:** November 18, 2026  

---

## 1. Primary Model Comparison — Holdout Test Set (20,354 Encounters)

Evaluated on the unselected 20% holdout test partition (20,354 encounters; 2,271 positive `<30` days, 18,083 negative/`>30` days; prevalence = 11.16%).

> **Clinical Prediction Timing:** All models operate at **discharge-time**, utilizing features accumulated during the completed inpatient stay (length of stay, inpatient lab and procedure counts, inpatient medication modifications, and discharge disposition) to support post-discharge planning and transitional care triage.

### Table 1: Standard Operating Evaluation (Decision Threshold $t = 0.50$)

| Model Family | ROC-AUC | PR-AUC | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | Brier Score | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Tuned) 🏆 | **0.6777** | **0.2167** | 0.6780 | 0.1854 | 0.5557 | 0.6934 | **0.2781** | 0.2078 | 14.5s |
| **Keras MLP** (3-Layer DNN) | 0.6750 | 0.2126 | 0.6009 | 0.1695 | **0.6609** | 0.5934 | 0.2698 | 0.2319 | 18.7s |
| **XGBoost** (Hist/Tuned) | 0.6709 | 0.2154 | 0.6570 | 0.1786 | 0.5764 | 0.6671 | 0.2727 | 0.2163 | 3.6s |
| **Logistic Regression** (Balanced) | 0.6684 | 0.2077 | 0.6387 | 0.1745 | 0.5997 | 0.6436 | 0.2703 | 0.2241 | 3.6s |
| **LogReg + SMOTE** | 0.6535 | 0.1996 | 0.6284 | 0.1662 | 0.5804 | 0.6345 | 0.2585 | 0.2305 | 6.8s |
| **Calibrated RF** (Served Model @ 0.50) | 0.6772 | 0.2165 | **0.8891** | **0.6842** | 0.0114 | **0.9993** | 0.0225 | **0.0946** | 1.8s |

*Data source: `models/metrics.json`, `models/preds_test.npz`, and `results/calibrated_model_audit.json`. Note on Brier score: The actual test Brier score on the 20,354 test set is 0.0946 (`brier_score_loss(y_test, test_probas)`). An earlier preliminary figure of 0.0884 arose from an in-fold validation split and has been harmonized.*

---

## 2. Detailed Audit of the Calibrated Random Forest (Section 4 Audit)

Class weighting (`class_weight="balanced_subsample"`) shifts model scores into an artificial probability range centered around ~0.42. To make output probabilities reflect true clinical risk, the Random Forest model was calibrated with isotonic regression (`CalibratedClassifierCV(method="isotonic", cv=3)`).

### Why the Default 0.50 Threshold Fails on Calibrated Models
1. **Probability Realignment:** Isotonic calibration shifts the mean predicted risk from ~0.421 down to **0.1116**, perfectly aligning with the true empirical test prevalence (11.16%).
2. **Threshold Artifact at $t=0.50$:** Because true readmission is an 11% event, less than 0.2% of patients have estimated probabilities exceeding 0.50. At the arbitrary 0.50 cutoff, the calibrated model achieves an apparent accuracy of **88.91%**, but recall collapses to **1.14%** (missing 2,245 out of 2,271 readmissions).
3. **Threshold Realignment via Validation Tuning:** Selecting the operating threshold on a held-out validation split (without test label leakage) yields an optimal F1 cutoff of **$t = 0.1442$**:

| Operating Threshold Mode | Threshold ($t$) | Accuracy | Precision | Recall | F1-Score | Specificity | True Positives | False Negatives | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Cutoff** | 0.5000 | 0.8891 | **0.6842** | 0.0114 | 0.0225 | **0.9993** | 26 | 2,245 | **0.0946** |
| **Validation F1-Optimal** | 0.1442 | 0.7406 | 0.2007 | 0.4443 | **0.2765** | 0.7778 | 1,009 | 1,262 | **0.0946** |
| **Validation Youden-Optimal** | 0.1393 | 0.7182 | 0.1941 | 0.4839 | 0.2771 | 0.7477 | 1,099 | 1,172 | **0.0946** |

> **Audit Conclusion:** Calibrated models are research prototypes requiring clinical cost-utility threshold selection; they cannot be called "deployment-ready" or clinically safe without prospective validation.

---

## 3. Methodological Protocols: Encounter-Level vs. Patient-Level Validation

| Validation Dimension | Protocol A: Stratified Encounter Split | Protocol B: Patient-Grouped Split |
| :--- | :--- | :--- |
| **Unit of Partitioning** | Hospital Encounter (`encounter_id`) | Unique Patient Identifier (`patient_nbr`) |
| **Total Cohort Size** | 98,493 encounters (adult non-hospice) | 98,493 encounters (adult non-hospice) |
| **Training Partition** | 78,794 encounters (58,303 unique patients) | 78,795 encounters (55,479 unique patients) |
| **Holdout Testing Partition**| 19,699 encounters (17,787 unique patients) | 19,698 encounters (13,835 unique patients) |
| **Test Patients in Train Set**| **8,223 encounters (41.74%)** | **0 encounters (0.0% overlap)** |
| **Internal CV Protocol** | 5-Fold Stratified K-Fold | 5-Fold Stratified Group K-Fold |
| **Model Executed** | `A_LR` (Logistic Regression, C=0.01) | `A_RF` (Random Forest, depth=20, leaf=10) |
| **5-Fold CV ROC-AUC** | 0.6625 ± 0.0069 | 0.6698 ± 0.0024 |
| **Holdout Test ROC-AUC** | **0.6682** [95% CI: 0.6558, 0.6804] | **0.6753** [95% CI: 0.6626, 0.6883] |
| **Holdout Test PR-AUC** | **0.2214** [95% CI: 0.2054, 0.2406] | **0.2181** [95% CI: 0.2015, 0.2377] |
| **Test Accuracy (@ 0.50)** | 0.6623 | 0.7044 |
| **Test Precision (@ 0.50)** | 0.1811 | 0.1989 |
| **Test Recall (@ 0.50)** | 0.5541 | 0.5226 |
| **Test Specificity (@ 0.50)** | 0.6763 | 0.7279 |
| **Test F1-Score (@ 0.50)** | 0.2730 | 0.2881 |
| **Test Confusion Matrix (@ 0.50)** | TN: 11,798, FP: 5,647, FN: 1,005, TP: 1,249 | TN: 12,698, FP: 4,746, FN: 1,076, TP: 1,178 |
| **Holdout Test F1 (@ OOF Thr)** | **0.2768** (at $t=0.5237$) | **0.2873** (at $t=0.5089$) |
| **Test Confusion Matrix (@ OOF Thr)** | TN: 12,796, FP: 4,649, FN: 1,145, TP: 1,109 | TN: 13,207, FP: 4,237, FN: 1,165, TP: 1,089 |

> **Methodological Boundary Note on Split Comparison:** Direct comparison between encounter-level Logistic Regression (ROC-AUC 0.6682) and patient-grouped Random Forest (ROC-AUC 0.6753) confounds the model family (linear classifier vs. decision tree ensemble) with the validation split protocol (encounter vs. patient grouping). Because an identical model family was not evaluated across both protocols under the exact same feature engineering, the isolated effect of patient grouping alone has not been experimentally isolated. Rather, each model demonstrates defensible validation within its respective protocol design.

---

## 4. SHAP Feature Importance (XGBoost TreeExplainer)

Computed on 2,500 holdout test instances (`results/shap_summary.json` and `reports/figures/13_shap_summary.png`):

| Rank | Clinical Feature | Mean \|SHAP Value\| | Directional Association with Readmission |
| :---: | :--- | :---: | :--- |
| 1 | `number_inpatient` | **0.2443** | Heavy positive shift: high prior inpatient stays dramatically increase readmission log-odds. |
| 2 | `discharge_disposition_Transferred` | 0.1353 | Strong positive shift: transfers to SNF, ICF, or rehab reflect high baseline frailty. |
| 3 | `discharge_disposition_Home` | 0.0966 | Negative shift: direct discharge to home correlates with clinical stabilization. |
| 4 | `service_utilization` | 0.0798 | Positive shift: high total prior healthcare encounters (ER + outpatient + inpatient) increase risk. |
| 5 | `n_active_medications` | 0.0778 | Positive shift: medication polypharmacy reflects multi-system chronic disease burden. |
| 6 | `num_medications` | 0.0722 | Positive shift: total inpatient medications administered. |
| 7 | `discharge_disposition_Expired/Hospice` | 0.0712 | Absolute negative shift: patients deceased or in hospice cannot be readmitted. |
| 8 | `num_lab_procedures` | 0.0659 | Moderate positive shift: high laboratory testing volume indicates diagnostic complexity. |
| 9 | `number_diagnoses` | 0.0626 | Positive shift: high comorbid diagnosis count. |
| 10 | `time_in_hospital` | 0.0611 | Moderate positive shift: longer inpatient stays associate with higher post-discharge vulnerability. |

### 4.1 Structural Cohort Issue vs. Actionable Clinical Predictors
Feature #7, `discharge_disposition_Expired/Hospice` (mean \|SHAP\| = 0.0712), represents a **structural cohort artifact rather than an actionable clinical predictor**. Inpatients who expire or are discharged to end-of-life hospice care cannot be readmitted within 30 days. Tree models exploit this feature to deterministically output near-zero readmission probabilities. While mathematically sound within an unexcluded dataset, this feature provides no clinical utility when triaging living patients at discharge.

### 4.2 Eligible (Living) Cohort Sub-Analysis
**Execution Procedure:** The original fitted model (`models/xgboost.joblib`) was retained without retraining. Evaluating SHAP exclusively across eligible living patients discharged to non-hospice destinations (19,877 test encounters; `reports/figures/15_shap_bar_eligible.png` and `results/shap_summary_eligible.json`) confirms that `discharge_disposition_Expired/Hospice` drops out entirely ($|\text{SHAP}| = 0.000$). The leading true clinical predictors are:
1. `number_inpatient` (mean \|SHAP\| = **0.2364**): Strongest positive risk driver.
2. `discharge_disposition_Transferred` (mean \|SHAP\| = **0.1325**): Facility transfer associates with higher predicted log-odds.
3. `discharge_disposition_Home` (mean \|SHAP\| = **0.0970**): Protective statistical marker relative to facility transfers.
4. `service_utilization` (mean \|SHAP\| = **0.0799**): Multi-setting historical healthcare utilization.
5. `n_active_medications` & `num_medications` (mean \|SHAP\| = **0.0781** & **0.0769**): Complex polypharmacy burden.

*Note on SHAP Interpretation:* Mean absolute SHAP measures impact magnitude on model output, not direction. Directional associations are derived from signed values and the beeswarm distribution. SHAP captures statistical associations learned from observational data, not causal clinical mechanisms.

---

## 5. Reference Paper Comparison & Critical Literature Audit

> **Methodological Non-Equivalence Disclaimer:** Workspace evaluations and the reference paper results do not represent a strict head-to-head replication. Differences in data cleaning, cohort exclusion timing, feature engineering, five-fold cross-validation pipelines, and decision thresholds preclude asserting definitive algorithmic superiority. Rather, workspace experiments provide a transparent, reproducible benchmark under documented methodological controls.

### Published Findings (Emi-Johnson & Nkrumah, 2025) vs. Empirical Workspace Findings:

| Evaluation Metric | Emi-Johnson & Nkrumah Reported | Workspace Empirical Result | Audit Finding & Verification |
| :--- | :---: | :---: | :--- |
| **XGBoost ROC-AUC** | 0.667 | **0.6709** | Replicated closely (within sampling variance). |
| **Logistic Regression ROC-AUC** | 0.642 | **0.6684** | Workspace +0.026 higher due to grid search tuning. |
| **Random Forest ROC-AUC** | 0.630 | **0.6777** | Workspace +0.048 higher due to regularized leaf minimums. |
| **DNN ROC-AUC** | 0.579 | **0.6750** | Workspace +0.096 higher due to dropout and early stopping. |
| **Reported Positive Class F1** | Table 1: **0.89** / Table 2: **0.032** | **0.2781** (RF) / **0.2727** (XGB) | Literature error in Table 1; Workspace models balanced. |

### Literature Discrepancies Documented:
1. **Ambiguous Headline F1 Averaging:** Table 1 and the abstract of Emi-Johnson & Nkrumah reported an F1 score of 0.89 for XGBoost without defining the averaging methodology (macro, weighted, or negative-class). However, Table 2 of the published paper reported the true positive-class (<30-day readmission) F1 score as **0.032** (precision 43.7%, recall 1.6%). Comparing model performance against the paper's 0.89 headline figure would be invalid because the headline figure does not reflect positive-class performance.
2. **Internal Arithmetic Inconsistency:** Table 3 listed XGBoost errors as FP = 1,172 and FN = 2,244. Given 2,285 positive test cases, TP = 41. Actual precision is $41 / (41 + 1,172) = 0.0338$ (3.38%). However, Table 2 recorded Class 1 Precision as **0.437** (43.7%)—a 13-fold discrepancy within the paper's own tables.
3. **Cohort Partitioning Questions:** Stated exclusions of hospice/expired patients would yield an 80/20 test set of 19,869 encounters, whereas the paper reports 20,354 encounters (exactly 20.00% of the raw 101,766 dataset). While this matching raw-data test size raises a significant methodological question regarding whether and how exclusions were applied, it does not independently prove the exact implementation steps taken by the authors.
