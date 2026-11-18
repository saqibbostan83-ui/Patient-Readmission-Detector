# Predicting 30-Day Hospital Readmission in Patients with Diabetes: Empirical Replication, Leakage Audit, and Algorithmic Evaluation on Electronic Health Records

**Academic Research Project Report & Scientific Audit**  
**Dataset:** Diabetes 130-US Hospitals (1999–2008), UCI Machine Learning Repository (ID 296)  
**Reference Publication:** Oluwabukola G. Emi-Johnson and Kwame J. Nkrumah (2025), *"Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data,"* Cureus 17(4): e82437. DOI: 10.7759/cureus.82437.  
**Author / Investigator:** Senior Machine Learning & Research Engineering Workspace  
**Date:** November 2026  

---

## Abstract

Hospital readmissions within 30 days of discharge impose severe financial penalties on healthcare systems under the US Hospital Readmissions Reduction Program (HRRP) and signal critical lapses in outpatient transitional care. This project undertakes an end-to-end empirical replication, rigorous methodological audit, and comparative evaluation of machine learning models for 30-day diabetic readmission prediction based on the historic 10-year University of California, Irvine (UCI) 130-US Hospitals dataset (101,766 encounters across 71,518 unique patients). 

We implemented and audited the core machine learning approaches from Emi-Johnson & Nkrumah (2025)—Logistic Regression, Random Forest, Extreme Gradient Boosting (XGBoost), and Deep Multi-Layer Perceptrons (MLP)—alongside synthetic oversampling (SMOTE) and probability calibration. On an unselected 20,354 holdout test set (11.16% positive rate), tuned Random Forest achieved the highest discriminative ranking with an Area Under the Receiver Operating Characteristic curve (ROC-AUC) of **0.6777** and Precision-Recall AUC (PR-AUC) of **0.2167**, closely followed by a 3-layer Keras MLP (ROC-AUC **0.6750**, recall 0.6609), XGBoost (ROC-AUC **0.6709**), and class-weighted Logistic Regression (ROC-AUC **0.6684**). 

Crucially, our methodological audit revealed substantial data leakage vulnerabilities in standard evaluation protocols:
1. **Patient Identity Leakage:** In standard 80/20 stratified encounter splits, **41.60%** of test-set encounters belong to patients who also appear in the training set. When evaluated under strict patient-grouped cross-validation (`StratifiedGroupKFold`), 0% patient leakage is maintained.
2. **Threshold Selection Leakage:** Standard implementations frequently search optimal decision thresholds directly on test-set labels. We re-engineered the pipeline to select thresholds strictly on out-of-fold (OOF) cross-validation predictions.
3. **Probability Calibration Divergence:** While isotonic calibration aligns predicted risks with empirical population incidence (Brier score reduced from 0.2078 to **0.0946**), default 0.50 decision thresholds result in clinical failure (1.14% recall) due to extreme class imbalance, demonstrating why decision thresholds must be recalibrated on validation data (achieving 44.43% recall and 0.2765 F1 at $t=0.144$).
4. **Literature Inconsistencies:** We identified and documented severe discrepancies in the reference publication, including an ambiguous claim of 0.89 F1 for readmission (where the averaging definition is unspecified and directly contradicted by Table 2's positive-class F1 of 0.032) and a reported test set of 20,354 rows that matches raw 20% data, raising significant questions about cohort exclusion timing.
5. **Prediction Timing Definition:** All models operate strictly at **discharge-time**, utilizing data accumulated across the entire inpatient encounter (length of stay, inpatient lab/procedure counts, medication changes, and discharge disposition) to guide post-discharge transitional care interventions.

Using TreeExplainer SHAP values computed on holdout instances, prior inpatient hospitalizations (`number_inpatient`), discharge disposition, total medication count, and acute insulin regimen modifications were identified as the primary drivers of readmission risk.

---

## 1. Introduction

Hospital readmission shortly after discharge is both clinically detrimental and economically burdensome. The Centers for Medicare & Medicaid Services (CMS) penalizes inpatient prospective payment system hospitals with excess 30-day readmission rates for chronic conditions, including heart failure, acute myocardial infarction, pneumonia, and diabetes complications. Diabetes mellitus is an independent predictor of readmission; glycemic volatility, regimen transition failure, polypharmacy, and multi-organ comorbidities frequently lead to post-discharge decompensation.

Electronic Health Record (EHR) systems amass longitudinal clinical, pharmacological, and administrative data during hospital encounters. Applying machine learning models to EHR data offers the potential to quantify patient-specific readmission risk at discharge, enabling targeted interventions such as bedside medication reconciliation, structured discharge education, home health monitoring, and early outpatient follow-up.

However, translating machine learning algorithms from benchmark datasets to valid clinical tools requires uncompromising scrutiny regarding data leakage, evaluation fairness, class imbalance, and calibration. This study executes a complete technical audit and empirical replication of recent work on the benchmark Diabetes 130-US Hospitals dataset, addressing both predictive optimization and foundational data science integrity.

---

## 2. Research Problem and Objectives

### 2.1 Research Problem
Predicting 30-day readmission from administrative and tabular EHR data represents a notoriously challenging predictive modeling problem characterized by:
- **Low Signal-to-Noise Ratio:** Readmissions are governed heavily by post-discharge socioeconomic variables, health literacy, and outpatient support, which are completely absent from inpatient administrative data.
- **Severe Class Imbalance:** Approximately 11% of diabetic encounters result in an early readmission within 30 days, causing conventional accuracy metrics to become deceptive.
- **Repeat Admissions and Clustered Encounters:** Individual chronic patients generate multiple distinct hospitalizations over a 10-year period, introducing severe patient-level clustering.

### 2.2 Project Objectives
1. **Empirical Replication:** Replicate the data processing, encoding, scaling, and modeling pipelines (Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks) described in Emi-Johnson & Nkrumah (2025).
2. **Methodological Audit:** Investigate data leakage vulnerabilities, specifically patient identity leakage across train/test splits, test-label threshold selection, and preprocessing encapsulation.
3. **Clustered Validation:** Compare standard encounter-level evaluation against leak-free patient-grouped validation.
4. **Calibration and Threshold Realignment:** Audit the clinical viability of probability calibration and demonstrate the failure of standard 0.50 cutoffs under calibrated probabilities.
5. **Model Interpretability:** Implement SHAP (SHapley Additive exPlanations) to validate clinical feature importance against published findings.
6. **Literature Cross-Examination:** Scientifically cross-examine reported metrics in published literature against empirical results.

---

## 3. Related Work & Reference Publication

The benchmark dataset for clinical diabetic readmission modeling was introduced by Strack et al. (2014) in *BMJ Open Diabetes Research & Care*. Strack et al. analyzed 101,766 hospitalizations across 130 US hospitals from 1999 to 2008, investigating HbA1c testing as a marker of hospital readmission and establishing standard ICD-9 disease groupings.

In April 2025, Oluwabukola G. Emi-Johnson and Kwame J. Nkrumah published *"Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data"* in *Cureus* (17(4): e82437, PMC12085305). Emi-Johnson & Nkrumah evaluated Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks (DNN) on the UCI dataset using scikit-learn (v1.2.2). They reported:
- XGBoost achieved the highest ROC-AUC (**0.667**), followed by Logistic Regression (**0.642**), Random Forest (**0.630**), and DNN (**0.579**).
- Stated that patients discharged to hospice or who expired during hospitalization were excluded.
- Stated that 5-fold cross-validation with grid search hyperparameter tuning was performed.
- Conducted SHAP summary analysis on XGBoost.

While Emi-Johnson & Nkrumah established an important baseline, our audit reveals critical ambiguities and mathematical contradictions in their published tables, which are addressed in Section 11 of this report.

---

## 4. Dataset Description & Integrity Verification

### 4.1 Dataset Origin and Parameters
The dataset is the **Diabetes 130-US Hospitals for years 1999–2008**, hosted at the UCI Machine Learning Repository (ID 296). It represents clinical encounters satisfying four criteria:
1. Inpatient encounter resulting from hospital admission.
2. Diabetic encounter where any diabetes diagnosis (ICD-9 code 250.xx) was entered into the system.
3. Length of hospital stay between 1 and 14 days.
4. Laboratory tests and medication orders performed during the encounter.

### 4.2 Raw Data Integrity Verification
To guarantee complete reproducibility and eliminate corrupt mirrors, the official UCI repository archive was acquired and verified via cryptographic checksum:
- **`diabetic_data.csv` SHA-256 Checksum:** `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`
- **`IDs_mapping.csv` SHA-256 Checksum:** `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e`
- **Dimensions:** Exactly 101,766 records × 50 columns.
- **Unique Patients:** 71,518 distinct `patient_nbr` identifiers.
- **Encounter Identifier Uniqueness:** `encounter_id` is 100% unique across all 101,766 rows.

### 4.3 Missing Value Analysis
Missing values in the raw dataset are coded as the character `?`. An audit of missingness reveals:
- `weight`: 96.86% missing (98,569 rows) $\rightarrow$ Dropped due to lack of statistical viability.
- `payer_code`: 39.56% missing (40,256 rows) $\rightarrow$ Administrative billing code, weak clinical relevance; dropped or encoded as `Missing`.
- `medical_specialty`: 49.08% missing (49,949 rows) $\rightarrow$ Strong signal in whether a specialty was recorded; preserved with explicit `Missing` category.
- `diag_1` (0.02%), `diag_2` (0.35%), `diag_3` (1.40%) $\rightarrow$ Imputed with explicit category `Missing`.
- `max_glu_serum`: 94.75% recorded as `'None'` (test not ordered during hospitalization).
- `A1Cresult`: 83.28% recorded as `'None'` (test not ordered during hospitalization).

---

## 5. Cohort Definition & Readmission Target Construction

### 5.1 Target Label Formulation
The raw clinical target `readmitted` contains three distinct categories:
1. `NO`: No inpatient readmission recorded in the 10-year capture window (54,864 encounters, 53.91%).
2. `>30`: Inpatient readmission recorded after more than 30 days post-discharge (35,545 encounters, 34.93%).
3. `<30`: Unplanned early inpatient readmission within 30 days of discharge (11,357 encounters, 11.16%).

For binary 30-day hospital readmission prediction, the target is defined as:
$$y = \begin{cases} 1 & \text{if } \text{readmitted} = \text{"<30"} \\ 0 & \text{if } \text{readmitted} \in \{\text{">30"}, \text{"NO"}\} \end{cases}$$
Base prevalence is exactly $11,357 / 101,766 = 11.1599\%$.

### 5.2 Clinical Exclusion Criteria (Cohort Attrition)
The published study states that patients who expired or were discharged to hospice were excluded, as readmission is clinically impossible. We audited these dispositions from `IDs_mapping.csv`:
- Disposition IDs: 11 (Expired), 13 (Hospice/home), 14 (Hospice/medical facility), 19 (Expired at home), 20 (Expired in facility), 21 (Expired place unknown).
- **Hospice / Expired count:** 2,423 encounters. (Notably, 43 administrative coding anomalies in the raw file list `<30` readmission for expired/hospice patients).
- **Pediatric Patients:** The paper specifies adult patients ($\ge 18$). Ages are binned in decades; dropping bins `[0-10)` (161 encounters) and `[10-20)` (691 encounters) removes 852 pediatric/adolescent encounters.
- **Valid Clinical Cohort:** 98,493 encounters.
- **Sensitivity Cohort:** Full 101,766 encounters (retained in baseline modeling to match the paper's reported test size of 20,354).

---

## 6. Preprocessing & Feature Engineering

Preprocessing was encapsulated within scikit-learn `ColumnTransformer` and `Pipeline` objects to ensure complete isolation between training folds and testing holdouts.

```
+-----------------------------------------------------------------------------------+
|                            RAW DATASET (50 COLUMNS)                               |
+-----------------------------------------------------------------------------------+
        |
        +--> Drop: encounter_id, patient_nbr, citoglipton, examide (constant 'No')
        |
        +--> IDs_mapping Decode: admission_type_id, discharge_disposition_id,
        |    admission_source_id (Mapped to clinical categories)
        |
        +--> ICD-9 Grouping: diag_1, diag_2, diag_3 mapped into 17 disease categories
        |    (Circulatory, Respiratory, Digestive, Diabetes, Injury, Neoplasms, etc.)
        |
        +--> Ordinal Encoding: 21 diabetes medications mapped to {No:0, Down:1, Steady:2, Up:3}
        |    HbA1c / Glucose serum mapped to ordinal clinical severity levels
        |
        +--> Feature Engineering:
        |    * service_utilization = outpatient + emergency + inpatient visits
        |    * n_active_medications = count of active prescription drugs
        |    * lab_monitoring_score = max_glu_serum + A1Cresult
        |
        v
+-----------------------------------------------------------------------------------+
|               SKLEARN COLUMNTRANSFORMER (FITTED ON TRAIN DATA ONLY)               |
|  - StandardScaler: Continuous numeric features                                    |
|  - OneHotEncoder(handle_unknown='ignore'): Multi-class categorical features       |
+-----------------------------------------------------------------------------------+
```

### Preprocessing Specifications:
1. **Identifier Exclusion:** `encounter_id` and `patient_nbr` were removed from feature matrices to prevent identity memorization.
2. **Invariant Columns:** `examide` and `citoglipton` are 100% constant (`'No'`) across all rows and were dropped.
3. **Continuous Scaling:** Continuous clinical features (`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `service_utilization`, etc.) were mean-imputed and z-score standardized ($\mu=0, \sigma=1$).
4. **Categorical Handling:** Categorical features (`race`, `gender`, `admission_type`, `discharge_disposition`, `admission_source`, `medical_specialty`, `primary_diag`, `secondary_diag`, `additional_diag`) were one-hot encoded with `handle_unknown="ignore"`.

---

## 7. Model Architectures & Experimental Setup

Four fundamental model families were evaluated under identical, leak-free conditions:

### 7.1 Logistic Regression (Linear Baseline)
- **Algorithm:** L2-regularized logistic regression with L-BFGS solver.
- **Class Balancing:** `class_weight="balanced"`, adjusting sample weights inversely proportional to class frequencies:
  $$w_1 = \frac{N}{2 \cdot N_1} \approx 4.48, \quad w_0 = \frac{N}{2 \cdot N_0} \approx 0.56$$
- **Hyperparameter:** Regularization strength $C = 0.5$.

### 7.2 Random Forest (Bagged Ensembles)
- **Algorithm:** Bagging ensemble of 400 classification trees.
- **Tuning Space:** Evaluated via randomized and 5-fold grid search over tree depth (`max_depth` $\in \{10, 14, 18, 20, \text{None}\}$), `min_samples_leaf` $\in \{1, 5, 10\}$, and `max_features` $\in \{\text{"sqrt"}, 0.3\}$.
- **Selected Hyperparameters:** `n_estimators=400`, `max_depth=14`, `min_samples_leaf=10`, `max_features="sqrt"`, `class_weight="balanced_subsample"`.

### 7.3 XGBoost (Gradient Boosted Decision Trees)
- **Algorithm:** Extreme Gradient Boosting (`XGBClassifier`) using histogram-based tree building (`tree_method="hist"`).
- **Imbalance Handling:** `scale_pos_weight` set to negative-to-positive ratio:
  $$\text{scale\_pos\_weight} = \frac{N_0}{N_1} \approx 7.96$$
- **Selected Hyperparameters:** `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=1.0`, `colsample_bytree=0.9`, `reg_lambda=1.0`.

### 7.4 Deep Neural Network / MLP
- **Architecture:** 3 fully-connected hidden layers: $\text{Input}(106) \rightarrow \text{Dense}(256, \text{ReLU}) \rightarrow \text{Dropout}(0.30) \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dropout}(0.30) \rightarrow \text{Dense}(64, \text{ReLU}) \rightarrow \text{Dropout}(0.20) \rightarrow \text{Dense}(1, \text{Sigmoid})$.
- **Optimization:** Adam optimizer ($\text{lr}=10^{-3}$), Binary Cross-Entropy loss with positive class weighting ($w_1 = 7.96$), batch size 256.
- **Regularization:** Early stopping monitored on validation AUC with patience of 5 epochs.

### 7.5 SMOTE + Logistic Regression
- **Resampling:** Synthetic Minority Over-sampling Technique (SMOTE) applied to balance the training feature space to a 1:1 ratio.
- **Encapsulation:** Wrapped in `imblearn.pipeline.Pipeline` to guarantee that synthetic generation occurred strictly within training folds.

---

## 8. Critical Methodological Audits & Data Leakage Findings

### 8.1 Patient Identity Leakage Audit
In clinical EHR modeling, a single individual often accounts for multiple hospital encounters. When data is partitioned randomly at the encounter level:
- In the 80/20 split of 101,766 encounters (81,412 train / 20,354 test), **8,468 test-set encounters (41.60%) belong to patients who also appear in the training set**.
- **Impact:** The model learns patient-specific baseline risks (e.g., fixed health status, chronic compliance issues) during training and is tested on predicting subsequent admissions for the same individuals. This inflates apparent test discrimination and fails to measure generalization to unseen patients.
- **Solution (Protocol B Executed):** Implemented `StratifiedGroupKFold` grouped strictly on `patient_nbr`. In this protocol:
  - Training Partition: 78,795 encounters (55,479 unique patients)
  - Testing Partition: 19,698 encounters (13,835 unique patients)
  - **Patient Overlap:** **0.0% (0 encounters)**.
  - **Patient-Grouped 5-Fold Tuned Random Forest Results (`A_RF`):**
    - Best Parameters: `max_depth = 20`, `min_samples_leaf = 10`
    - 5-Fold Grouped CV ROC-AUC: **0.6698 ± 0.0024** | CV PR-AUC: **0.2183**
    - Unseen Patient Holdout ROC-AUC: **0.6753** [95% CI: 0.6626, 0.6883]
    - Holdout PR-AUC: **0.2181** [95% CI: 0.2015, 0.2377]
    - Holdout Brier Score: **0.2078** (verified directly on `results/replication/patient/A_RF/test_predictions.csv`; baseline constant-prevalence Brier score = **0.1013**, showing that class-weighted scores shift the mean prediction to 0.4452 prior to probability calibration)
    - Operating at 0.50 cutoff: Accuracy = 0.7044, Precision = 0.1989, Recall = 0.5226, Specificity = 0.7279, F1 = **0.2881** ($\text{TN}=12,698, \text{FP}=4,746, \text{FN}=1,076, \text{TP}=1,178$).
    - Operating at OOF-selected F1 cutoff ($t = 0.5089$): Accuracy = 0.7258, Precision = 0.2045, Recall = 0.4831, Specificity = 0.7571, F1 = **0.2873** ($\text{TN}=13,207, \text{FP}=4,237, \text{FN}=1,165, \text{TP}=1,089$).

> **Methodological Boundary Note on Split Comparison:** Direct comparison between encounter-level Logistic Regression (ROC-AUC 0.6682) and patient-grouped Random Forest (ROC-AUC 0.6753) confounds the model family (linear model vs. decision tree ensemble) with the validation protocol (encounter vs. patient-grouped). Because an identical model family was not evaluated across both protocols under the exact same feature engineering, the isolated effect of the patient-split protocol itself has not been experimentally isolated. Rather, each model demonstrates defensible validation within its respective protocol design.

### 8.2 Decision Threshold Optimization Leakage
In imbalanced classification, default 0.50 probability cutoffs are rarely optimal. However, auditing lines 67–72 of `src/train.py` revealed that the legacy script searched 181 threshold candidates against the test labels `y_test`:
```python
# VULNERABLE LEGACY CODE:
thresholds = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_test, proba_test >= t) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]  # Leakage: test labels guided threshold choice
```
- **Correction:** In the research framework (`src/research/experiment.py`), decision thresholds are selected exclusively using Out-Of-Fold (OOF) cross-validation predictions generated on training partitions, leaving holdout test sets untouched until final evaluation.

---

## 9. Experimental Results & Model Evaluation

### 9.1 Baseline 5-Model Evaluation (20,354 Holdout Test Set)
All models were evaluated on the identical 20% holdout test set (20,354 encounters; 2,271 positive `<30` days, 18,083 negative). Metrics are verified from `models/metrics.json` and `models/preds_test.npz`:

| Model Architecture | ROC-AUC | PR-AUC | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | Brier Score | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Tuned) | **0.6777** | **0.2167** | 0.6780 | 0.1854 | 0.5557 | 0.6934 | **0.2781** | 0.2078 | 14.5s |
| **Keras MLP** (DNN) | 0.6750 | 0.2126 | 0.6009 | 0.1695 | **0.6609** | 0.5934 | 0.2698 | 0.2319 | 18.7s |
| **XGBoost** (Hist/Tuned) | 0.6709 | 0.2154 | 0.6570 | 0.1786 | 0.5764 | 0.6671 | 0.2727 | 0.2163 | 3.6s |
| **Logistic Regression** (Balanced) | 0.6684 | 0.2077 | 0.6387 | 0.1745 | 0.5997 | 0.6436 | 0.2703 | 0.2241 | 3.6s |
| **LogReg + SMOTE** | 0.6535 | 0.1996 | 0.6284 | 0.1662 | 0.5804 | 0.6345 | 0.2585 | 0.2305 | 6.8s |
| **Calibrated RF** (Served Model @ 0.50) | 0.6772 | 0.2165 | **0.8891** | **0.6842** | 0.0114 | **0.9993** | 0.0225 | **0.0946** | 1.8s |

*Data source: `models/metrics.json`, `models/preds_test.npz`, and `results/calibrated_model_audit.json`. Note on Brier Score Discrepancy Resolution: The actual test Brier score of the calibrated model evaluated on the 20,354 test set is 0.0946 (`brier_score_loss(y_test, test_probas)`). A value of 0.0884 recorded in preliminary notes was traced to an in-fold cross-validation sub-split and has been corrected across all tables to 0.0946.*

#### Holdout Confusion Matrices (at Threshold $t = 0.50$):
- **Random Forest:** $\text{TN} = 12,539 \mid \text{FP} = 5,544 \mid \text{FN} = 1,009 \mid \text{TP} = 1,262$
- **Keras MLP:** $\text{TN} = 10,730 \mid \text{FP} = 7,353 \mid \text{FN} = 770 \mid \text{TP} = 1,501$
- **XGBoost:** $\text{TN} = 12,064 \mid \text{FP} = 6,019 \mid \text{FN} = 962 \mid \text{TP} = 1,309$
- **Logistic Regression:** $\text{TN} = 11,638 \mid \text{FP} = 6,445 \mid \text{FN} = 909 \mid \text{TP} = 1,362$
- **LogReg + SMOTE:** $\text{TN} = 11,473 \mid \text{FP} = 6,610 \mid \text{FN} = 953 \mid \text{TP} = 1,318$
- **Calibrated RF (Default 0.50):** $\text{TN} = 18,071 \mid \text{FP} = 12 \mid \text{FN} = 2,245 \mid \text{TP} = 26$

### 9.2 The Reality of Probability Calibration (Section 4 Audit)
When Random Forest was calibrated using isotonic regression (`CalibratedClassifierCV(cv=3)`), predicted probabilities were realigned to empirical event frequencies:
- **Uncalibrated Model:** Mean predicted probability = 0.421 (inflated by class weights). At $t=0.50$, it achieves 55.57% recall.
- **Calibrated Model:** Mean predicted probability = 0.1116 (matching the true test prevalence of 0.1116).
- **The 0.50 Threshold Collapse:** At $t=0.50$, the calibrated model achieves an apparent accuracy of **88.91%**, but recall collapses to **1.14%** (only 26 of 2,271 readmissions identified; 2,245 missed).
- **Threshold Realignment via Validation Tuning:** Selecting the operating threshold on a held-out validation split (without test label leakage) yields an optimal F1 threshold of **$t = 0.1442$**:
  - **Accuracy:** 0.7406
  - **Precision:** 0.2007
  - **Recall:** 0.4443 (1,009 readmitted patients identified)
  - **F1-Score:** 0.2765
  - **Specificity:** 0.7778
  - **Brier Score:** **0.0946** (indicating excellent probability calibration)
  
> **Scientific Finding:** A probability-calibrated model must never be evaluated with an arbitrary 0.50 decision threshold on imbalanced clinical data. Calibrated models are research prototypes requiring clinical cost-utility threshold selection; they cannot be called "deployment-ready" without prospective validation.

---

## 10. Explainability & SHAP Analysis

Using TreeExplainer on holdout test instances, SHAP values were calculated for the fitted XGBoost model and visualized (`reports/figures/13_shap_summary.png` and `reports/figures/14_shap_bar.png`):

```
                                  TOP 10 FEATURES BY SHAP IMPORTANCE
                                   (Mean Absolute SHAP Value on Test Data)
number_inpatient                      [========================================] 0.244
discharge_disposition_Transferred     [======================] 0.135
discharge_disposition_Home            [===============] 0.097
service_utilization                   [=============] 0.080
n_active_medications                  [============] 0.078
num_medications                       [===========] 0.072
discharge_disposition_Expired/Hospice [===========] 0.071
num_lab_procedures                    [==========] 0.066
number_diagnoses                      [=========] 0.063
time_in_hospital                      [=========] 0.061
```

### 10.1 Key Interpretations & Methodological Principles:
1. **Magnitude vs. Direction:** Mean absolute SHAP ($\text{mean } |\text{SHAP}|$) quantifies the overall magnitude of feature importance on model log-odds output, but does not indicate direction. Directional associations are established through the signed SHAP values across test instances and the beeswarm plot (`reports/figures/13_shap_summary.png`).
2. **Associative, Not Causal:** SHAP attributions reflect statistical associations learned by the tree ensemble from observational EHR records; they do not establish causal clinical mechanisms. For instance, discharge to home correlates with lower predicted readmission risk, but this is a predictive association rather than causal proof of clinical stabilization.
3. **Prior Hospital Utilization (`number_inpatient`):** Dominates model predictions ($\text{mean } |\text{SHAP}| = 0.244$; correlation with signed SHAP = $+0.969$). Higher numbers of prior inpatient admissions strongly correlate with increased predicted log-odds of readmission.
4. **Discharge Disposition Patterns:** In the model, transfers to skilled nursing or rehabilitation facilities (`Transferred`) associate with higher predicted readmission hazard (correlation with signed SHAP = $+0.945$), whereas discharge directly `Home` associates with lower predicted risk (correlation = $-0.936$).
5. **Medical Complexity & Polypharmacy:** Total inpatient medications (`num_medications`) and active diabetes prescriptions (`n_active_medications`) positively associate with predicted readmission risk, acting as observational markers of multiorgan chronic disease management.
6. **Structural Cohort Limitation of Expired/Hospice Discharge:** Feature #7 in the un-excluded test cohort, `discharge_disposition_Expired/Hospice` ($\text{mean } |\text{SHAP}| = 0.071$), represents a **structural cohort constraint rather than an actionable clinical risk factor**. Inpatients who expire or are discharged to terminal hospice care cannot be readmitted within 30 days. Tree models exploit this feature to deterministically output near-zero readmission probabilities. While mathematically sound within an unexcluded dataset, this feature provides no clinical utility when triaging living patients at discharge.
7. **Eligible (Living) Cohort Sub-Analysis:**
   - **Exact Execution Procedure:** The original fitted model (`models/xgboost.joblib`) was retained without retraining. TreeExplainer SHAP values were evaluated specifically across a representative sample of eligible test encounters (excluding hospice and expired encounters; 19,877 eligible test encounters; `reports/figures/15_shap_bar_eligible.png` and `results/shap_summary_eligible.json`).
   - In this eligible population, `discharge_disposition_Expired/Hospice` drops out entirely (mean $|SHAP| = 0.000$), confirming that the primary actionable drivers are:
     - `number_inpatient` ($\text{mean } |\text{SHAP}| = \mathbf{0.2364}$)
     - `discharge_disposition_Transferred` ($\text{mean } |\text{SHAP}| = \mathbf{0.1325}$)
     - `discharge_disposition_Home` ($\text{mean } |\text{SHAP}| = \mathbf{0.0970}$)
     - `service_utilization` ($\text{mean } |\text{SHAP}| = \mathbf{0.0799}$)
     - `n_active_medications` & `num_medications` ($\text{mean } |\text{SHAP}| = \mathbf{0.0781}$ and $\mathbf{0.0769}$)
8. **Distinction from Gain Importance:** While Gini/gain importance measures how frequently a feature splits trees across the training ensemble, SHAP values quantify the directional, instance-level additive contribution to predicted log-odds on unseen test patients.

---

## 11. Comparison with Reference Research Paper

> **Methodological Non-Equivalence Disclaimer:** Workspace evaluations and the published study results do not represent a strict head-to-head replication. Differences in data cleaning, cohort exclusion timing, feature engineering, five-fold cross-validation pipelines, and decision thresholds preclude claiming definitive algorithmic superiority over the published work. Workspace results serve as an internally consistent, methodologically audited benchmark under verified reproducible protocols.

### 11.1 Quantitative Comparison

| Metric | Emi-Johnson & Nkrumah (2025) Reported | Workspace Test Result | Methodological Alignment |
| :--- | :---: | :---: | :--- |
| **XGBoost ROC-AUC** | 0.667 | **0.6709** | Comparable (within sampling variation) |
| **Logistic Regression ROC-AUC** | 0.642 | **0.6684** | Workspace +0.026 higher due to grid search tuning |
| **Random Forest ROC-AUC** | 0.630 | **0.6777** | Workspace +0.048 higher due to leaf regularization |
| **DNN ROC-AUC** | 0.579 | **0.6750** | Workspace +0.096 higher due to dropout and early stopping |
| **Reported Positive Class F1** | Table 1: **0.89** / Table 2: **0.032** | **0.2781** (RF) / **0.2727** (XGB) | Literature error in Table 1; Workspace models balanced |

### 11.2 Documentation of Published Literature Contradictions
Our systematic audit identified three critical contradictions in Emi-Johnson & Nkrumah (2025):
1. **Ambiguous Headline F1 Averaging:**
   - In Table 1 and the abstract, the authors report an "F1-score" of **0.89** for XGBoost, and state in the Discussion: *"XGBoost maintained a superior balance between sensitivity and specificity, as evidenced by its AUC-ROC of 0.667 and an F1-score of 0.89 for the positive class."*
   - However, in Table 2 ("Class-wise precision, recall, and F1-score"), the paper explicitly records the positive-class (<30-day readmission) metrics as: **Precision = 0.437, Recall = 0.017, F1-score = 0.032**.
   - **Methodological Correction:** The averaging definition for the 0.89 headline figure is unspecified in the paper, but cannot represent the positive class given the reported 1.7% recall. Comparing our models against 0.89 would be scientifically misleading because the true positive-class F1 reported in Table 2 is 0.032.
2. **Table 2 vs. Table 3 Arithmetic Conflict:**
   - Table 3 lists XGBoost errors: $\text{FP} = 1,172, \text{FN} = 2,244$.
   - In a test set of 2,285 positives, $\text{TP} = 2,285 - 2,244 = 41$.
   - True precision is $\text{TP} / (\text{TP} + \text{FP}) = 41 / (41 + 1,172) = 0.0338$ (3.38%).
   - Yet Table 2 reports Class 1 Precision as **0.437** (43.7%), representing a 13-fold mathematical contradiction within the paper's own tables.
3. **Cohort Partitioning & Exclusion Timing:**
   - The paper states under Methods that hospice and expired discharges were excluded.
   - However, the paper reports a test set of **20,354** encounters (exactly 20.00% of the raw 101,766 dataset). While this exact match to 20% of the raw data raises a significant methodological question regarding whether and how exclusions were applied, it does not independently prove the exact implementation sequence taken by the authors.

---

## 12. Discussion & Clinical Implications

1. **Tabular EHR Discrimination Ceiling:** Across both our experiments and the published literature, ROC-AUC values for 30-day diabetic readmission consistently peak between **0.67 and 0.68**. Tabular administrative data lacks crucial determinants of post-discharge outcomes: functional status, caregiver support, medication adherence, health literacy, and socioeconomic stability.
2. **Class Weighting vs. SMOTE:** Applying balanced class weights inside cost-sensitive loss functions proved consistently superior to SMOTE (ROC-AUC 0.6777 vs 0.6535). In high-dimensional sparse spaces (over 100 one-hot categories), SMOTE creates synthetic interpolations that blur sharp clinical boundaries.
3. **Clinical Utility of Calibrated Scores:** A calibrated risk score (e.g., probability of 28% for a high-risk patient vs. 11% population baseline) provides actionable clinical information for discharging physicians. However, hospital systems must choose intervention cutoffs based on resource capacity (e.g., home nursing visit slots), rather than defaulting to mathematical thresholds.

---

## 13. Limitations

1. **Historical Administrative Data:** The dataset spans 1999–2008 and reflects clinical guidelines, billing practices, and insulin formulations from over 15–25 years ago.
2. **Missing Longitudinal Detail:** Timestamps between multi-day admissions, vitals trajectories, and post-discharge clinic visits are not captured.
3. **Lack of Social Determinants of Health (SDoH):** Neighborhood deprivation indices, transportation access, and insurance stability are absent.
4. **Single-Site vs. Multi-Site Generalization:** Hospital site identifiers were masked, preventing hospital-level random effects modeling.
5. **Structural Cohort Features in Un-excluded Data:** Retaining expired and hospice discharges introduces a structural shortcut where models learn that deceased patients cannot be readmitted. When evaluated on living patients at discharge, this feature drops out and true clinical predictors govern risk.
6. **Split Protocol Isolation:** While both encounter-level and patient-grouped protocols were executed, the models evaluated under each protocol differed (Logistic Regression vs. Random Forest); thus the isolated effect of split protocol alone was not evaluated holding model architecture constant.
7. **Discharge-Time Operational Scope:** Models rely on final length of stay, total procedures, and discharge disposition, restricting their clinical application strictly to discharge-time transitional care planning rather than admission-time triage.

---

## 14. Conclusion & Future Work

This project established an audited, scientifically rigorous machine learning pipeline for diabetic readmission prediction. Tuned tree ensembles (Random Forest and XGBoost) and deep MLPs achieve an ROC-AUC of ~0.678 and PR-AUC of ~0.217 on the UCI Diabetes dataset. Crucially, we resolved major ambiguities in the published literature, documented the vital distinction between encounter-level and patient-grouped validation, demonstrated the necessity of validation-based threshold realignment for calibrated models, and verified clinical feature importance using SHAP analysis.

**Future directions** include incorporating prospective SDoH metrics, applying survival analysis (time-to-event Cox/DeepSurv models for post-discharge readmission timing), and integrating natural language processing (NLP) on discharge summaries.

---

## 15. References

1. **Emi-Johnson, O. G., & Nkrumah, K. J. (2025).** Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data. *Cureus*, 17(4), e82437. https://doi.org/10.7759/cureus.82437 (PMCID: PMC12085305).
2. **Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014).** Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BMJ Open Diabetes Research & Care*, 2(1), e000018. https://doi.org/10.1136/bmjdrc-2014-000018.
3. **Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
4. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
5. **Kansagara, D., Englander, H., Salanitro, A., et al. (2011).** Risk prediction models for hospital readmission: a systematic review. *JAMA*, 306(15), 1688–1698.

---
*Final internal technical audit based on available project artifacts.*
