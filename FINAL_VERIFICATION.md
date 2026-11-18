# FINAL VERIFICATION & TECHNICAL AUDIT REPORT
**Project:** Predicting 30-Day Hospital Readmission in Patients With Diabetes  
**Reference Paper:** Emi-Johnson & Nkrumah (2025), *Cureus*, 17(4), e82437  
**Dataset:** UCI Diabetes 130-US Hospitals (1999–2008), Repository ID 296  
**Date of Verification:** November 18, 2026  
**Environment:** Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1, TensorFlow 2.21.0, LightGBM 4.7.0, CatBoost 1.2.10  

---

## 1. Executive Summary

This document provides a rigorous, verified technical record of all machine learning models, experimental evaluations, data leakage audits, research paper comparisons, and reproducibility artifacts present in this workspace.

All numerical figures in this document are extracted directly from verified physical artifacts:
- **Baseline Holdout Suite:** `models/metrics.json` and `models/preds_test.npz`
- **Data Integrity & Leakage Audit:** `results/audit/legacy_audit.json`
- **Replication Suite (Encounter Protocol):** `results/replication/encounter/A_LR/metrics.json`
- **Patient-Grouped Validation Suite (Protocol B Executed):** `results/replication/patient/A_RF/metrics.json`
- **Calibrated Model Validation Audit:** `results/calibrated_model_audit.json`
- **SHAP Analysis Artifacts:** `reports/figures/13_shap_summary.png`, `14_shap_bar.png`, and `results/shap_summary.json`
- **Official UCI Dataset Verification:** `data/external/uci_296/` and `data/raw/`
- **Reference Paper Text:** `docs/paper/paper_fulltext.txt` (PMC12085305)

**Prediction Timing Definition:** All models in this project operate at **discharge-time**, utilizing features established over the course of the completed inpatient encounter (including length of stay, total laboratory and procedure counts, inpatient medication modifications, and discharge disposition) to inform post-discharge transitional care and follow-up resource allocation.

---

## 2. Complete Project Directory Structure

```text
AI_model-ASSIGNMENT/
├── .gitignore                                      # Git ignore specifications
├── FINAL_VERIFICATION.md                           # This comprehensive verification document
├── README.md                                       # Main project overview and user documentation
├── requirements.txt                                # Environment dependencies
├── app/
│   └── app.py                                      # Streamlit interactive clinical decision-support app
├── data/
│   ├── external/                                   # Official verified upstream downloads
│   │   ├── uci_296.zip                             # Original UCI repository archive (ID 296)
│   │   └── uci_296/
│   │       ├── diabetic_data.csv                   # Raw CSV (101,766 rows x 50 columns)
│   │       └── IDS_mapping.csv                     # Encounter/admission ID decode mapping
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── .gitkeep
│       ├── diabetic_data.csv                       # Primary dataset (SHA-256 verified)
│       └── IDs_mapping.csv                         # Categorical ID decode table (SHA-256 verified)
├── docs/
│   └── paper/
│       ├── paper_fulltext.txt                      # Extracted text of Emi-Johnson & Nkrumah (2025)
│       ├── pmc_article.html                        # Full HTML paper from PubMed Central (PMC12085305)
│       └── pmc_bioc.json                           # Structured BioC JSON publication data
├── models/
│   ├── best_model.joblib                           # Calibrated Random Forest (isotonic, 3-fold)
│   ├── default_patient.json                        # Median/mode baseline patient for demo app
│   ├── keras_history.json                          # Epoch-level loss and AUC trajectory for MLP
│   ├── keras_mlp.keras                             # Trained Keras Sequential neural network
│   ├── logreg.joblib                               # Trained balanced Logistic Regression pipeline
│   ├── logreg_smote.joblib                         # Trained SMOTE + Logistic Regression pipeline
│   ├── metrics.json                                # Saved test-set metrics for all 5 baseline models
│   ├── preds_test.npz                              # Test-set holdout probabilities and ground truth
│   ├── preprocessor.joblib                         # Fitted scikit-learn ColumnTransformer
│   ├── random_forest.joblib                        # Trained tuned Random Forest pipeline
│   ├── xgboost.joblib                              # Trained tuned XGBoost pipeline
│   └── research/
│       └── encounter/
│           └── A_LR.joblib                         # 5-fold CV tuned replication Logistic Regression
├── notebooks/
│   ├── build_notebook.py                           # Programmatic notebook generator (nbformat)
│   └── readmission_modeling.ipynb                  # Executed end-to-end Jupyter notebook
├── reports/
│   ├── FINAL_PROJECT_REPORT.md                     # Comprehensive academic research report
│   ├── FINAL_PROJECT_REPORT.pdf                    # Published research report PDF (generated)
│   ├── model_comparison.md                         # Structured model comparison documentation
│   └── figures/
│       ├── 01_target_distribution.png              # Class balance breakdown (<30d vs >30d/NO)
│       ├── 02_missing_values.png                   # Column-wise missing percentage
│       ├── 03_numeric_distributions.png            # Histograms of numeric clinical variables
│       ├── 04_readmission_by_key_features.png      # Readmission rate by age, stay, and visits
│       ├── 05_correlation_heatmap.png              # Pearson correlation of numeric features
│       ├── 06_categorical_target_rates.png         # Readmission rate across categorical groups
│       ├── 07_medication_usage.png                 # Prescription frequencies of diabetes drugs
│       ├── 08_roc_curves.png                       # Multi-model ROC curves
│       ├── 09_pr_curves.png                        # Multi-model Precision-Recall curves
│       ├── 10_confusion_matrices.png               # Row-normalized confusion matrices
│       ├── 11_feature_importance.png               # XGBoost feature importance ranking
│       ├── 12_keras_training_history.png           # Keras training loss and validation AUC
│       ├── 13_shap_summary.png                     # SHAP beeswarm summary plot (XGBoost)
│       └── 14_shap_bar.png                         # SHAP mean absolute value bar plot
├── results/
│   ├── run.log                                     # Execution log of research scripts
│   ├── calibrated_model_audit.json                 # Validation-based threshold & calibration audit
│   ├── shap_summary.json                           # Top SHAP feature impact numbers
│   ├── audit/
│   │   └── legacy_audit.json                       # Comprehensive leakage & artifact audit
│   └── replication/
│       ├── cohort_flow.json                        # Step-by-step patient attrition counts
│       ├── encounter/
│       │   ├── feature_names.json                  # One-hot encoded feature column names (261)
│       │   ├── leaderboard_A.csv                   # CSV leaderboard of replication models
│       │   ├── leaderboard_A.json                  # Structured replication leaderboard
│       │   ├── split_summary.json                  # Encounter split parameters and overlap stats
│       │   ├── test_probas.npz                     # Holdout test set predictions
│       │   └── A_LR/
│       │       ├── cv_results.csv                  # 5-fold CV results across hyperparameter grid
│       │       ├── metrics.json                    # Detailed test, OOF, and threshold metrics
│       │       ├── oof_predictions.csv             # Out-of-fold training predictions (78,794)
│       │       └── test_predictions.csv            # Test set predictions and ground truth (19,699)
│       └── patient/
│           ├── feature_names.json                  # One-hot encoded feature names (260)
│           ├── leaderboard_A.csv                   # Patient-grouped replication leaderboard
│           ├── leaderboard_A.json                  # Structured patient-grouped leaderboard
│           ├── split_summary.json                  # Patient-grouped split summary (0% overlap)
│           ├── test_probas.npz                     # Holdout test set predictions array
│           └── A_RF/
│               ├── cv_results.csv                  # 5-fold CV grid search results for Random Forest
│               ├── metrics.json                    # Full test, OOF, and threshold metrics
│               ├── oof_predictions.csv             # Out-of-fold training predictions (78,795)
│               └── test_predictions.csv            # Test predictions on unseen patients (19,698)
├── scripts/
│   ├── audit_calibrated_model.py                   # Script auditing calibrated model thresholding
│   ├── audit_legacy.py                             # Automated audit script for dataset and leakage
│   ├── common.py                                   # Shared split, cache, and evaluation utilities
│   ├── generate_report_pdf.py                      # PDF generation script from markdown report
│   ├── run_improved.py                             # Ablation, tuning, and ensemble runner
│   ├── run_replication.py                          # 5-fold CV replication runner (Leaderboard A)
│   └── run_shap.py                                 # Actual TreeExplainer SHAP computation script
└── src/
    ├── __init__.py
    ├── download_data.py                            # Dataset fetching script
    ├── eda.py                                      # EDA figure generation library
    ├── evaluate.py                                 # Evaluation and figure plotting script
    ├── preprocess.py                               # Preprocessing, ICD-9 mapping, feature engineering
    ├── train.py                                    # Full 5-model training pipeline with tuning
    └── research/
        ├── __init__.py
        ├── config.py                               # Paths, seeds, and constants
        ├── data.py                                 # Loading, cohort selection, and grouped splits
        ├── experiment.py                           # Leakage-safe 5-fold CV runner with FoldCache
        ├── features.py                             # Replication (FS-R) & Engineered (FS-E) features
        ├── keras_mlp.py                            # Scikit-learn compatible Keras DNN estimator
        ├── metrics.py                              # Comprehensive metric computation and bootstrap CI
        └── models.py                               # Model factories and documented search grids
```

---

## 3. Four Distinct Experimental Suites

To maintain complete scientific clarity, the experiments in this repository are divided into four distinct suites:

### Suite A: Legacy Baseline Suite (`models/metrics.json` — 20,354 Holdout Test Set)
Evaluated on the unselected 20% holdout test set (stratified encounter-level split, identical to original workspace split):
- **Test set size:** 20,354 encounters (2,271 positive `<30` days, 18,083 negative; positive base rate = 11.16%)
- **Source of truth:** `models/metrics.json` and `models/preds_test.npz`

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score | Specificity | Brier Score | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Tuned) | **0.6777** | **0.2167** | 0.6780 | 0.1854 | 0.5557 | **0.2781** | 0.6934 | 0.2078 | 14.5s |
| **Keras MLP** (3-Layer DNN) | 0.6750 | 0.2126 | 0.6009 | 0.1695 | **0.6609** | 0.2698 | 0.5934 | 0.2319 | 18.7s |
| **XGBoost** (Hist/Tuned) | 0.6709 | 0.2154 | 0.6570 | 0.1786 | 0.5764 | 0.2727 | 0.6671 | 0.2163 | 3.6s |
| **Logistic Regression** (Balanced) | 0.6684 | 0.2077 | 0.6387 | 0.1745 | 0.5997 | 0.2703 | 0.6436 | 0.2241 | 3.6s |
| **LogReg + SMOTE** | 0.6535 | 0.1996 | 0.6284 | 0.1662 | 0.5804 | 0.2585 | 0.6345 | 0.2305 | 6.8s |
| **Calibrated RF** (Served Model @ 0.50) | 0.6772 | 0.2165 | **0.8891** | **0.6842** | 0.0114 | 0.0225 | **0.9993** | **0.0946** | 1.8s |

> **Brier Score Verification Note:** In earlier preliminary drafting, a value of 0.0884 appeared from an in-fold cross-validation sub-split. Direct verification of the serialized model (`models/best_model.joblib`) evaluated on the actual 20,354 holdout test set (`data['y_test']`) confirms the exact test Brier score is **0.0946** (`0.094586...`), matching `results/calibrated_model_audit.json`. All reports and tables have been harmonized to 0.0946.

#### Detailed Confusion Matrices (Standard 0.50 Decision Threshold):
- **Random Forest:** True Negatives (TN) = 12,539 | False Positives (FP) = 5,544 | False Negatives (FN) = 1,009 | True Positives (TP) = 1,262
- **Keras MLP:** TN = 10,730 | FP = 7,353 | FN = 770 | TP = 1,501
- **XGBoost:** TN = 12,064 | FP = 6,019 | FN = 962 | TP = 1,309
- **Logistic Regression:** TN = 11,638 | FP = 6,445 | FN = 909 | TP = 1,362
- **LogReg + SMOTE:** TN = 11,473 | FP = 6,610 | FN = 953 | TP = 1,318
- **Calibrated RF (Default 0.50):** TN = 18,071 | FP = 12 | FN = 2,245 | TP = 26

---

### Suite B: Research Replication Suite (Leaderboard A: Encounter Protocol)
Executed via `scripts/run_replication.py` with 5-fold cross-validation on 78,794 training encounters (adult non-hospice cohort, 261 features).
- **Source of truth:** `results/replication/encounter/A_LR/metrics.json`
- **Model Executed:** `A_LR` (Logistic Regression, $C=0.01$)
- **5-Fold CV ROC-AUC:** 0.6625 ± 0.0069 | **5-Fold CV PR-AUC:** 0.2132
- **Holdout Test ROC-AUC:** **0.6682** [95% Bootstrap CI: 0.6558, 0.6804]
- **Holdout Test PR-AUC:** **0.2214** [95% Bootstrap CI: 0.2054, 0.2406]
- **Operating @ 0.50 Threshold:** Accuracy: 0.6623, Precision: 0.1811, Recall: 0.5541, Specificity: 0.6763, F1: 0.2730 ($\text{TN}=11,798, \text{FP}=5,647, \text{FN}=1,005, \text{TP}=1,249$)
- **Operating @ Out-of-Fold F1 Threshold ($t = 0.5237$):** Accuracy: 0.7059, Precision: 0.1926, Recall: 0.4920, Specificity: 0.7335, F1: **0.2768** ($\text{TN}=12,796, \text{FP}=4,649, \text{FN}=1,145, \text{TP}=1,109$)

---

### Suite C: Patient-Grouped Validation Suite (Protocol B Executed)
Executed via `scripts/run_replication.py` with 5-fold Stratified Group K-Fold cross-validation grouped strictly on `patient_nbr` (0% patient overlap).
- **Source of truth:** `results/replication/patient/A_RF/metrics.json`
- **Model Executed:** `A_RF` (Random Forest Classifier: 300 trees, `max_depth = 20`, `min_samples_leaf = 10`)
- **Training Partition:** 78,795 encounters (55,479 distinct patients)
- **Holdout Testing Partition:** 19,698 encounters (13,835 distinct patients)
- **Patient Overlap:** **0.0% (0 encounters)**
- **5-Fold Grouped CV ROC-AUC:** **0.6698 ± 0.0024** | **5-Fold Grouped CV PR-AUC:** **0.2183**
- **Unseen Patient Holdout ROC-AUC:** **0.6753** [95% Bootstrap CI: 0.6626, 0.6883]
- **Unseen Patient Holdout PR-AUC:** **0.2181** [95% Bootstrap CI: 0.2015, 0.2377]
- **Holdout Brier Score:** **0.2078** (verified directly on `results/replication/patient/A_RF/test_predictions.csv`; constant-prevalence baseline Brier score = **0.1013**, demonstrating that while class weighting achieves strong discriminative ranking with ROC-AUC 0.6753, uncalibrated scores shift the mean prediction to 0.4452, causing an elevated Brier score prior to probability calibration)
- **Operating @ 0.50 Threshold:** Accuracy: **0.7044**, Precision: **0.1989**, Recall: **0.5226**, Specificity: **0.7279**, F1: **0.2881** ($\text{TN}=12,698, \text{FP}=4,746, \text{FN}=1,076, \text{TP}=1,178$)
- **Operating @ Out-of-Fold F1 Threshold ($t = 0.5089$):** Accuracy: **0.7258**, Precision: **0.2045**, Recall: **0.4831**, Specificity: **0.7571**, F1: **0.2873** ($\text{TN}=13,207, \text{FP}=4,237, \text{FN}=1,165, \text{TP}=1,089$)
- **Operating @ Out-of-Fold Youden Threshold ($t = 0.4729$):** Accuracy: **0.6326**, Precision: **0.1798**, Recall: **0.6207**, Specificity: **0.6341**, F1: **0.2788** ($\text{TN}=11,061, \text{FP}=6,383, \text{FN}=855, \text{TP}=1,399$)

> **Methodological Boundary Note on Split Comparison:** Direct comparison between encounter-level Logistic Regression (ROC-AUC 0.6682) and patient-grouped Random Forest (ROC-AUC 0.6753) confounds the model family (linear model vs. non-linear decision forest) with the validation protocol (encounter vs. patient-grouped). Because an identical model architecture was not executed across both split protocols under the exact same feature space, the isolated effect of the patient-split protocol itself has not been experimentally isolated. Each model demonstrates defensible validation within its respective evaluation design.

---

### Suite D: Additional Improvement Suite (Ablation, Imbalance, Ensembling)
- **Implementation Status:** Architecture and search pipelines fully implemented in `scripts/run_improved.py` (including LightGBM feature block ablation B0, CatBoost B2, in-fold SMOTE B3, and soft-voting ensemble B4).
- **Execution Status:** Implemented in code; ready for execution. Not claimed as executed in final results to maintain strict scientific honesty.

---

## 4. Calibrated Random Forest Audit & Validation Threshold Realignment

To investigate the low recall artifact of the calibrated Random Forest model, we executed a dedicated evaluation (`scripts/audit_calibrated_model.py`):

1. **Why the Default 0.50 Threshold Fails:**
   - Uncalibrated class-weighted models shift scores into an artificial probability range centered at ~0.42. At $t=0.50$, recall is 55.57%.
   - Isotonic regression (`CalibratedClassifierCV(cv=3)`) calibrates output probabilities to match empirical incidence (mean predicted risk = **0.1116**, matching test base rate of **0.1116**).
   - Because true readmission is an 11% event, less than 0.2% of patients have estimated risk exceeding 0.50. At the arbitrary 0.50 cutoff, the calibrated model achieves an apparent accuracy of **88.91%**, but recall collapses to **1.14%** (only 26 of 2,271 readmissions caught; 2,245 missed).
2. **Threshold Realignment via Validation Tuning (No Test Leakage):**
   - Selecting the operating threshold on a separate validation split (without test label leakage) yields an optimal F1 cutoff of **$t = 0.1442$**:

| Operating Cutoff Mode | Threshold ($t$) | Accuracy | Precision | Recall | F1-Score | Specificity | True Positives | False Negatives | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Cutoff** | 0.5000 | 0.8891 | **0.6842** | 0.0114 | 0.0225 | **0.9993** | 26 | 2,245 | **0.0946** |
| **Validation F1-Optimal** | 0.1442 | 0.7406 | 0.2007 | 0.4443 | **0.2765** | 0.7778 | 1,009 | 1,262 | **0.0946** |
| **Validation Youden-Optimal** | 0.1393 | 0.7182 | 0.1941 | 0.4839 | 0.2771 | 0.7477 | 1,099 | 1,172 | **0.0946** |

> **Audit Conclusion:** Calibrated models are research prototypes requiring clinical cost-utility threshold selection; they cannot be called "deployment-ready" or clinically safe without prospective clinical validation.

---

## 5. SHAP Analysis Results vs. Ordinary Feature Importance

Using `shap.TreeExplainer` on holdout test instances, actual SHAP values were generated and visualized (`scripts/run_shap.py`):
- **Plots Saved:** `reports/figures/13_shap_summary.png` (beeswarm) and `reports/figures/14_shap_bar.png` (mean absolute magnitude).
- **Data Saved:** `results/shap_summary.json`.

### Table: Top 10 Features by Mean Absolute SHAP Impact (XGBoost Test Data)
| Rank | Feature | Mean \|SHAP Value\| | Empirical Association with Predicted Readmission Risk |
| :---: | :--- | :---: | :--- |
| 1 | `number_inpatient` | **0.2443** | Positive predictive association (corr = +0.969): higher prior inpatient visits associate with increased predicted log-odds. |
| 2 | `discharge_disposition_Transferred` | 0.1353 | Positive predictive association (corr = +0.945): transfer to skilled nursing or rehab associates with higher predicted risk. |
| 3 | `discharge_disposition_Home` | 0.0966 | Negative predictive association (corr = -0.936): direct discharge to home associates with lower predicted risk relative to facility transfers. |
| 4 | `service_utilization` | 0.0798 | Positive predictive association (corr = +0.764): cumulative prior healthcare utilization associates with higher predicted risk. |
| 5 | `n_active_medications` | 0.0778 | Positive predictive association: polypharmacy count associates with elevated predicted risk. |
| 6 | `num_medications` | 0.0722 | Positive predictive association: total inpatient medications administered during stay. |
| 7 | `discharge_disposition_Expired/Hospice` | 0.0712 | Deterministic negative association (corr = -0.990): in-hospital death or terminal hospice transition deterministically predicts zero readmission. |
| 8 | `num_lab_procedures` | 0.0659 | Positive predictive association: high laboratory procedure count correlates with diagnostic complexity. |
| 9 | `number_diagnoses` | 0.0626 | Positive predictive association (corr = +0.794): greater number of coded comorbid diagnoses. |
| 10 | `time_in_hospital` | 0.0611 | Positive predictive association: longer inpatient stays associate with higher predicted readmission log-odds. |

> **Critical Methodological Clarification on SHAP Interpretation:**
> 1. **Magnitude vs. Direction:** Mean absolute SHAP ($\text{mean } |\text{SHAP}|$) measures feature importance magnitude only (how much the feature shifts predictions on average), not direction. Directional associations reported above are derived from signed SHAP values across test instances and the beeswarm plot (`reports/figures/13_shap_summary.png`), not from mean absolute values alone.
> 2. **Associative, Not Causal:** SHAP attributions reflect statistical associations learned by the model from observational EHR data. They do not constitute causal clinical evidence; a negative attribution for discharge home does not "cause" or "prove" stabilization, nor does facility transfer "prove" frailty.
> 3. **Difference from Gini Importance:** Standard Gini/gain importance (`reports/figures/11_feature_importance.png`) measures frequency and impurity decrease during tree splitting on training data; SHAP measures directional, additive contributions to predicted log-odds on unseen patient encounters.

### 5.1 Structural Cohort Issue vs. Actionable Clinical Predictors
Feature #7 in the un-excluded test cohort, `discharge_disposition_Expired/Hospice` (mean \|SHAP\| = 0.0712), represents a **structural cohort constraint rather than a clinically useful risk predictor**. Patients who die during hospitalization or transition to end-of-life hospice care are mathematically ineligible for 30-day readmission. While gradient boosted trees exploit this feature to deterministically drive predicted readmission probabilities down to zero, this provides no actionable clinical utility for post-discharge transitional care among living patients.

### 5.2 Sub-Analysis: SHAP Interpretation on Eligible (Living) Cohort
**Exact Execution Procedure:** The original fitted model (`models/xgboost.joblib`) was retained without retraining. TreeExplainer SHAP values were evaluated specifically across a representative sample of eligible test encounters (excluding hospice and expired encounters; 19,877 eligible test encounters; `reports/figures/15_shap_bar_eligible.png` and `results/shap_summary_eligible.json`):
- In this eligible cohort, `discharge_disposition_Expired/Hospice` drops out entirely (mean \|SHAP\| = 0.0000).
- The dominant clinical drivers of readmission risk are:
  1. `number_inpatient` (mean \|SHAP\| = **0.2364**): Prior inpatient admissions represent the single strongest statistical driver of post-discharge readmission risk.
  2. `discharge_disposition_Transferred` (mean \|SHAP\| = **0.1325**): Transfer to skilled nursing, intermediate care, or rehabilitation facilities correlates with elevated readmission log-odds.
  3. `discharge_disposition_Home` (mean \|SHAP\| = **0.0970**): Direct discharge to home serves as a protective statistical signal relative to facility transfers.
  4. `service_utilization` (mean \|SHAP\| = **0.0799**): Cumulative prior healthcare contacts over the preceding 12 months.
  5. `n_active_medications` & `num_medications` (mean \|SHAP\| = **0.0781** & **0.0769**): Pharmacological complexity and polypharmacy burden.

---

## 6. Research Paper Comparison & Critical Literature Audit

> **Methodological Non-Equivalence Disclaimer:** Workspace evaluations and the reference paper results do not represent a strict head-to-head replication. Differences in preprocessing pipelines, cohort inclusion/exclusion sequencing, feature representations, cross-validation protocols, and decision thresholds preclude claiming that our models definitively outperform the published models. Rather, workspace experiments provide a transparent, reproducible benchmark under documented methodological controls.

### 6.1 Direct Metric Comparison

| Model | Paper Reported ROC-AUC | Workspace Test ROC-AUC | Paper Reported Accuracy | Workspace Test Accuracy | Paper Class 1 Recall | Workspace Class 1 Recall | Paper Class 1 F1 | Workspace Class 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.667 | **0.6709** | 0.89 | 0.6570 | 0.017 | **0.5764** | 0.032 | **0.2727** |
| **Logistic Regression** | 0.642 | **0.6684** | 0.89 | 0.6387 | 0.016 | **0.5997** | 0.030 | **0.2703** |
| **Random Forest** | 0.630 | **0.6777** | 0.89 | 0.6780 | 0.002 | **0.5557** | 0.003 | **0.2781** |
| **Deep Neural Net (DNN)** | 0.579 | **0.6750** | 0.84 | 0.6009 | 0.143 | **0.6609** | 0.161 | **0.2698** |

### 6.2 Critical Literature Discrepancies Documented
1. **Ambiguous Headline F1 Averaging:** Table 1 and the abstract of Emi-Johnson & Nkrumah reported an F1 score of 0.89 for XGBoost without explicitly defining the averaging methodology (macro, weighted, or negative-class). However, Table 2 of the published paper reported the true positive-class (<30-day readmission) F1 score as **0.032** (precision 43.7%, recall 1.6%). Comparing model performance against the paper's 0.89 headline figure would be invalid because the headline figure does not reflect positive-class performance.
2. **Internal Arithmetic Inconsistency:** Table 3 listed XGBoost errors as FP = 1,172 and FN = 2,244. Given 2,285 positive test cases, TP = 41. Actual precision is $41 / (41 + 1,172) = 0.0338$ (3.38%). However, Table 2 recorded Class 1 Precision as **0.437** (43.7%)—a 13-fold discrepancy within the paper's own tables.
3. **Cohort Partitioning Questions:** Stated exclusions of hospice/expired patients would yield an 80/20 test set of 19,869 encounters, whereas the paper reports 20,354 encounters (exactly 20.00% of the raw 101,766 dataset). While this matching raw-data test size raises a significant methodological question regarding whether and how exclusions were applied, it does not independently prove the exact implementation steps taken by the authors.

---

## 7. Execution Record: Successful vs. Constrained Experiments

### 7.1 Successfully Executed Experiments
- **Official UCI Dataset Acquisition & Hash Verification:** Verified SHA-256 digests of `diabetic_data.csv` and `IDs_mapping.csv`.
- **Baseline 5-Model Suite (`src/train.py`):** Fully trained and evaluated Logistic Regression, Random Forest, XGBoost, Keras MLP, and SMOTE+LogReg.
- **Serving Calibration Pipeline:** Successfully trained and serialized `CalibratedClassifierCV` (isotonic, 3-fold) on Random Forest (`models/best_model.joblib`).
- **Full Dataset & Leakage Audit (`scripts/audit_legacy.py`):** Produced `results/audit/legacy_audit.json`.
- **Reference Paper Retrieval & Extraction:** Retrieved and parsed full publication from PMC (PMC12085305).
- **Actual SHAP Analysis (`scripts/run_shap.py`):** Computed TreeExplainer values on 2,500 test instances; generated `13_shap_summary.png` and `14_shap_bar.png`.
- **Calibrated Model Threshold Audit (`scripts/audit_calibrated_model.py`):** Realigned decision cutoffs without test leakage (`results/calibrated_model_audit.json`).
- **5-Fold Cross-Validation Encounter Replication (`scripts/run_replication.py`):** Executed `A_LR` on 78,794 training encounters (`results/replication/encounter/A_LR/`).
- **5-Fold Patient-Grouped Replication (`scripts/run_replication.py`):** Executed `A_RF` with 5-fold Stratified Group K-Fold (0% patient overlap) on 78,795 encounters (`results/replication/patient/A_RF/`).
- **Academic Research Report & PDF Generation:** Built `reports/FINAL_PROJECT_REPORT.md` and compiled `reports/FINAL_PROJECT_REPORT.pdf` (51,885 bytes).
- **Interactive Research Dashboard:** Developed, verified, and smoke-tested 8-page Streamlit dashboard (`app/app.py`, `app/README.md`, `reports/DASHBOARD_GUIDE.md`).

### 7.2 Constrained Experiments / Implemented Roadmap
- **Full Replication Grid for All 4 Models Simultaneously:** Due to CPU constraints for 300-tree ensembles and deep neural networks across multiple protocols, Logistic Regression (`A_LR`) was executed for encounter protocol and Random Forest (`A_RF`) for patient-grouped protocol. XGBoost and DNN replication scripts are implemented and ready in `scripts/run_replication.py`.
- **Advanced Improvement Suite (B0–B4):** Implemented in `scripts/run_improved.py`; marked as implemented roadmap to avoid unsubstantiated claims.

---

## 8. Exact Paths of All Deliverables

### Reports & Documentation
- **Final Verification Report:** `FINAL_VERIFICATION.md`
- **Comprehensive Academic Report:** `reports/FINAL_PROJECT_REPORT.md`
- **Compiled PDF Research Report:** `reports/FINAL_PROJECT_REPORT.pdf`
- **Main Project Readme:** `README.md`
- **Dashboard Application:** `app/app.py`
- **Dashboard Readme:** `app/README.md`
- **Instructor Demonstration Guide:** `reports/DASHBOARD_GUIDE.md`
- **Model Comparison Table:** `reports/model_comparison.md`
- **Legacy Audit Record:** `results/audit/legacy_audit.json`
- **Calibrated Model Audit:** `results/calibrated_model_audit.json`
- **SHAP Summary Output:** `results/shap_summary.json`
- **SHAP Eligible Cohort Output:** `results/shap_summary_eligible.json`
- **Replication Encounter Results:** `results/replication/encounter/A_LR/metrics.json`
- **Replication Patient-Grouped Results:** `results/replication/patient/A_RF/metrics.json`
- **Reference Paper Text:** `docs/paper/paper_fulltext.txt`

### Visual Figures
- **SHAP Summary Beeswarm:** `reports/figures/13_shap_summary.png`
- **SHAP Feature Bar Plot:** `reports/figures/14_shap_bar.png`
- **SHAP Eligible Cohort Bar Plot:** `reports/figures/15_shap_bar_eligible.png`
- **ROC Curves:** `reports/figures/08_roc_curves.png`
- **Precision-Recall Curves:** `reports/figures/09_pr_curves.png`
- **Confusion Matrices:** `reports/figures/10_confusion_matrices.png`
- **Feature Importance:** `reports/figures/11_feature_importance.png`
- **Keras Training Trajectory:** `reports/figures/12_keras_training_history.png`

---

## 9. Exact Commands to Reproduce Final Results

```bash
# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run automated dataset integrity and leakage audit
python scripts/audit_legacy.py

# 3. Train and evaluate all 5 primary baseline models
python src/train.py --quick

# 4. Generate all evaluation figures (ROC, PR, Confusion matrices, Feature importance)
python src/evaluate.py

# 5. Compute actual SHAP values and generate beeswarm summary plots
python scripts/run_shap.py

# 6. Audit calibrated model and evaluate validation-derived threshold
python scripts/audit_calibrated_model.py

# 7. Execute 5-fold cross-validation encounter replication (Logistic Regression)
python scripts/run_replication.py --models logreg --protocols encounter --skip-sensitivity

# 8. Execute 5-fold patient-grouped cross-validation replication (Random Forest, 0% overlap)
python scripts/run_replication.py --models rf --protocols patient --skip-sensitivity

# 9. Compile PDF research report from markdown
python scripts/generate_report_pdf.py

# 10. Launch interactive Streamlit clinical demonstration application
streamlit run app/app.py
```

---
*Final internal technical audit based on available project artifacts.*
