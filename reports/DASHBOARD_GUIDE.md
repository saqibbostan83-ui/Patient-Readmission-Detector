# Instructor Demonstration & Viva Defense Guide: Streamlit Research Dashboard

**Project:** Predicting 30-Day Hospital Readmission in Patients With Diabetes  
**Application:** `app/app.py`  
**Target Audience:** University Evaluator / Academic Supervisor / Viva Committee  

---

## 1. Quick Launch Command

To start the presentation dashboard, open PowerShell or terminal in the project root and execute:

```powershell
.\.venv\Scripts\streamlit run app/app.py
```

The application will immediately compile and open in your default browser at `http://localhost:8501`.

---

## 2. Recommended 3–5 Minute Demonstration Sequence

When presenting this project to an instructor or evaluator, use this structured sequence:

### Minute 1: Research Problem, Dataset & Pipeline (Pages 1 & 2)
1. **Open Page 1 (Project Overview):**
   - Highlight the high-level KPI cards: 101,766 hospital encounters across 130 US medical facilities (1999–2008).
   - Point out the low positive prevalence (**11.16%** readmitted within 30 days) and emphasize that this is a **discharge-time prediction problem** designed to guide post-discharge transitional care.
2. **Switch briefly to Page 2 (Exploratory Data Analysis):**
   - Show the class imbalance chart (Figure 1) and missing data profile (Figure 2: `weight` 97% missing, `payer_code` 40% missing).
   - Highlight that prior inpatient admissions ($\ge 2$) double the observed readmission rate to >25%.

### Minute 2: Modeling Benchmark & Why Accuracy Fails (Page 3)
3. **Navigate to Page 3 (Model Performance Benchmark):**
   - Present the 5-model comparison table: Tuned Random Forest leads with **ROC-AUC 0.6777** and **PR-AUC 0.2167**, closely followed by the Keras MLP (ROC-AUC 0.6750) and XGBoost (0.6709).
   - **Crucial Discussion Point:** Point out that Calibrated Random Forest achieves **88.91% accuracy** at threshold 0.50, but **only 1.14% recall** (missing 2,245 of 2,271 readmissions). Explain that accuracy in imbalanced clinical data merely reflects majority-class prediction.
   - Show the interactive ROC and PR curves.

### Minute 3: Data Leakage & Patient-Grouped Generalization (Page 4)
4. **Navigate to Page 4 (Validation & Leakage Audit):**
   - **Highlight the key methodological contribution:** In standard encounter splits, **41.6% of test encounters belong to patients who also appear in training data** (patient identity leakage).
   - Show our executed `StratifiedGroupKFold` experiment (`A_RF`), which guarantees **0.0% patient overlap**. On 19,698 encounters from completely unseen patients, the model achieves **ROC-AUC 0.6753** and **PR-AUC 0.2181**.
   - Note the boundary condition: different model families (LR vs RF) preclude claiming the split alone caused the score difference.

### Minute 4: Explainability & Eligible Cohort SHAP (Page 5)
5. **Navigate to Page 5 (Explainable AI & SHAP):**
   - Show the TreeExplainer beeswarm plot (Figure 13) and explain that `number_inpatient` is the single strongest statistical driver of risk (mean $|SHAP| = 0.244$).
   - **Show the Eligible Cohort Tab (Figure 15):** Explain that `discharge_disposition_Expired/Hospice` in un-excluded data is a **structural cohort artifact** (in-hospital death deterministically prevents readmission). Show that when evaluated on the eligible living cohort (19,877 encounters), it drops out ($|SHAP| = 0.000$) and true clinical predictors govern risk: prior inpatient visits, transfer to other facilities, discharge home, and medication burden.
   - Reiterate that mean $|SHAP|$ measures magnitude, not direction, and reflects predictive associations rather than causal clinical mechanisms.

### Minute 5: Interactive Scoring Demonstration & Critical Audit (Pages 6 & 7)
6. **Navigate to Page 6 (Interactive Prediction Demo):**
   - Demonstrate the interactive form pre-populated with median patient defaults (Age 65, 0 prior visits, 4-day stay, diabetes medication).
   - Click **Compute Calibrated Readmission Risk**: show that baseline risk is **8.9%** (below the validation-selected threshold of 14.42% $\rightarrow$ "Standard Risk").
   - Now change **Inpatient Admissions to 3** and **Discharge Destination to 'Transferred'**, then re-click Predict:
     - The predicted risk jumps above 15% $\rightarrow$ **"HIGH RISK (Flagged for Post-Discharge Support)"**.
   - Emphasize the educational disclaimer: research prototype, not a medical device.
7. **Conclude with Page 7 (Research Paper Comparison):**
   - Point out the published literature contradictions documented in Emi-Johnson & Nkrumah (2025): the headline 0.89 F1 score which Table 2 proves was only **0.032** on the positive class, and the 13-fold internal arithmetic conflict in Table 3.

---

## 3. Key Metrics to Memorize for Questions

| Metric / Parameter | Value | Significance |
| :--- | :---: | :--- |
| **Total Cohort Size** | 101,766 | Full dataset size from 130 US hospitals (1999–2008) |
| **Test Set Size** | 20,354 | 20.0% holdout test partition (2,271 positive readmissions) |
| **Base Readmission Prevalence** | **11.16%** | Severe class imbalance (~8:1 ratio) |
| **Best Baseline ROC-AUC** | **0.6777** | Tuned Random Forest (class-weighted) |
| **Best Baseline PR-AUC** | **0.2167** | Tuned Random Forest (doubles 0.11 base rate) |
| **Patient-Grouped ROC-AUC** | **0.6753** | Random Forest with **0.0% patient overlap** (Protocol B) |
| **Calibrated Test Brier Score** | **0.0946** | Isotonic calibration matches empirical risk (mean risk = 0.1116) |
| **Validation-Derived Threshold** | **$t = 0.1442$** | Realigns recall to 44.43% and F1 to 0.2765 without test leakage |

---

## 4. Key Limitations to Transparently Disclose

If the instructor asks: *"What are the limitations of this model?"*, state these four points clearly:
1. **Discrimination Ceiling on Administrative EHR:** ROC-AUC values peak around 0.67–0.68 because administrative billing codes lack post-discharge social determinants of health (housing, caregiver support, medication adherence).
2. **Discharge-Time Operational Scope:** The model requires variables established during the completed stay (length of stay, discharge disposition); it cannot act as an admission screener.
3. **Historical Data:** The dataset is from 1999–2008; temporal drift and modern diabetes management guidelines require prospective recalibration.
4. **Calibration Threshold Sensitivity:** Calibrated models cannot use an arbitrary 0.50 cutoff on imbalanced data; decision thresholds must be selected based on hospital intervention capacity.
