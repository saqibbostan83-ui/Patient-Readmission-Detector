# Streamlit Research Dashboard: Diabetes 30-Day Readmission Prediction

This interactive clinical research dashboard accompanies the academic project predicting 30-day hospital readmission in patients with diabetes using the UCI Diabetes 130-US Hospitals dataset (1999–2008), following and auditing the published study by Emi-Johnson & Nkrumah (2025), *Cureus*, 17(4), e82437.

---

## 1. Quick Start / How to Run

Ensure your virtual environment is activated, then run:

```bash
# From the project root directory:
.\.venv\Scripts\streamlit run app/app.py
```

Or using standard python:
```bash
python -m streamlit run app/app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 2. Architecture & Design

- **Theme:** Professional clinical research interface with a navy/teal visual identity (`#1B365D` and `#0D5C75`).
- **Data Caching:** Uses `@st.cache_resource` and `@st.cache_data` for instantaneous page transitions without retraining models.
- **Safety:** Operates strictly on pre-computed model artifacts (`models/best_model.joblib`), verified evaluation metrics, and anonymized aggregate data. No private patient identifiers are used.

---

## 3. Dashboard Structure (8 Research Pages)

1. **1. Project Overview:** High-level project KPIs, research motivation, dataset parameters (101,766 encounters, 50 features), discharge-time prediction scope, and end-to-end ML pipeline architecture.
2. **2. Exploratory Data Analysis:** Class imbalance visualization (11.16% readmission rate), missing value profiles, clinical driver distributions (stay length, prior visits), and diabetes medication regimens.
3. **3. Model Performance Benchmark:** Detailed comparison of 5 baseline models (Random Forest, Keras MLP, XGBoost, Logistic Regression, SMOTE), ROC and PR curves, interactive confusion matrices, and analysis of why accuracy is misleading in clinical EHR.
4. **4. Validation & Leakage Audit:** Encounter-level vs. patient-grouped validation, empirical demonstration of 41.6% patient identity leakage in naive splits vs. 0.0% overlap under `StratifiedGroupKFold` (`A_RF`: ROC-AUC 0.6753), and threshold leakage correction.
5. **5. Explainable AI & SHAP:** Global TreeExplainer SHAP beeswarm and magnitude plots, clarification of magnitude vs. direction, structural cohort constraint of expired/hospice cases, and sub-analysis of the eligible living patient cohort.
6. **6. Interactive Prediction Demo:** Form allowing hypothetical clinical encounter entry, instant calibrated risk probability calculation, population base rate comparison (11.16%), and risk stratification relative to the validation-derived threshold ($t = 0.1442$).
7. **7. Research Paper Comparison:** Contextual benchmark against Emi-Johnson & Nkrumah (2025), documenting published contradictions (unspecified headline F1 averaging, internal arithmetic conflict, and raw data cohort size).
8. **8. Limitations & Conclusion:** Honest scientific boundaries (historical dataset, lack of social determinants, discharge-time constraint, prospective validation requirement) and executive conclusion.

---

## 4. Academic Disclaimer

*Academic Research Prototype | UCI Diabetes 130-US Hospitals Dataset | Reference: Cureus 17(4), e82437 | Not for Clinical Use.*
