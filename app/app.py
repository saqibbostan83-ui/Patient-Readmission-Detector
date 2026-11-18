"""
Diabetes 30-Day Readmission Prediction
An Explainable Machine Learning Research Dashboard Using the UCI Diabetes 130-US Hospitals Dataset

Academic Research Dashboard
Reference Study: Emi-Johnson & Nkrumah (2025), Cureus, 17(4), e82437
Dataset: UCI Machine Learning Repository (ID 296)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# ------------------------------------------------------------------ #
# Page Configuration & Professional Navy/Teal Styling
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Diabetes Readmission Research Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Professional Clinical Research Theme */
    :root {
        --primary-navy: #1B365D;
        --secondary-teal: #0D5C75;
        --accent-teal: #008080;
        --bg-light: #F8FAFC;
        --card-border: #E2E8F0;
        --text-dark: #1E293B;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1B365D 0%, #0D5C75 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-top: 4px solid #0D5C75;
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #1B365D;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .alert-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 16px 0;
        color: #1E3A8A;
        font-size: 0.95rem;
    }
    
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 16px 0;
        color: #92400E;
        font-size: 0.95rem;
    }

    .footer {
        text-align: center;
        padding: 24px;
        color: #64748B;
        font-size: 0.85rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Cached Data & Model Loaders
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model_pipeline():
    try:
        model_path = MODELS_DIR / "best_model.joblib"
        if model_path.exists():
            return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Note: Model pipeline could not be loaded: {e}")
    return None

@st.cache_data
def load_all_metrics():
    metrics = {}
    
    # 1. Baseline metrics
    if (MODELS_DIR / "metrics.json").exists():
        with open(MODELS_DIR / "metrics.json") as f:
            metrics["legacy"] = json.load(f)
            
    # 2. Calibrated audit
    if (RESULTS_DIR / "calibrated_model_audit.json").exists():
        with open(RESULTS_DIR / "calibrated_model_audit.json") as f:
            metrics["calibrated_audit"] = json.load(f)
            
    # 3. Default patient
    if (MODELS_DIR / "default_patient.json").exists():
        with open(MODELS_DIR / "default_patient.json") as f:
            metrics["default_patient"] = json.load(f)
            
    # 4. Replication encounter
    p_enc = RESULTS_DIR / "replication" / "encounter" / "A_LR" / "metrics.json"
    if p_enc.exists():
        with open(p_enc) as f:
            metrics["rep_encounter"] = json.load(f)
            
    # 5. Replication patient
    p_pat = RESULTS_DIR / "replication" / "patient" / "A_RF" / "metrics.json"
    if p_pat.exists():
        with open(p_pat) as f:
            metrics["rep_patient"] = json.load(f)
            
    # 6. SHAP summaries
    if (RESULTS_DIR / "shap_summary.json").exists():
        with open(RESULTS_DIR / "shap_summary.json") as f:
            metrics["shap_full"] = json.load(f)
            
    if (RESULTS_DIR / "shap_summary_eligible.json").exists():
        with open(RESULTS_DIR / "shap_summary_eligible.json") as f:
            metrics["shap_eligible"] = json.load(f)
            
    return metrics


# Load cached resources
model_pipeline = load_model_pipeline()
metrics_data = load_all_metrics()

# ------------------------------------------------------------------ #
# Sidebar Navigation
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <h2 style="color: #1B365D; margin-bottom: 2px;">🏥 ML Dashboard</h2>
        <p style="color: #64748B; font-size: 0.85rem; margin: 0;">UCI Diabetes Readmission Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "1. Project Overview",
            "2. Exploratory Data Analysis",
            "3. Model Performance Benchmark",
            "4. Validation & Leakage Audit",
            "5. Explainable AI & SHAP",
            "6. Interactive Prediction Demo",
            "7. Research Paper Comparison",
            "8. Limitations & Conclusion",
        ],
        index=0,
    )
    
    st.divider()
    
    st.markdown("""
    <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
        <strong>Dataset:</strong> UCI ID 296 (101,766 encounters)<br>
        <strong>Reference:</strong> Cureus 17(4), e82437 (2025)<br>
        <strong>Prediction Target:</strong> Readmission &lt;30 Days<br>
        <strong>Prediction Time:</strong> Discharge-Time
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("Developed for Academic Research & Viva Defense")


# ------------------------------------------------------------------ #
# Header Banner (Across All Pages)
# ------------------------------------------------------------------ #
st.markdown("""
<div class="main-header">
    <h1>Diabetes 30-Day Hospital Readmission Prediction</h1>
    <p>An Explainable Machine Learning Research Dashboard Using the UCI Diabetes 130-US Hospitals Dataset</p>
</div>
""", unsafe_allow_html=True)


# ================================================================== #
# PAGE 1: PROJECT OVERVIEW
# ================================================================== #
if page == "1. Project Overview":
    st.subheader("1. Research Scope & Executive Summary")
    
    # KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Raw Encounters</div>
            <div class="kpi-value">101,766</div>
            <div class="kpi-sub">130 US Hospitals (1999–2008)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Positive Rate (&lt;30d)</div>
            <div class="kpi-value">11.16%</div>
            <div class="kpi-sub">Severe Class Imbalance</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Best Baseline AUC</div>
            <div class="kpi-value">0.6777</div>
            <div class="kpi-sub">Tuned Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Patient Protocol AUC</div>
            <div class="kpi-value">0.6753</div>
            <div class="kpi-sub">0.0% Patient Overlap (RF)</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Calibrated Brier</div>
            <div class="kpi-value">0.0946</div>
            <div class="kpi-sub">Empirical Risk Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.markdown("### Research Objectives")
        st.markdown("""
        Hospital readmission within 30 days is both clinically adverse and subject to Centers for Medicare & Medicaid Services (CMS) reimbursement penalties. This project audits, reproduces, and extends recent research on electronic health record (EHR) data:
        
        1. **Reproduce Published Benchmarks:** Implement the core model architectures from Emi-Johnson & Nkrumah (2025): Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks (MLP).
        2. **Audit Data Leakage:** Investigate patient clustering across repeated encounters, threshold-selection leakage, and feature pipeline containment.
        3. **Evaluate Patient-Grouped Generalization:** Implement a strict `StratifiedGroupKFold` protocol (0% patient overlap) to evaluate model performance on completely unseen individuals.
        4. **Audit Probability Calibration:** Characterize the behavior of isotonic calibration under imbalanced clinical data and resolve decision threshold collapse.
        5. **Provide Model Explainability:** Use TreeExplainer SHAP to determine the primary statistical drivers of risk and separate structural cohort constraints from actionable clinical predictors.
        """)
        
        st.markdown("""
        <div class="alert-box">
            <strong>Operational Timing: Discharge-Time Prediction</strong><br>
            All models in this project operate at <strong>discharge-time</strong>. They utilize features accumulated over the full hospital course (total length of stay, total laboratory and procedure counts, medication adjustments, and final discharge disposition) to guide post-discharge transitional care triage.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Research Pipeline Architecture")
        pipeline_df = pd.DataFrame([
            {"Stage": "1. Data Acquisition", "Method": "UCI ID 296 Archive (SHA-256 Verified)", "Status": "Verified"},
            {"Stage": "2. Preprocessing & Encoding", "Method": "ICD-9 Mapping, One-Hot, Missing Categorization", "Status": "Leakage-Safe"},
            {"Stage": "3. Baseline Modeling", "Method": "5 Models on 80/20 Encounter Split (20,354 Test)", "Status": "Completed"},
            {"Stage": "4. Patient-Grouped Protocol", "Method": "5-Fold StratifiedGroupKFold (0% Patient Overlap)", "Status": "Completed (RF)"},
            {"Stage": "5. Probability Calibration", "Method": "Isotonic CalibratedClassifierCV (3-Fold)", "Status": "Audited"},
            {"Stage": "6. Model Explainability", "Method": "TreeExplainer SHAP (Full & Eligible Cohorts)", "Status": "Completed"},
            {"Stage": "7. Interactive Decision Support", "Method": "Streamlit Clinical Demonstration UI", "Status": "Active"},
        ])
        st.dataframe(pipeline_df, hide_index=True, use_container_width=True)


# ================================================================== #
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ================================================================== #
elif page == "2. Exploratory Data Analysis":
    st.subheader("2. Exploratory Clinical & Cohort Analysis")
    st.markdown("Analysis of the 101,766 hospital encounters across 130 US medical facilities (1999–2008).")
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Target Distribution & Imbalance",
        "Missing Value Patterns",
        "Key Clinical Drivers",
        "Medications & Comorbidities"
    ])
    
    with eda_tab1:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### Class Distribution (<30-Day Readmission)")
            st.markdown("""
            - **Positive Class (`<30` days):** 11,357 encounters (**11.16%**)
            - **Negative Class (`>30` days or `NO`):** 90,409 encounters (**88.84%**)
            
            This severe class imbalance (~8:1 ratio) means naive classifiers predicting 'no readmission' achieve 88.84% accuracy with 0% recall. Consequently, accuracy is a misleading metric; models must be evaluated using **ROC-AUC, PR-AUC, and clinical sensitivity**.
            """)
        with c2:
            fig_path = FIGURES_DIR / "01_target_distribution.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Figure 1: Target Class Balance in Full Dataset", use_container_width=True)
            else:
                st.info("Target distribution figure available in report artifacts.")

    with eda_tab2:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### Missing Value Patterns")
            st.markdown("""
            Three features in the raw dataset exhibit heavy missingness:
            - **`weight`:** 96.85% missing (infrequently recorded during routine inpatient stays)
            - **`medical_specialty`:** 49.08% missing (retained as 'Unknown' category)
            - **`payer_code`:** 39.56% missing (administrative billing code dropped)
            
            Following rigorous data science practices, `weight` and `payer_code` are excluded from predictive modeling, while `medical_specialty` is encoded with an explicit missingness indicator to preserve institutional signal.
            """)
        with c2:
            fig_path = FIGURES_DIR / "02_missing_values.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Figure 2: Missing Value Frequency Across Columns", use_container_width=True)

    with eda_tab3:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### Clinical Determinants of Readmission")
            st.markdown("""
            Bivariate analysis reveals strong empirical gradients:
            - **Prior Inpatient Admissions:** Patients with >= 2 prior inpatient admissions in the preceding 12 months exhibit readmission rates exceeding **25%** (more than 2.2× baseline).
            - **Discharge Disposition:** Patients discharged to other medical facilities (SNF, intermediate care) have elevated readmission rates compared to those discharged home.
            - **Length of Stay:** Longer hospitalization stays correlate positively with readmission probability.
            """)
        with c2:
            fig_path = FIGURES_DIR / "04_readmission_by_key_features.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Figure 4: Readmission Rates by Key Clinical Factors", use_container_width=True)

    with eda_tab4:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("#### Diabetes Pharmacotherapy & Comorbidities")
            st.markdown("""
            - **Insulin Regimen:** Insulin is the most commonly prescribed medication (~53% of encounters). Any dosage increase or modification during hospitalization correlates with elevated risk.
            - **Primary Diagnoses:** Circulatory conditions (ICD-9 390–459) represent over 30% of admissions, followed by endocrine/diabetes (25%) and respiratory disorders (14%).
            """)
        with c2:
            fig_path = FIGURES_DIR / "07_medication_usage.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Figure 7: Medication Frequency and Adjustments", use_container_width=True)


# ================================================================== #
# PAGE 3: MODEL PERFORMANCE COMPARISON
# ================================================================== #
elif page == "3. Model Performance Benchmark":
    st.subheader("3. Model Performance Benchmark (Holdout Test Set)")
    st.markdown("Evaluated on 20,354 holdout encounters (2,271 positive `<30` days, 18,083 negative; prevalence = 11.16%).")
    
    # Summary Table of 5 Baseline Models + Calibrated
    models_table = pd.DataFrame([
        {
            "Model Architecture": "Random Forest (Tuned) 🏆",
            "ROC-AUC": 0.6777,
            "PR-AUC": 0.2167,
            "Accuracy": 0.6780,
            "Precision": 0.1854,
            "Recall (Sens.)": 0.5557,
            "Specificity": 0.6934,
            "F1-Score": 0.2781,
            "Brier Score": 0.2078,
            "Fit Time": "14.5s"
        },
        {
            "Model Architecture": "Keras MLP (3-Layer DNN)",
            "ROC-AUC": 0.6750,
            "PR-AUC": 0.2126,
            "Accuracy": 0.6009,
            "Precision": 0.1695,
            "Recall (Sens.)": 0.6609,
            "Specificity": 0.5934,
            "F1-Score": 0.2698,
            "Brier Score": 0.2319,
            "Fit Time": "18.7s"
        },
        {
            "Model Architecture": "XGBoost (Hist/Tuned)",
            "ROC-AUC": 0.6709,
            "PR-AUC": 0.2154,
            "Accuracy": 0.6570,
            "Precision": 0.1786,
            "Recall (Sens.)": 0.5764,
            "Specificity": 0.6671,
            "F1-Score": 0.2727,
            "Brier Score": 0.2163,
            "Fit Time": "3.6s"
        },
        {
            "Model Architecture": "Logistic Regression (Balanced)",
            "ROC-AUC": 0.6684,
            "PR-AUC": 0.2077,
            "Accuracy": 0.6387,
            "Precision": 0.1745,
            "Recall (Sens.)": 0.5997,
            "Specificity": 0.6436,
            "F1-Score": 0.2703,
            "Brier Score": 0.2241,
            "Fit Time": "3.6s"
        },
        {
            "Model Architecture": "LogReg + SMOTE",
            "ROC-AUC": 0.6535,
            "PR-AUC": 0.1996,
            "Accuracy": 0.6284,
            "Precision": 0.1662,
            "Recall (Sens.)": 0.5804,
            "Specificity": 0.6345,
            "F1-Score": 0.2585,
            "Brier Score": 0.2305,
            "Fit Time": "6.8s"
        },
        {
            "Model Architecture": "Calibrated RF (Served Model @ 0.50)",
            "ROC-AUC": 0.6772,
            "PR-AUC": 0.2165,
            "Accuracy": 0.8891,
            "Precision": 0.6842,
            "Recall (Sens.)": 0.0114,
            "Specificity": 0.9993,
            "F1-Score": 0.0225,
            "Brier Score": 0.0946,
            "Fit Time": "1.8s"
        },
    ])
    
    st.dataframe(
        models_table.style.highlight_max(subset=["ROC-AUC", "PR-AUC", "Recall (Sens.)", "F1-Score"], color="#D1FAE5")
                          .highlight_min(subset=["Brier Score"], color="#E0F2FE"),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("*Data source: `models/metrics.json` and `results/calibrated_model_audit.json`. Brier score of calibrated model is 0.0946 on the full test set.*")
    
    st.markdown("""
    <div class="warning-box">
        <strong>Why Accuracy is Misleading for Clinical EHR:</strong><br>
        Notice that Calibrated RF achieves <strong>88.91% accuracy</strong> at the default 0.50 cutoff, yet identifies only 26 of 2,271 readmissions (<strong>1.14% recall</strong>). In low-prevalence screening, accuracy merely reflects majority-class prediction. Effective clinical evaluation must focus on <strong>ROC-AUC, PR-AUC, Brier score, and validation-tuned sensitivity</strong>.
    </div>
    """, unsafe_allow_html=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        p_roc = FIGURES_DIR / "08_roc_curves.png"
        if p_roc.exists():
            st.image(str(p_roc), caption="Figure 8: Comparative Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
    with col_fig2:
        p_pr = FIGURES_DIR / "09_pr_curves.png"
        if p_pr.exists():
            st.image(str(p_pr), caption="Figure 9: Precision-Recall Curves (Baseline Prevalence = 0.1116)", use_container_width=True)

    st.markdown("### Interactive Confusion Matrix Inspector")
    cm_model = st.selectbox(
        "Select Model Architecture to View Confusion Matrix Details:",
        ["Random Forest (Tuned)", "Keras MLP (DNN)", "XGBoost", "Logistic Regression", "LogReg + SMOTE", "Calibrated RF (@ 0.50)"]
    )
    
    cm_data = {
        "Random Forest (Tuned)": {"tn": 12539, "fp": 5544, "fn": 1009, "tp": 1262},
        "Keras MLP (DNN)": {"tn": 10730, "fp": 7353, "fn": 770, "tp": 1501},
        "XGBoost": {"tn": 12064, "fp": 6019, "fn": 962, "tp": 1309},
        "Logistic Regression": {"tn": 11638, "fp": 6445, "fn": 909, "tp": 1362},
        "LogReg + SMOTE": {"tn": 11473, "fp": 6610, "fn": 953, "tp": 1318},
        "Calibrated RF (@ 0.50)": {"tn": 18071, "fp": 12, "fn": 2245, "tp": 26},
    }
    
    selected_cm = cm_data[cm_model]
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("True Negatives (TN)", f"{selected_cm['tn']:,}")
    with c_m2:
        st.metric("False Positives (FP)", f"{selected_cm['fp']:,}")
    with c_m3:
        st.metric("False Negatives (FN)", f"{selected_cm['fn']:,}")
    with c_m4:
        st.metric("True Positives (TP)", f"{selected_cm['tp']:,}")


# ================================================================== #
# PAGE 4: VALIDATION & LEAKAGE AUDIT
# ================================================================== #
elif page == "4. Validation & Leakage Audit":
    st.subheader("4. Critical Data Leakage & Validation Protocol Audit")
    
    st.markdown("""
    In clinical machine learning, naive evaluation protocols frequently introduce subtle data leakage that leads to overly optimistic performance claims. We conducted rigorous technical audits of the evaluation protocols:
    """)
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.markdown("### Protocol A vs. Protocol B Comparison")
        val_df = pd.DataFrame([
            {"Dimension": "Partitioning Unit", "Protocol A (Encounter-Level)": "Hospital Encounter (`encounter_id`)", "Protocol B (Patient-Grouped)": "Unique Patient (`patient_nbr`)"},
            {"Dimension": "Training Encounters", "Protocol A (Encounter-Level)": "78,794 encounters (58,303 patients)", "Protocol B (Patient-Grouped)": "78,795 encounters (55,479 patients)"},
            {"Dimension": "Test Encounters", "Protocol A (Encounter-Level)": "19,699 encounters (17,787 patients)", "Protocol B (Patient-Grouped)": "19,698 encounters (13,835 patients)"},
            {"Dimension": "Patient Overlap in Test", "Protocol A (Encounter-Level)": "8,223 encounters (41.74% LEAKAGE)", "Protocol B (Patient-Grouped)": "0 encounters (0.0% LEAKAGE)"},
            {"Dimension": "Cross-Validation", "Protocol A (Encounter-Level)": "5-Fold Stratified K-Fold", "Protocol B (Patient-Grouped)": "5-Fold Stratified Group K-Fold"},
            {"Dimension": "Model Architecture", "Protocol A (Encounter-Level)": "Logistic Regression (`A_LR`)", "Protocol B (Patient-Grouped)": "Random Forest (`A_RF`)"},
            {"Dimension": "Holdout Test ROC-AUC", "Protocol A (Encounter-Level)": "0.6682 [95% CI: 0.6558, 0.6804]", "Protocol B (Patient-Grouped)": "0.6753 [95% CI: 0.6626, 0.6883]"},
            {"Dimension": "Holdout Test PR-AUC", "Protocol A (Encounter-Level)": "0.2214 [95% CI: 0.2054, 0.2406]", "Protocol B (Patient-Grouped)": "0.2181 [95% CI: 0.2015, 0.2377]"},
            {"Dimension": "Holdout Test Brier Score", "Protocol A (Encounter-Level)": "0.2272", "Protocol B (Patient-Grouped)": "0.2078 (Baseline: 0.1013)"},
        ])
        st.dataframe(val_df, hide_index=True, use_container_width=True)
        
    with c2:
        st.markdown("""
        <div class="alert-box">
            <strong>Key Finding: Patient Identity Leakage</strong><br>
            When hospital encounters are partitioned randomly without grouping, <strong>41.6% of test encounters belong to patients who also appear in the training partition</strong>.<br><br>
            Under our strict <code>StratifiedGroupKFold</code> protocol (0% patient overlap), Random Forest achieved <strong>ROC-AUC 0.6753</strong> and <strong>PR-AUC 0.2181</strong> on unseen patients, proving that discriminative capacity is preserved when evaluated prospectively.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>Methodological Boundary Note:</strong><br>
            Because encounter-level replication evaluated Logistic Regression (linear) while patient-grouped replication evaluated Random Forest (non-linear ensemble), the isolated effect of patient grouping alone was not tested holding model architecture constant. Each model demonstrates performance under its validated protocol.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Threshold Optimization Leakage Audit")
    st.markdown("""
    - **Vulnerable Practice Identified:** Legacy scripts evaluated optimal F1 decision thresholds directly against test labels `y_test`.
    - **Corrected Methodology:** In the research pipeline (`src/research/experiment.py`), decision thresholds are tuned strictly on **Out-Of-Fold (OOF)** cross-validation predictions, preserving test set isolation.
    """)


# ================================================================== #
# PAGE 5: EXPLAINABLE AI & SHAP
# ================================================================== #
elif page == "5. Explainable AI & SHAP":
    st.subheader("5. Explainable AI & SHAP Feature Attributions")
    st.markdown("TreeExplainer SHAP values computed on holdout test instances for the fitted XGBoost ensemble.")
    
    st.markdown("""
    <div class="alert-box">
        <strong>Methodological Principles for SHAP Interpretation:</strong><br>
        1. <strong>Magnitude vs. Direction:</strong> Mean absolute SHAP (mean |SHAP|) quantifies overall impact magnitude on model log-odds output, but does not indicate direction. Directional associations must be deduced from signed values and the beeswarm distribution.<br>
        2. <strong>Predictive Associations, Not Causal Claims:</strong> SHAP values describe statistical relationships learned by the machine learning algorithm from observational EHR data. They do not constitute causal clinical evidence (e.g., discharge home does not 'cause' clinical stabilization).
    </div>
    """, unsafe_allow_html=True)

    shap_tab1, shap_tab2 = st.tabs(["Full Cohort SHAP", "Eligible (Living) Cohort Sub-Analysis"])
    
    with shap_tab1:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            p_beeswarm = FIGURES_DIR / "13_shap_summary.png"
            if p_beeswarm.exists():
                st.image(str(p_beeswarm), caption="Figure 13: SHAP Beeswarm Plot (Full Test Sample)", use_container_width=True)
        with c2:
            p_bar = FIGURES_DIR / "14_shap_bar.png"
            if p_bar.exists():
                st.image(str(p_bar), caption="Figure 14: Top 20 Features by Mean |SHAP|", use_container_width=True)
                
            st.markdown("""
            **Key Full-Cohort Observations:**
            - `number_inpatient` is the single strongest statistical driver (mean |SHAP| = 0.244, correlation with signed SHAP = +0.969).
            - `discharge_disposition_Transferred` correlates positively with elevated risk.
            - `discharge_disposition_Expired/Hospice` shows high importance (mean |SHAP| = 0.071), but represents a **structural cohort artifact** (in-hospital death deterministically prevents readmission).
            """)

    with shap_tab2:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            p_eligible = FIGURES_DIR / "15_shap_bar_eligible.png"
            if p_eligible.exists():
                st.image(str(p_eligible), caption="Figure 15: SHAP Bar Plot (Eligible Living Cohort)", use_container_width=True)
        with c2:
            st.markdown("""
            ### Eligible Cohort Analysis Procedure
            **Exact Execution Procedure:** The original fitted model (`models/xgboost.joblib`) was retained without retraining. TreeExplainer SHAP values were calculated specifically on the subset of eligible test encounters (excluding hospice and expired encounters; 19,877 eligible test encounters).
            
            **Empirical Findings:**
            - In the eligible living population, `discharge_disposition_Expired/Hospice` drops out entirely (mean |SHAP| = 0.0000).
            - The true clinical predictors governing risk ranking for living patients at discharge are:
              1. **Prior inpatient admissions** (`number_inpatient`, mean |SHAP| = **0.2364**)
              2. **Discharge transfer** (`discharge_disposition_Transferred`, mean |SHAP| = **0.1325**)
              3. **Discharge home** (`discharge_disposition_Home`, mean |SHAP| = **0.0970**)
              4. **Service utilization** (mean |SHAP| = **0.0799**)
              5. **Active medications** (mean |SHAP| = **0.0781**)
            """)


# ================================================================== #
# PAGE 6: INTERACTIVE PREDICTION DEMO
# ================================================================== #
elif page == "6. Interactive Prediction Demo":
    st.subheader("6. Interactive Clinical Decision-Support Demonstration")
    st.markdown("Educational prototype demonstrating model scoring for a hypothetical patient at discharge.")
    
    defaults = metrics_data.get("default_patient", {})
    if not defaults:
        st.error("Default patient schema not found in `models/default_patient.json`.")
        st.stop()
        
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Clinical Research Prototype Disclaimer:</strong><br>
        This tool is strictly an educational research demonstration based on historical EHR data (1999–2008). It is <strong>NOT a medical device</strong> and must NOT be used to make clinical decisions or diagnose patients.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("patient_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Patient Demographics")
            age = st.slider("Patient Age (Years)", 10, 95, int(defaults.get("age", 65)), 5)
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"], index=0)
            race = st.selectbox("Race / Ethnicity", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"], index=0)
            a1c = st.selectbox("Last HbA1c Test Result", ["None", "Norm", ">7", ">8"], index=0)
            glu = st.selectbox("Last Serum Glucose Test Result", ["None", "Norm", ">200", ">300"], index=0)
            on_diabetes_meds = st.radio("Prescribed Diabetes Medication?", ["Yes", "No"], index=0)

        with col2:
            st.markdown("#### Current Encounter Details")
            adm_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn", "Trauma Center", "Unknown"], index=0)
            adm_source = st.selectbox("Admission Source", ["Emergency Room", "Referral/Clinic", "Transfer", "Other/Unknown"], index=0)
            time_in_hospital = st.slider("Length of Inpatient Stay (Days)", 1, 14, int(defaults.get("time_in_hospital", 4)))
            n_diagnoses = st.slider("Total Number of Diagnoses", 1, 16, int(defaults.get("number_diagnoses", 7)))
            n_lab_proc = st.slider("Lab Procedures Performed", 0, 120, int(defaults.get("num_lab_procedures", 44)))
            n_procedures = st.slider("Other Procedures Performed", 0, 6, int(defaults.get("num_procedures", 1)))

        with col3:
            st.markdown("#### Medical History & Discharge")
            discharge = st.selectbox("Discharge Destination", ["Home", "Another facility", "Transferred", "Left AMA/Other", "Other/Unknown", "Expired/Hospice"], index=0)
            n_outpatient = st.slider("Outpatient Visits (Prior 12 Months)", 0, 25, int(defaults.get("number_outpatient", 0)))
            n_emergency = st.slider("Emergency Visits (Prior 12 Months)", 0, 25, int(defaults.get("number_emergency", 0)))
            n_inpatient = st.slider("Inpatient Admissions (Prior 12 Months)", 0, 15, int(defaults.get("number_inpatient", 0)))
            n_medications = st.slider("Total Medications Prescribed", 0, 80, int(defaults.get("num_medications", 15)))
            primary_diag = st.selectbox("Primary Diagnosis Category", ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Genitourinary", "Injury", "Musculoskeletal", "Neoplasms", "Other", "Unknown"], index=0)

        submitted = st.form_submit_button("🔮 Compute Calibrated Readmission Risk", type="primary", use_container_width=True)

    if submitted:
        if model_pipeline is None:
            st.error("Model pipeline could not be loaded. Please ensure `models/best_model.joblib` exists.")
        else:
            A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
            GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
            
            patient_dict = dict(defaults)
            patient_dict.update({
                "age": float(age),
                "gender": gender,
                "race": race,
                "admission_type": adm_type,
                "admission_source": adm_source,
                "discharge_disposition": discharge,
                "time_in_hospital": float(time_in_hospital),
                "number_diagnoses": float(n_diagnoses),
                "num_lab_procedures": float(n_lab_proc),
                "num_procedures": float(n_procedures),
                "number_outpatient": float(n_outpatient),
                "number_emergency": float(n_emergency),
                "number_inpatient": float(n_inpatient),
                "num_medications": float(n_medications),
                "primary_diag": primary_diag,
                "service_utilization": float(n_outpatient + n_emergency + n_inpatient),
                "lab_monitoring_score": float(A1C_MAP[a1c] + GLU_MAP[glu]),
                "n_active_medications": float(defaults.get("n_active_medications", 1.0)),
            })
            
            X_df = pd.DataFrame([patient_dict])[list(defaults.keys())]
            
            try:
                raw_proba = float(model_pipeline.predict_proba(X_df)[0, 1])
                base_rate = 0.1116
                optimal_threshold = 0.1442
                
                res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])
                
                with res_col1:
                    st.metric("Predicted 30-Day Readmission Risk", f"{raw_proba:.1%}")
                    st.progress(min(raw_proba, 1.0))
                    
                    if raw_proba >= optimal_threshold:
                        risk_label = "HIGH RISK (Flagged for Post-Discharge Support)"
                        risk_color = "#DC2626"
                        risk_desc = f"Probability ({raw_proba:.1%}) exceeds the validation-selected threshold ({optimal_threshold:.1%}). Patient may benefit from transitional care follow-up."
                    else:
                        risk_label = "STANDARD RISK (Routine Follow-Up)"
                        risk_color = "#16A34A"
                        risk_desc = f"Probability ({raw_proba:.1%}) is below the validation threshold ({optimal_threshold:.1%}). Routine discharge planning recommended."
                        
                    st.markdown(f"""
                    <div style="background-color: {risk_color}15; border-left: 4px solid {risk_color}; padding: 12px 16px; border-radius: 6px; margin-top: 10px;">
                        <strong style="color: {risk_color}; font-size: 1.05rem;">{risk_label}</strong><br>
                        <span style="color: #334155; font-size: 0.9rem;">{risk_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.metric("Population Base Rate", f"{base_rate:.1%}")
                    rel_risk = raw_proba / base_rate
                    st.metric("Relative Risk Ratio", f"{rel_risk:.2f}×")
                    
                with res_col3:
                    st.metric("Operating Cutoff Threshold", f"{optimal_threshold:.1%}")
                    st.caption("Derived on held-out validation set to optimize F1 score without test-set leakage.")
                    
                with st.expander("Inspect Transmitted Feature Vector"):
                    st.dataframe(X_df.T.rename(columns={0: "Input Value"}), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction execution error: {e}")


# ================================================================== #
# PAGE 7: RESEARCH PAPER COMPARISON
# ================================================================== #
elif page == "7. Research Paper Comparison":
    st.subheader("7. Reference Paper Comparison & Critical Literature Audit")
    st.markdown("Critical comparison against the published study: **Emi-Johnson & Nkrumah (2025), Cureus 17(4), e82437**.")
    
    st.markdown("""
    <div class="alert-box">
        <strong>Methodological Non-Equivalence Disclaimer:</strong><br>
        Workspace evaluations and the published study results do not represent a strict head-to-head replication. Differences in data cleaning, cohort exclusion timing, feature engineering, five-fold cross-validation pipelines, and decision thresholds preclude claiming definitive algorithmic superiority. Rather, workspace experiments provide a transparent, reproducible benchmark under documented methodological controls.
    </div>
    """, unsafe_allow_html=True)

    paper_comp = pd.DataFrame([
        {
            "Model Architecture": "XGBoost",
            "Paper Reported ROC-AUC": 0.667,
            "Workspace Test ROC-AUC": 0.6709,
            "Paper Reported Accuracy": "0.89",
            "Workspace Accuracy": "0.6570",
            "Paper Class 1 Recall": 0.017,
            "Workspace Class 1 Recall": 0.5764,
            "Paper Class 1 F1": 0.032,
            "Workspace Class 1 F1": 0.2727
        },
        {
            "Model Architecture": "Logistic Regression",
            "Paper Reported ROC-AUC": 0.642,
            "Workspace Test ROC-AUC": 0.6684,
            "Paper Reported Accuracy": "0.89",
            "Workspace Accuracy": "0.6387",
            "Paper Class 1 Recall": 0.016,
            "Workspace Class 1 Recall": 0.5997,
            "Paper Class 1 F1": 0.030,
            "Workspace Class 1 F1": 0.2703
        },
        {
            "Model Architecture": "Random Forest",
            "Paper Reported ROC-AUC": 0.630,
            "Workspace Test ROC-AUC": 0.6777,
            "Paper Reported Accuracy": "0.89",
            "Workspace Accuracy": "0.6780",
            "Paper Class 1 Recall": 0.002,
            "Workspace Class 1 Recall": 0.5557,
            "Paper Class 1 F1": 0.003,
            "Workspace Class 1 F1": 0.2781
        },
        {
            "Model Architecture": "Deep Neural Net (DNN)",
            "Paper Reported ROC-AUC": 0.579,
            "Workspace Test ROC-AUC": 0.6750,
            "Paper Reported Accuracy": "0.84",
            "Workspace Accuracy": "0.6009",
            "Paper Class 1 Recall": 0.143,
            "Workspace Class 1 Recall": 0.6609,
            "Paper Class 1 F1": 0.161,
            "Workspace Class 1 F1": 0.2698
        },
    ])
    
    st.dataframe(paper_comp, hide_index=True, use_container_width=True)
    
    st.markdown("### Critical Contradictions Identified in the Reference Paper")
    st.markdown("""
    1. **Ambiguous Headline F1 Averaging:**
       - In Table 1 and the abstract, the authors report an "F1-score" of **0.89** for XGBoost, claiming in the discussion that it achieved an F1 of 0.89 for the positive class.
       - However, in Table 2 ("Class-wise precision, recall, and F1-score"), the paper explicitly reports the positive-class (<30-day readmission) F1 score as **0.032** (precision 43.7%, recall 1.6%).
       - *Methodological Correction:* The averaging definition for the 0.89 headline figure is unspecified in the paper, but cannot represent the positive class given the reported 1.7% recall. Comparing our models against 0.89 is invalid because the true positive-class F1 reported in Table 2 is 0.032.
       
    2. **Internal Arithmetic Inconsistency:**
       - Table 3 lists XGBoost errors: $\text{FP} = 1,172, \text{FN} = 2,244$.
       - In a test set of 2,285 positives, $\text{TP} = 2,285 - 2,244 = 41$.
       - True precision is $\text{TP} / (\text{TP} + \text{FP}) = 41 / (41 + 1,172) = 0.0338$ (3.38%).
       - Yet Table 2 reports Class 1 Precision as **0.437** (43.7%), representing a 13-fold internal mathematical contradiction.
       
    3. **Cohort Partitioning & Exclusion Timing:**
       - The paper states under Methods that hospice and expired discharges were excluded.
       - However, the paper reports a test set of **20,354 encounters** (exactly 20.00% of the raw 101,766 dataset). While this exact match to 20% of the raw data raises a significant methodological question regarding whether and how exclusions were applied, it does not independently prove the exact implementation sequence taken by the authors.
    """)


# ================================================================== #
# PAGE 8: LIMITATIONS & CONCLUSION
# ================================================================== #
elif page == "8. Limitations & Conclusion":
    st.subheader("8. Scientific Limitations & Academic Conclusions")
    
    st.markdown("""
    ### Honest Scientific Limitations
    1. **Historical Administrative Dataset (1999–2008):**
       - The data reflects clinical practice patterns, ICD-9 coding, and pharmacotherapy from 15–25 years ago. Temporal drift and changes in diabetes care require recalibration on contemporary EHR data.
    2. **Absence of Social Determinants of Health (SDoH):**
       - Tabular administrative billing records lack crucial real-world drivers of readmission: health literacy, caregiver availability, medication adherence, insurance stability, and transportation access.
    3. **Discharge-Time Operational Scope:**
       - Key features (length of stay, inpatient procedures, discharge destination) require the hospital stay to be completed; the model cannot be deployed as an admission-time triage screener.
    4. **Split Protocol Isolation:**
       - While both encounter-level (`A_LR`, ROC-AUC 0.6682) and patient-grouped (`A_RF`, ROC-AUC 0.6753) protocols were executed, they evaluated different model families; the isolated effect of patient clustering was not tested holding model architecture constant.
    5. **Unexecuted Improvement Pipelines:**
       - Advanced improvement experiments (LightGBM ablation B0, CatBoost B2, and soft-voting ensemble B4) were fully implemented in `scripts/run_improved.py`, but not executed due to computational deadline constraints.
    6. **Requirement for Prospective Clinical Validation:**
       - Calibrated risk models are research prototypes. Before clinical implementation, prospective clinical utility and health equity audits across racial and age demographics must be conducted.
    """)
    
    st.divider()
    
    st.markdown("""
    ### Final Academic Conclusion
    This project established an audited, reproducible, and methodologically sound machine learning research pipeline for diabetic readmission prediction. Tuned tree ensembles and deep MLPs achieve an ROC-AUC of **~0.678** and PR-AUC of **~0.217** on the benchmark UCI Diabetes dataset.
    
    Crucially, we resolved major ambiguities in the published literature, documented the vital distinction between encounter-level and patient-grouped validation, demonstrated the necessity of validation-based threshold realignment for calibrated models, and verified clinical feature importance using SHAP analysis.
    """)

# ------------------------------------------------------------------ #
# Professional Footer (On All Pages)
# ------------------------------------------------------------------ #
st.markdown("""
<div class="footer">
    <strong>Academic Research Prototype</strong> | UCI Diabetes 130-US Hospitals Dataset | Reference: Cureus 17(4), e82437 | <em>Not for Clinical Use</em>
</div>
""", unsafe_allow_html=True)
