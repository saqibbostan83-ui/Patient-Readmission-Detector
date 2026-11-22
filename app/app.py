
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Readmission | Research Dashboard",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "replication"

ENCOUNTER_DIR = RESULTS / "encounter"
PATIENT_DIR = RESULTS / "patient"

PAPER_AUC = {
    "LR": 0.642,
    "RF": 0.630,
    "XGB": 0.667,
    "DNN": 0.579,
}

MODEL_NAMES = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "DNN": "Deep Neural Network",
}

COLORS = {
    "background": "#07111F",
    "surface": "#101E30",
    "surface2": "#14263B",
    "border": "#243B53",
    "text": "#E8F0FA",
    "muted": "#9AAFC5",
    "teal": "#22D3B5",
    "cyan": "#38BDF8",
    "purple": "#A78BFA",
    "orange": "#FBBF24",
    "red": "#FB7185",
}

PLOT_TEMPLATE = "plotly_dark"


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111F;
        --surface: #101E30;
        --surface2: #14263B;
        --border: #243B53;
        --text: #E8F0FA;
        --muted: #9AAFC5;
        --teal: #22D3B5;
        --cyan: #38BDF8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(34,211,181,.09), transparent 27%),
            radial-gradient(circle at 90% 5%, rgba(56,189,248,.08), transparent 25%),
            #07111F;
        color: var(--text);
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.5px;
    }

    p, li, label {
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0B1727;
        border-right: 1px solid #243B53;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.2rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #12243A, #0D1A2B);
        border: 1px solid #243B53;
        padding: 17px 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #9AAFC5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #E8F0FA !important;
        font-family: 'Manrope', sans-serif;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #29445D;
        background:
            radial-gradient(circle at 90% 10%, rgba(56,189,248,.16), transparent 30%),
            linear-gradient(120deg, #10243A, #0B192A);
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #22D3B5;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: 2.2rem;
        line-height: 1.2;
        margin: 12px 0;
    }

    .hero p {
        color: #B5C6D8;
        max-width: 850px;
        font-size: 1rem;
    }

    .section-kicker {
        color: #22D3B5;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: .72rem;
        margin-bottom: 4px;
    }

    .panel {
        background: linear-gradient(145deg, #12243A, #0D1A2B);
        border: 1px solid #243B53;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        color: #E8F0FA;
        margin-bottom: 8px;
    }

    .muted {
        color: #9AAFC5;
    }

    .good {
        color: #22D3B5;
        font-weight: 700;
    }

    .warning {
        color: #FBBF24;
        font-weight: 700;
    }

    .bad {
        color: #FB7185;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #243B53;
        border-radius: 12px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"] {
        color: #B5C6D8;
    }

    .stTabs [aria-selected="true"] {
        color: #22D3B5 !important;
    }

    hr {
        border-color: #243B53;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path: str):
    file = Path(path)
    if not file.exists():
        return None
    return pd.read_csv(file)


@st.cache_data
def load_json(path: str):
    file = Path(path)
    if not file.exists():
        return None
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_npz(path: str):
    file = Path(path)
    if not file.exists():
        return None
    return dict(np.load(file, allow_pickle=True))


encounter_df = load_csv(str(ENCOUNTER_DIR / "leaderboard_A.csv"))
patient_df = load_csv(str(PATIENT_DIR / "leaderboard_A.csv"))

encounter_summary = load_json(str(ENCOUNTER_DIR / "split_summary.json"))
patient_summary = load_json(str(PATIENT_DIR / "split_summary.json"))

encounter_probas = load_npz(str(ENCOUNTER_DIR / "test_probas.npz"))
patient_probas = load_npz(str(PATIENT_DIR / "test_probas.npz"))


# ============================================================
# HELPERS
# ============================================================

def metric_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_metric(df, model, candidates):
    if df is None or "experiment" not in df.columns:
        return None

    row = df[df["experiment"].astype(str).str.contains(
        model, case=False, na=False
    )]

    if row.empty:
        return None

    col = metric_column(df, candidates)
    if col is None:
        return None

    value = pd.to_numeric(row.iloc[0][col], errors="coerce")
    return None if pd.isna(value) else float(value)


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def display_metric_cards(df):
    cols = st.columns(4)

    for col, model in zip(cols, ["LR", "RF", "XGB", "DNN"]):
        auc = get_metric(df, model, ["test_roc_auc"])
        pr = get_metric(df, model, ["test_pr_auc"])
        with col:
            st.metric(
                MODEL_NAMES[model],
                fmt(auc),
                help=f"Test PR-AUC: {fmt(pr)}",
            )


def chart_layout(fig, height=380):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE8F5", family="DM Sans"),
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(gridcolor="#243B53", zerolinecolor="#243B53")
    fig.update_yaxes(gridcolor="#243B53", zerolinecolor="#243B53")
    return fig


def section_header(kicker, title, description):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted">{description}</p>', unsafe_allow_html=True)


def show_missing_data_message():
    st.error(
        "Required replication results were not found. "
        "Check that the results/replication/encounter and "
        "results/replication/patient folders exist."
    )


def get_summary_value(summary, *keys):
    if not isinstance(summary, dict):
        return None

    for key in keys:
        if key in summary:
            return summary[key]

    return None


def leaderboard_table(df):
    if df is None:
        st.warning("Leaderboard file is unavailable.")
        return

    display = df.copy()

    if "experiment" in display.columns:
        display["Model"] = display["experiment"].map(
            lambda x: MODEL_NAMES.get(str(x).replace("A_", ""), str(x))
        )

    preferred = [
        "Model",
        "cv_roc_auc",
        "cv_roc_auc_std",
        "test_roc_auc",
        "test_roc_auc_ci_low",
        "test_roc_auc_ci_high",
        "test_pr_auc",
        "brier",
    ]

    columns = [c for c in preferred if c in display.columns]

    if columns:
        display = display[columns]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:12px 5px 20px 5px">
        <div style="color:#22D3B5;font-size:.75rem;font-weight:800;letter-spacing:2px">
            RESEARCH LAB
        </div>
        <h2 style="margin:5px 0 0 0;color:#E8F0FA">
            READMISSION<br>ANALYTICS
        </h2>
        <p style="color:#9AAFC5;font-size:.8rem">
            Explainable machine learning research
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "Research Overview",
    "Dataset & Cohort",
    "Model Benchmark",
    "Validation & Leakage Audit",
    "Explainable AI",
    "Prediction Lab",
    "Paper Comparison",
    "Conclusions & Limitations",
]

page = st.sidebar.radio("RESEARCH SECTIONS", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("**Study information**")
st.sidebar.caption("Dataset: UCI Diabetes 130-US Hospitals")
st.sidebar.caption("Dataset ID: 296")
st.sidebar.caption("Outcome: 30-day readmission")
st.sidebar.caption("Reference: Cureus, 2025")

st.sidebar.markdown("---")
st.sidebar.caption("Academic research dashboard")
st.sidebar.caption("Not intended for clinical decision-making.")


# ============================================================
# PAGE 1 — RESEARCH OVERVIEW
# ============================================================

if page == "Research Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Explainable machine learning · Clinical research</div>
            <h1>Predicting 30-Day Hospital Readmission</h1>
            <p>
                An academic research dashboard evaluating machine learning
                models using the UCI Diabetes 130-US Hospitals dataset.
                The study compares encounter-level and patient-level
                validation protocols to examine generalization and
                patient-overlap concerns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Research at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Source encounters", "101,766")
    c2.metric("Feature count", "261")
    c3.metric("Model families", "4")
    c4.metric("Validation protocols", "2")

    st.markdown("---")
    st.markdown("### Headline results")

    if patient_df is None or encounter_df is None:
        show_missing_data_message()
    else:
        patient_best = get_metric(patient_df, "XGB", ["test_roc_auc"])
        patient_pr = get_metric(patient_df, "XGB", ["test_pr_auc"])
        encounter_best = get_metric(encounter_df, "RF", ["test_roc_auc"])

        a, b, c = st.columns(3)
        a.metric("Best patient-level ROC-AUC", fmt(patient_best))
        b.metric("Best patient-level PR-AUC", fmt(patient_pr))
        c.metric("Best encounter-level ROC-AUC", fmt(encounter_best))

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Primary research interpretation</div>
                The patient-level protocol prevents the same patient from
                appearing in both training and test sets. It therefore
                provides the more defensible estimate of generalization
                to previously unseen patients.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Study objectives")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            **01 — Model evaluation**

            Compare Logistic Regression, Random Forest, XGBoost,
            and Deep Neural Network models using ROC-AUC,
            PR-AUC, and Brier score.

            **02 — Validation strategy**

            Examine the difference between encounter-level
            and patient-level evaluation.
            """
        )

    with right:
        st.markdown(
            """
            **03 — Explainability**

            Present available SHAP outputs to support
            model interpretation.

            **04 — Research comparison**

            Compare reproduced results with the
            reference paper's reported ROC-AUC values.
            """
        )


# ============================================================
# PAGE 2 — DATASET & COHORT
# ============================================================

elif page == "Dataset & Cohort":

    section_header(
        "01 · DATA",
        "Dataset & Cohort",
        "Dataset characteristics and final train/test split information.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Original encounters", "101,766")
    c2.metric("Input features", "261")
    c3.metric("Prediction horizon", "30 days")

    st.markdown("### Dataset description")

    st.markdown(
        """
        The UCI Diabetes 130-US Hospitals dataset contains hospital
        encounters associated with diabetes patients. It includes
        demographic characteristics, admission information,
        laboratory-related variables, medication indicators,
        and discharge information.

        The prediction target is whether a patient is readmitted
        within 30 days.
        """
    )

    st.markdown("---")
    st.markdown("### Final protocol split summaries")

    if encounter_summary is None or patient_summary is None:
        show_missing_data_message()
    else:
        split_rows = []

        for name, summary in [
            ("Encounter-level", encounter_summary),
            ("Patient-level", patient_summary),
        ]:
            split_rows.append({
                "Protocol": name,
                "Train encounters": get_summary_value(
                    summary, "n_train", "train_size", "train_encounters"
                ),
                "Test encounters": get_summary_value(
                    summary, "n_test", "test_size", "test_encounters"
                ),
                "Train patients": get_summary_value(
                    summary, "n_train_patients", "train_patients"
                ),
                "Test patients": get_summary_value(
                    summary, "n_test_patients", "test_patients"
                ),
                "Patient overlap": get_summary_value(
                    summary, "patient_overlap", "n_patient_overlap"
                ),
            })

        st.dataframe(
            pd.DataFrame(split_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Cohort interpretation</div>
                The encounter-level protocol allows patients to contribute
                encounters to both training and test partitions.
                The patient-level protocol uses grouped splitting to
                prevent this overlap.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Important cohort definition")

    st.info(
        "The replication excludes encounters discharged to hospice or "
        "expired and excludes age bins [0–10) and [10–20). "
        "This approximates an adult cohort but also excludes individuals "
        "aged 18–19."
    )


# ============================================================
# PAGE 3 — MODEL BENCHMARK
# ============================================================

elif page == "Model Benchmark":

    section_header(
        "02 · PERFORMANCE",
        "Model Performance Benchmark",
        "Compare final model performance under the two validation protocols.",
    )

    protocol = st.segmented_control(
        "Select validation protocol",
        options=["Patient-level", "Encounter-level"],
        default="Patient-level",
    )

    if protocol == "Patient-level":
        df = patient_df
        summary = patient_summary
    else:
        df = encounter_df
        summary = encounter_summary

    if df is None:
        show_missing_data_message()
    else:
        st.markdown("### Test ROC-AUC")

        display_metric_cards(df)

        st.markdown("---")
        st.markdown("### Detailed leaderboard")

        leaderboard_table(df)

        st.markdown("---")

        roc_col = metric_column(df, ["test_roc_auc"])
        pr_col = metric_column(df, ["test_pr_auc"])
        brier_col = metric_column(df, ["brier"])

        if roc_col:
            chart_data = df.copy()
            chart_data["Model"] = chart_data["experiment"].map(
                lambda x: MODEL_NAMES.get(
                    str(x).replace("A_", ""), str(x)
                )
            )

            fig = px.bar(
                chart_data,
                x="Model",
                y=roc_col,
                color="Model",
                title="Test ROC-AUC by model",
                text_auto=".4f",
            )
            fig.update_yaxes(range=[0.5, 1.0])
            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )

        col1, col2 = st.columns(2)

        if pr_col:
            with col1:
                fig = px.bar(
                    df,
                    x="experiment",
                    y=pr_col,
                    title="Test PR-AUC",
                    color="experiment",
                    text_auto=".4f",
                )
                st.plotly_chart(
                    chart_layout(fig, 330),
                    use_container_width=True,
                )

        if brier_col:
            with col2:
                fig = px.bar(
                    df,
                    x="experiment",
                    y=brier_col,
                    title="Brier score (lower is better)",
                    color="experiment",
                    text_auto=".4f",
                )
                st.plotly_chart(
                    chart_layout(fig, 330),
                    use_container_width=True,
                )

        st.caption(
            "ROC-AUC and PR-AUC are higher-is-better metrics. "
            "Brier score is a lower-is-better measure of probabilistic error."
        )


# ============================================================
# PAGE 4 — VALIDATION & LEAKAGE AUDIT
# ============================================================

elif page == "Validation & Leakage Audit":

    section_header(
        "03 · VALIDATION",
        "Validation & Leakage Audit",
        "Examine the validation protocols and patient overlap.",
    )

    if encounter_summary is None or patient_summary is None:
        show_missing_data_message()
    else:
        encounter_overlap = get_summary_value(
            encounter_summary,
            "patient_overlap",
            "n_patient_overlap",
        )

        patient_overlap = get_summary_value(
            patient_summary,
            "patient_overlap",
            "n_patient_overlap",
        )

        encounter_pct = get_summary_value(
            encounter_summary,
            "patient_overlap_pct",
            "overlap_percentage",
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Encounter-level overlap",
            str(encounter_overlap) if encounter_overlap is not None else "N/A",
        )

        c2.metric(
            "Patient-level overlap",
            str(patient_overlap) if patient_overlap is not None else "N/A",
        )

        c3.metric(
            "Encounter overlap (%)",
            fmt(float(encounter_pct), 2)
            if encounter_pct is not None
            else "41.74%",
        )

        st.markdown("---")
        st.markdown("### Why patient overlap matters")

        st.markdown(
            """
            In encounter-level splitting, different encounters from the
            same patient may appear in both training and testing data.

            This means the model may encounter patient-specific patterns
            during training that are also present in the test set.

            Such overlap can make the test evaluation less representative
            of performance on entirely unseen patients.
            """
        )

        st.markdown("### Protocol comparison")

        comparison = pd.DataFrame({
            "Property": [
                "Split unit",
                "Patient overlap",
                "Generalization target",
                "Interpretation",
            ],
            "Encounter-level": [
                "Hospital encounter",
                "Present",
                "Future encounters, potentially including known patients",
                "Potentially optimistic for unseen-patient generalization",
            ],
            "Patient-level": [
                "Unique patient",
                "Zero",
                "Previously unseen patients",
                "More defensible unseen-patient estimate",
            ],
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "The final patient-level split has zero patient overlap."
        )

        st.warning(
            "Patient-level validation is the primary headline protocol. "
            "The encounter-level results are retained as a comparison, "
            "not as the preferred unseen-patient estimate."
        )


# ============================================================
# PAGE 5 — EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":

    section_header(
        "04 · INTERPRETABILITY",
        "Explainable AI & SHAP",
        "Review available explainability outputs from the research workflow.",
    )

    st.markdown("### SHAP interpretation")

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) is a feature-attribution
        framework that estimates how individual input features
        contribute to a model's predictions.

        A SHAP summary plot can provide two complementary views:

        - Feature importance: the overall magnitude of feature contributions.
        - Feature effect: whether feature values tend to push predictions
          higher or lower.
        """
    )

    shap_candidates = [
        ROOT / "results" / "shap",
        ROOT / "results" / "explainability",
        ROOT / "results",
    ]

    image_files = []

    for folder in shap_candidates:
        if folder.exists():
            for extension in ["*.png", "*.jpg", "*.jpeg"]:
                image_files.extend(folder.rglob(extension))

    # Avoid displaying duplicate paths.
    image_files = sorted(set(image_files))

    if image_files:
        st.markdown("### Available explainability figures")

        for image_path in image_files[:12]:
            st.markdown(f"**{image_path.name}**")
            st.image(str(image_path), use_container_width=True)
    else:
        st.warning(
            "No saved SHAP image files were found in the expected "
            "results folders. This page does not fabricate feature "
            "importance values."
        )

    st.markdown("---")
    st.markdown("### Interpretation cautions")

    st.info(
        "Feature attribution describes model behavior, not causality. "
        "A high SHAP value does not establish that changing a feature "
        "would change a patient's clinical outcome."
    )


# ============================================================
# PAGE 6 — PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    section_header(
        "05 · INTERACTIVE ANALYSIS",
        "Prediction Lab",
        "Experimental prediction interface and model-artifact verification.",
    )

    st.warning(
        "The interactive prediction function is temporarily disabled."
    )

    st.markdown(
        """
        The legacy model artifact must be verified against the final
        patient-level XGBoost experiment before it can be used here.

        A valid prediction interface requires confirmation of:

        1. The exact model used in the final experiment.
        2. The feature names and feature order.
        3. The preprocessing pipeline.
        4. The training protocol and target definition.
        5. Probability calibration and threshold selection.
        """
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Research integrity safeguard</div>
            This dashboard will not present predictions from an
            unverified legacy model as if they came from the final
            patient-level experiment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "This research project is not a medical diagnostic system "
        "and must not be used to make clinical decisions."
    )


# ============================================================
# PAGE 7 — PAPER COMPARISON
# ============================================================

elif page == "Paper Comparison":

    section_header(
        "06 · REPRODUCIBILITY",
        "Research Paper Comparison",
        "Compare published ROC-AUC values with the final experimental results.",
    )

    rows = []

    for model, name in MODEL_NAMES.items():
        rows.append({
            "Model": name,
            "Published ROC-AUC": PAPER_AUC[model],
            "Encounter-level ROC-AUC": get_metric(
                encounter_df, model, ["test_roc_auc"]
            ),
            "Patient-level ROC-AUC": get_metric(
                patient_df, model, ["test_roc_auc"]
            ),
        })

    comparison = pd.DataFrame(rows)

    st.markdown("### ROC-AUC comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    plot_df = comparison.melt(
        id_vars="Model",
        value_vars=[
            "Published ROC-AUC",
            "Encounter-level ROC-AUC",
            "Patient-level ROC-AUC",
        ],
        var_name="Experiment",
        value_name="ROC-AUC",
    )

    fig = px.bar(
        plot_df,
        x="Model",
        y="ROC-AUC",
        color="Experiment",
        barmode="group",
        title="Published versus reproduced ROC-AUC",
        text_auto=".3f",
    )

    fig.update_yaxes(range=[0.5, 0.8])

    st.plotly_chart(
        chart_layout(fig, 430),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Reference study reporting observations")

    st.warning(
        "The reference paper contains internal numerical inconsistencies "
        "between some reported classification metrics and its confusion "
        "matrix. These are treated as reporting discrepancies, not as "
        "evidence of misconduct."
    )

    st.markdown(
        """
        **Examples identified during the research review:**

        - The paper's discussion reports an XGBoost positive-class F1
          score of 0.89, while its results table reports 0.032.
        - The confusion matrix implies a positive-class precision of
          approximately 0.034, which does not match the separately
          reported precision value.
        - The reported test-set size is 20,354, equal to 20% of the
          original 101,766 encounters, despite the stated exclusions.
        """
    )

    st.caption(
        "Reference: Emi-Johnson & Nkrumah (2025), Cureus, 17(4), e82437."
    )


# ============================================================
# PAGE 8 — CONCLUSIONS & LIMITATIONS
# ============================================================

elif page == "Conclusions & Limitations":

    section_header(
        "07 · DISCUSSION",
        "Conclusions & Limitations",
        "Research findings, interpretation, and methodological boundaries.",
    )

    st.markdown("### Main findings")

    st.markdown(
        """
        **1. Patient-level validation is the primary result.**

        Grouped splitting prevents the same patient from appearing in
        training and testing partitions, providing a more defensible
        estimate for generalization to unseen patients.

        **2. XGBoost achieved the highest patient-level test ROC-AUC.**

        Its test ROC-AUC was approximately 0.6754, and its test PR-AUC
        was approximately 0.2271.

        **3. Random Forest achieved the lowest patient-level Brier score.**

        Its Brier score was approximately 0.1705. This is a separate
        evaluation dimension from discrimination and should not be
        interpreted as proof of overall model superiority.

        **4. Encounter-level results should be interpreted cautiously.**

        The encounter-level split contains patient overlap, so it is
        less suitable as the primary estimate for previously unseen
        patient generalization.
        """
    )

    st.markdown("---")
    st.markdown("### Patient-level final results")

    if patient_df is not None:
        leaderboard_table(patient_df)
    else:
        st.warning("Patient-level leaderboard unavailable.")

    st.markdown("---")
    st.markdown("### Study limitations")

    limitations = [
        "The dataset is historical and may not represent current clinical practice.",
        "The analysis is based on a specific UCI dataset and its recorded variables.",
        "Predictive associations do not establish causal relationships.",
        "Performance metrics do not establish clinical utility or deployment readiness.",
        "The positive outcome class is relatively uncommon, making PR-AUC important.",
        "A full-cohort sensitivity experiment was not completed.",
        "External validation on an independent hospital dataset was not completed.",
        "The interactive prediction interface remains disabled pending model-artifact verification.",
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("### Final research conclusion")

    st.markdown(
        """
        The experiments demonstrate that validation design matters
        when repeated hospital encounters belong to the same patient.

        The patient-level protocol provides the more defensible
        headline estimate for unseen-patient generalization.
        Although XGBoost achieved the highest patient-level test
        ROC-AUC and PR-AUC in these experiments, the results do not
        establish statistically significant superiority or clinical
        readiness.

        Further work should include independent external validation,
        additional calibration assessment, and prospective evaluation
        before any clinical application is considered.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:15px;color:#8298AE;font-size:.8rem">
        DIABETES 30-DAY READMISSION PREDICTION · ACADEMIC RESEARCH DASHBOARD
        <br>
        UCI Diabetes 130-US Hospitals Dataset · Research use only
    </div>
    """,
    unsafe_allow_html=True,
)
