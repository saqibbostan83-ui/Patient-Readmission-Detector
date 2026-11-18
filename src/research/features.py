"""Feature sets.

FS-R  (replication) : the original UCI variables, as described in the paper
                      (one-hot categoricals, mean-imputed + standardised numerics).
FS-E  (engineered)  : FS-R + blocks of clinically motivated features, each
                      block evaluated by ablation (scripts/run_improved.py).

All engineered features are computed row-wise from information recorded
during the index encounter or the prior year (available at discharge), so no
statistic is learned from other rows - nothing here can leak test labels.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]  # examide & citoglipton are constant ('No') in all 101,766 rows -> dropped
ORAL_MEDS = [m for m in MEDS if m != "insulin"]

R_NUM = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
         "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
R_CAT = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
         "admission_source_id", "payer_code", "medical_specialty",
         "diag_1_grp", "diag_2_grp", "diag_3_grp", "max_glu_serum", "A1Cresult",
         "change", "diabetesMed"] + MEDS

AGE_NUM = {f"[{10*i}-{10*(i+1)})": 10 * i + 5 for i in range(10)}


# --------------------------------------------------------------------------- #
# ICD-9 groupings
# --------------------------------------------------------------------------- #
def icd9_strack(code) -> str:
    """9 disease groups of Strack et al. (2014), Table 2 of the dataset paper."""
    if pd.isna(code):
        return "Missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        c = float(s)
    except ValueError:
        return "Other"
    if 390 <= c <= 459 or int(c) == 785:
        return "Circulatory"
    if 460 <= c <= 519 or int(c) == 786:
        return "Respiratory"
    if 520 <= c <= 579 or int(c) == 787:
        return "Digestive"
    if int(c) == 250:
        return "Diabetes"
    if 800 <= c <= 999:
        return "Injury"
    if 710 <= c <= 739:
        return "Musculoskeletal"
    if 580 <= c <= 629 or int(c) == 788:
        return "Genitourinary"
    if 140 <= c <= 239:
        return "Neoplasms"
    return "Other"


def _icd3(code) -> str:
    if pd.isna(code):
        return "Missing"
    return str(code).split(".")[0]


def _num(code) -> float:
    try:
        return float(code)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- #
def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        out[f"{c}_grp"] = out[c].map(icd9_strack)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[c] = out[c].astype(str)
    return out


def _group_discharge(i: int) -> str:
    if i == 1:
        return "home"
    if i == 6 or i == 8:
        return "home_health"
    if i in (3, 4, 15, 22, 23, 24, 27, 28, 5, 2, 10, 9, 12, 16, 17):
        return "facility_or_transfer"
    if i == 7:
        return "left_ama"
    return "unknown_other"


def _group_source(i: int) -> str:
    if i == 7:
        return "emergency"
    if i in (1, 2, 3):
        return "referral"
    if i in (4, 5, 6, 10, 18, 19, 22, 25, 26):
        return "transfer"
    return "other_unknown"


def _group_admtype(i: int) -> str:
    return {1: "emergency", 2: "urgent", 3: "elective", 7: "emergency"}.get(i, "other_unknown")


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features (FS-E blocks)."""
    out = df.copy()
    # ---- utilisation block ------------------------------------------------
    out["service_utilization"] = out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        out[f"log1p_{c}"] = np.log1p(out[c])
    out["any_prior_inpatient"] = (out["number_inpatient"] > 0).astype(int)
    out["any_prior_emergency"] = (out["number_emergency"] > 0).astype(int)
    # ---- medication block -------------------------------------------------
    med = out[MEDS]
    out["n_meds_active"] = (med != "No").sum(axis=1)
    out["n_meds_up"] = (med == "Up").sum(axis=1)
    out["n_meds_down"] = (med == "Down").sum(axis=1)
    out["n_meds_changed"] = out["n_meds_up"] + out["n_meds_down"]
    out["insulin_active"] = (out["insulin"] != "No").astype(int)
    out["insulin_changed"] = out["insulin"].isin(["Up", "Down"]).astype(int)
    out["n_oral_active"] = (out[ORAL_MEDS] != "No").sum(axis=1)
    out["meds_per_day"] = out["num_medications"] / out["time_in_hospital"]
    # ---- labs block --------------------------------------------------------
    out["a1c_tested"] = (out["A1Cresult"] != "None").astype(int)
    out["a1c_high"] = out["A1Cresult"].isin([">7", ">8"]).astype(int)
    out["a1c_high_and_med_changed"] = out["a1c_high"] * (out["change"] == "Ch").astype(int)
    out["glu_tested"] = (out["max_glu_serum"] != "None").astype(int)
    out["labs_per_day"] = out["num_lab_procedures"] / out["time_in_hospital"]
    # ---- diagnosis block --------------------------------------------------
    dnum = pd.concat([out[c].map(_num) for c in ["diag_1", "diag_2", "diag_3"]], axis=1)
    d3 = pd.concat([out[c].map(_icd3) for c in ["diag_1", "diag_2", "diag_3"]], axis=1)
    out["diag_1_icd3"] = d3.iloc[:, 0]
    out["dx_heart_failure"] = (d3 == "428").any(axis=1).astype(int)
    out["dx_renal"] = d3.isin(["584", "585", "586", "403", "404"]).any(axis=1).astype(int)
    out["dx_copd"] = d3.isin(["491", "492", "493", "496"]).any(axis=1).astype(int)
    out["dx_diab_complicated"] = ((dnum >= 250.1) & (dnum < 251)).any(axis=1).astype(int)
    out["dx_n_distinct_groups"] = out[["diag_1_grp", "diag_2_grp", "diag_3_grp"]].nunique(axis=1)
    # ---- grouping block --------------------------------------------------
    out["discharge_grp"] = out["discharge_disposition_id"].astype(int).map(_group_discharge)
    out["admsrc_grp"] = out["admission_source_id"].astype(int).map(_group_source)
    out["admtype_grp"] = out["admission_type_id"].astype(int).map(_group_admtype)
    out["age_num"] = out["age"].map(AGE_NUM).astype(float)
    # ---- interaction block ------------------------------------------------
    out["inpatient_x_los"] = out["number_inpatient"] * out["time_in_hospital"]
    out["inpatient_x_emergency"] = out["number_inpatient"] * out["number_emergency"]
    out["ndx_x_nmeds"] = out["number_diagnoses"] * out["num_medications"]
    return out


BLOCKS: Dict[str, Dict[str, List[str]]] = {
    "util": {"num": ["service_utilization", "log1p_number_outpatient", "log1p_number_emergency",
                     "log1p_number_inpatient", "any_prior_inpatient", "any_prior_emergency"], "cat": [], "hi": []},
    "meds": {"num": ["n_meds_active", "n_meds_up", "n_meds_down", "n_meds_changed", "insulin_active",
                     "insulin_changed", "n_oral_active", "meds_per_day"], "cat": [], "hi": []},
    "labs": {"num": ["a1c_tested", "a1c_high", "a1c_high_and_med_changed", "glu_tested", "labs_per_day"],
             "cat": [], "hi": []},
    "dx": {"num": ["dx_heart_failure", "dx_renal", "dx_copd", "dx_diab_complicated", "dx_n_distinct_groups"],
           "cat": [], "hi": ["diag_1_icd3"]},
    "groups": {"num": ["age_num"], "cat": ["discharge_grp", "admsrc_grp", "admtype_grp"], "hi": []},
    "interact": {"num": ["inpatient_x_los", "inpatient_x_emergency", "ndx_x_nmeds"], "cat": [], "hi": []},
}


def feature_lists(blocks: List[str] | None = None) -> Dict[str, List[str]]:
    """Return {'num','cat','hi'} column lists for FS-R plus the given blocks."""
    num, cat, hi = list(R_NUM), list(R_CAT), []
    for b in blocks or []:
        num += BLOCKS[b]["num"]
        cat += BLOCKS[b]["cat"]
        hi += BLOCKS[b]["hi"]
    return {"num": num, "cat": cat, "hi": hi}


def build_preprocessor(cols: Dict[str, List[str]], scale: bool = True) -> ColumnTransformer:
    """Mean-impute + standardise numerics; constant-impute + one-hot categoricals.

    Fitted inside every Pipeline, therefore only ever on training (fold) data.
    """
    num_steps = [("impute", SimpleImputer(strategy="mean"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [
        ("num", Pipeline(num_steps), cols["num"]),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cols["cat"]),
    ]
    if cols.get("hi"):
        transformers.append(("hi", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=40,
                                  sparse_output=False)),
        ]), cols["hi"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0,
                             verbose_feature_names_out=True)


def make_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature frame (base + engineered); categorical columns as str."""
    out = add_engineered(add_base_columns(df))
    obj_cols = R_CAT + [c for b in BLOCKS.values() for c in b["cat"] + b["hi"]]
    for c in obj_cols:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
        out.loc[out[c].notna(), c] = out.loc[out[c].notna(), c].astype(str)
    return out
