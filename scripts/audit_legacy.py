"""Audit checks on the dataset and on the legacy (pre-existing) artifacts.

Writes results/audit/legacy_audit.json. Read-only with respect to legacy files.
Run from project root:  python scripts/audit_legacy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

res: dict = {}
raw = pd.read_csv(ROOT / "data" / "raw" / "diabetic_data.csv", na_values="?", low_memory=False)
res["shape"] = list(raw.shape)
res["n_patients"] = int(raw["patient_nbr"].nunique())
res["encounter_id_unique"] = bool(raw["encounter_id"].is_unique)
res["readmitted_counts"] = raw["readmitted"].value_counts().to_dict()
res["missing_pct"] = (raw.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict()
res["age_counts"] = raw["age"].value_counts().sort_index().to_dict()
res["discharge_counts"] = raw["discharge_disposition_id"].value_counts().sort_index().to_dict()
hospice_expired = [11, 13, 14, 19, 20, 21]
res["n_hospice_expired"] = int(raw["discharge_disposition_id"].isin(hospice_expired).sum())
res["readmit30_among_hospice_expired"] = int(
    (raw.loc[raw["discharge_disposition_id"].isin(hospice_expired), "readmitted"] == "<30").sum())
enc_per_pat = raw.groupby("patient_nbr").size()
res["patients_with_multiple_encounters"] = int((enc_per_pat > 1).sum())
res["encounters_from_multi_patients"] = int(enc_per_pat[enc_per_pat > 1].sum())
res["max_encounters_one_patient"] = int(enc_per_pat.max())
feat_cols = [c for c in raw.columns if c not in ("encounter_id", "patient_nbr", "readmitted")]
res["exact_duplicate_feature_rows"] = int(raw.duplicated(subset=feat_cols).sum())
res["exact_duplicate_feature_rows_conflicting_label"] = int(
    raw[raw.duplicated(subset=feat_cols, keep=False)].groupby(feat_cols, dropna=False)["readmitted"]
    .nunique().gt(1).sum())

# --- patient overlap in the legacy stratified encounter split ---
y = (raw["readmitted"] == "<30").astype(int)
idx_tr, idx_te = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=42, stratify=y)
p_tr = set(raw["patient_nbr"].iloc[idx_tr])
te_pat = raw["patient_nbr"].iloc[idx_te]
res["legacy_split_test_rows"] = int(len(idx_te))
res["legacy_split_test_positives"] = int(y.iloc[idx_te].sum())
res["legacy_split_test_encounters_whose_patient_in_train"] = int(te_pat.isin(p_tr).sum())
res["legacy_split_test_patients_also_in_train_pct"] = float(te_pat.isin(p_tr).mean() * 100)

# --- legacy artifacts consistency ---
legacy = {}
try:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    preds = np.load(ROOT / "models" / "preds_test.npz")
    y_te_saved = preds["y_test"]
    legacy["saved_y_test_matches_regenerated_split"] = bool(
        np.array_equal(y_te_saved, y.iloc[idx_te].values))
    for k in preds.files:
        if k == "y_test":
            continue
        auc = roc_auc_score(y_te_saved, preds[k])
        legacy[f"{k}_auc_from_saved_preds"] = round(float(auc), 6)
        legacy[f"{k}_auc_in_metrics_json"] = round(metrics["models"][k]["roc_auc"], 6)
        # was best threshold chosen on TEST labels?
        ths = np.linspace(0.05, 0.95, 181)
        f1s = [f1_score(y_te_saved, preds[k] >= t, zero_division=0) for t in ths]
        legacy[f"{k}_best_threshold_reproduced_from_test_labels"] = round(float(ths[int(np.argmax(f1s))]), 4)
        legacy[f"{k}_best_threshold_in_metrics_json"] = round(metrics["models"][k]["best_threshold"], 4)
except Exception as e:  # noqa: BLE001
    legacy["error_preds"] = repr(e)

# can the saved pipelines be loaded and reproduce saved predictions?
try:
    from src.preprocess import get_dataset  # needs data/raw/IDs_mapping.csv
    data = get_dataset()
    for name in ["logreg", "xgboost", "random_forest"]:
        try:
            m = joblib.load(ROOT / "models" / f"{name}.joblib")
            p = m.predict_proba(data["X_test"])[:, 1]
            legacy[f"{name}_reload_max_abs_diff_vs_saved_preds"] = float(np.max(np.abs(p - preds[name])))
        except Exception as e:  # noqa: BLE001
            legacy[f"{name}_reload_error"] = repr(e)[:300]
except Exception as e:  # noqa: BLE001
    legacy["error_reload"] = repr(e)[:300]
res["legacy_artifacts"] = legacy

(OUT / "legacy_audit.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
