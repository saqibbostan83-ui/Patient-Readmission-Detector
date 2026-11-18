"""Audit of the calibrated model: threshold selection on validation vs test."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from src.preprocess import get_dataset

data = get_dataset()
cal_model = joblib.load(ROOT / "models" / "best_model.joblib")

# Split X_train into train_sub and val to simulate a clean validation threshold selection
X_tr, X_val, y_tr, y_val = train_test_split(
    data["X_train"], data["y_train"], test_size=0.20, random_state=42, stratify=data["y_train"]
)

val_probas = cal_model.predict_proba(X_val)[:, 1]
test_probas = cal_model.predict_proba(data["X_test"])[:, 1]
y_test = data["y_test"].values

# Select threshold on validation set ONLY (no test data used)
thresholds = np.linspace(0.01, 0.90, 180)
val_f1s = [f1_score(y_val, val_probas >= t, zero_division=0) for t in thresholds]
best_val_idx = int(np.argmax(val_f1s))
val_selected_threshold = float(thresholds[best_val_idx])

# Also Youden's index on validation
val_youdens = []
for t in thresholds:
    p = val_probas >= t
    tpr = p[y_val == 1].mean()
    tnr = (~p[y_val == 0]).mean()
    val_youdens.append(tpr + tnr - 1)
val_youden_threshold = float(thresholds[int(np.argmax(val_youdens))])

def eval_at_t(y, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": t,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

res = {
    "test_roc_auc": float(roc_auc_score(y_test, test_probas)),
    "test_pr_auc": float(average_precision_score(y_test, test_probas)),
    "test_brier": float(brier_score_loss(y_test, test_probas)),
    "mean_predicted_probability": float(np.mean(test_probas)),
    "base_rate": float(np.mean(y_test)),
    "val_selected_f1_threshold": val_selected_threshold,
    "val_selected_youden_threshold": val_youden_threshold,
    "test_at_default_0_50": eval_at_t(y_test, test_probas, 0.50),
    "test_at_val_f1_threshold": eval_at_t(y_test, test_probas, val_selected_threshold),
    "test_at_val_youden_threshold": eval_at_t(y_test, test_probas, val_youden_threshold),
}

import json
print(json.dumps(res, indent=2))
(ROOT / "results" / "calibrated_model_audit.json").write_text(json.dumps(res, indent=2))
