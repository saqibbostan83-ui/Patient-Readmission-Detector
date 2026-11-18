"""Compute SHAP values specifically for the eligible (non-hospice, non-expired) cohort.
Addresses reviewer observation:
Expired/Hospice discharge is a structural cohort artifact (deterministic non-readmission),
not a clinically useful predictive feature for living patients at discharge.
Outputs:
  - reports/figures/15_shap_bar_eligible.png
  - results/shap_summary_eligible.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.preprocess import get_dataset

FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset and model...")
data = get_dataset()
xgb_pipe = joblib.load(ROOT / "models" / "xgboost.joblib")
prep = xgb_pipe.named_steps["prep"]
clf = xgb_pipe.named_steps["clf"]

raw_feature_names = prep.get_feature_names_out()
clean_feature_names = [f.replace("cat__", "").replace("num__", "") for f in raw_feature_names]

X_test_raw = data["X_test"]
# Filter to eligible patients only (excluding hospice and expired encounters)
eligible_mask = (X_test_raw["discharge_disposition"] != "Expired/Hospice").values
eligible_indices = np.where(eligible_mask)[0]

print(f"Total test instances: {len(X_test_raw)}, Eligible instances: {len(eligible_indices)}")

np.random.seed(42)
sample_idx = np.random.choice(eligible_indices, 2500, replace=False)

X_test_trans = prep.transform(X_test_raw.iloc[sample_idx])

print(f"Computing TreeExplainer SHAP values on {len(X_test_trans)} eligible holdout test instances...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_trans)

# Global mean |SHAP| for eligible patients
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_features = []
for i in sorted_idx[:25]:
    top_features.append({
        "feature": clean_feature_names[i],
        "mean_abs_shap": float(mean_abs_shap[i])
    })

# Plot comparison
plt.figure(figsize=(10, 7))
y_pos = np.arange(20)
plt.barh(y_pos, [f["mean_abs_shap"] for f in top_features[:20]][::-1], color="#1b7837")
plt.yticks(y_pos, [f["feature"] for f in top_features[:20]][::-1])
plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds Output)")
plt.title("Top 20 Features by Mean Absolute SHAP Value (Eligible Non-Expired Cohort)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "15_shap_bar_eligible.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved reports/figures/15_shap_bar_eligible.png")

(RES_DIR / "shap_summary_eligible.json").write_text(json.dumps(top_features, indent=2))
print("Saved results/shap_summary_eligible.json")
