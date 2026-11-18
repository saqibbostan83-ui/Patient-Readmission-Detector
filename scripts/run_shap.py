"""Generate actual SHAP values and summary plots for the fitted XGBoost model.
Outputs:
  - reports/figures/13_shap_summary.png
  - reports/figures/14_shap_bar.png
  - results/shap_summary.json
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
import pandas as pd
import shap

from src.preprocess import get_dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset and model...")
data = get_dataset()
xgb_pipe = joblib.load(ROOT / "models" / "xgboost.joblib")
prep = xgb_pipe.named_steps["prep"]
clf = xgb_pipe.named_steps["clf"]

# Clean feature names for plotting
raw_feature_names = prep.get_feature_names_out()
clean_feature_names = [f.replace("cat__", "").replace("num__", "") for f in raw_feature_names]

X_test_trans = prep.transform(data["X_test"])

# Sample 2,500 test instances for reliable SHAP estimation
np.random.seed(42)
sample_idx = np.random.choice(len(X_test_trans), 2500, replace=False)
X_sample = X_test_trans[sample_idx]

print(f"Computing TreeExplainer SHAP values on {len(X_sample)} holdout test instances...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)

# Beeswarm summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=clean_feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Beeswarm Summary Plot (XGBoost Test Sample)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved reports/figures/13_shap_summary.png")

# Bar plot of mean absolute SHAP values (global importance)
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_features = []
for i in sorted_idx[:25]:
    top_features.append({
        "feature": clean_feature_names[i],
        "mean_abs_shap": float(mean_abs_shap[i])
    })

plt.figure(figsize=(10, 7))
y_pos = np.arange(20)
plt.barh(y_pos, [f["mean_abs_shap"] for f in top_features[:20]][::-1], color="#2b5c8f")
plt.yticks(y_pos, [f["feature"] for f in top_features[:20]][::-1])
plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds Output)")
plt.title("Top 20 Features by Mean Absolute SHAP Value (XGBoost)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "14_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved reports/figures/14_shap_bar.png")

(RES_DIR / "shap_summary.json").write_text(json.dumps(top_features, indent=2))
print("Saved results/shap_summary.json")
