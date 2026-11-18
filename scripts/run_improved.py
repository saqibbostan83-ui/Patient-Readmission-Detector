"""LEADERBOARD B - improved experiments (kept separate from the replication).

Stages (each recorded in results/improved/<protocol>/):
  B0  feature-block ablation    : LightGBM (fixed params), 5-fold CV on TRAIN only
                                  -> selects the engineered feature set FS-E
  B1  tuning effect only        : tuned XGBoost on FS-R (random search)
  B2  model zoo on FS-E         : LR (L1/L2), RF, XGBoost, LightGBM, CatBoost, improved DNN
  B3  imbalance handling        : LightGBM (best params) with none / class weights / SMOTE (in-fold)
  B4  soft-voting ensemble      : average of the top-3 B2 models by CV ROC-AUC
Threshold for every model = F1-optimal threshold on its out-of-fold predictions.

Usage: python scripts/run_improved.py [--protocols patient,encounter] [--fast]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (C, estimator_factory, get_cache, load_cohort, log, make_split, run_experiment,
                    save_leaderboard, save_probas, split_summary)
from src.research.experiment import cv_search
from src.research.features import BLOCKS
from src.research.metrics import bootstrap_ci, ranking_metrics, select_threshold, threshold_metrics
from src.research.models import improved_candidates

ABLATION_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6)


def ablation(frame, split, out_dir):
    """B0: CV-only comparison of feature blocks (test set never touched)."""
    rows = []
    configs = [("FS-R", [])] + [(f"FS-R+{b}", [b]) for b in BLOCKS] + [("FS-R+all", list(BLOCKS))]
    for name, blocks in configs:
        cache = get_cache(frame, split, blocks)
        mk = estimator_factory("lgbm", cache.y_train, threads=1)
        df, _ = cv_search(mk, [ABLATION_PARAMS], cache, n_jobs=4)
        rows.append({"feature_set": name, "blocks": "+".join(blocks), "n_features": cache.X_train_t.shape[1],
                     "cv_roc_auc_mean": df.loc[0, "cv_roc_auc_mean"], "cv_roc_auc_std": df.loc[0, "cv_roc_auc_std"],
                     "cv_pr_auc_mean": df.loc[0, "cv_pr_auc_mean"]})
        log.info(f"  ablation {name:16s} CV AUC {rows[-1]['cv_roc_auc_mean']:.4f} "
                 f"PR-AUC {rows[-1]['cv_pr_auc_mean']:.4f} ({rows[-1]['n_features']} features)")
    abl = pd.DataFrame(rows)
    base = abl.loc[0, "cv_roc_auc_mean"]
    abl["delta_vs_FS-R"] = abl["cv_roc_auc_mean"] - base
    abl.to_csv(out_dir / "B0_feature_ablation.csv", index=False)
    # Decision rule (pre-specified): keep a block if it improves mean CV ROC-AUC over FS-R by > 0.0005;
    # if FS-R+all beats the union of kept blocks, the comparison is reported below.
    kept = [b for b in BLOCKS if float(abl.loc[abl["feature_set"] == f"FS-R+{b}", "delta_vs_FS-R"].iloc[0]) > 0.0005]
    return abl, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="patient,encounter")
    ap.add_argument("--fast", action="store_true", help="fewer candidates (debug)")
    args = ap.parse_args()
    C.set_seeds()
    t0 = time.time()
    coh, frame = load_cohort(exclude=True, adults=True)
    out_root = C.RESULTS / "improved"
    selected_blocks = None

    for protocol in args.protocols.split(","):
        split = make_split(frame, protocol)
        summ = split_summary(frame, split)
        out_dir = out_root / protocol
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "split_summary.json").write_text(json.dumps(summ, indent=2))
        test_groups = frame["patient_nbr"].values[split.test_idx]
        test_ids = frame["encounter_id"].values[split.test_idx]
        log.info(f"=== Improved | protocol={protocol} | {json.dumps(summ)}")

        # ---------------- B0 ablation (on the first protocol = patient) ----------
        if selected_blocks is None:
            abl, kept = ablation(frame, split, out_dir)
            # compare union-of-kept vs all, using CV only
            if kept and set(kept) != set(BLOCKS):
                cache_k = get_cache(frame, split, kept)
                df_k, _ = cv_search(estimator_factory("lgbm", cache_k.y_train), [ABLATION_PARAMS], cache_k, 4)
                auc_kept = df_k.loc[0, "cv_roc_auc_mean"]
                auc_all = float(abl.loc[abl["feature_set"] == "FS-R+all", "cv_roc_auc_mean"].iloc[0])
                selected_blocks = kept if auc_kept >= auc_all else list(BLOCKS)
                log.info(f"  union of kept blocks {kept}: CV AUC {auc_kept:.4f} vs all {auc_all:.4f}")
            else:
                selected_blocks = kept or []
            (out_root / "selected_feature_blocks.json").write_text(json.dumps(
                {"selected_blocks": selected_blocks, "selected_on_protocol": protocol,
                 "rule": "block kept if FS-R+block improves mean 5-fold CV ROC-AUC (LightGBM, train only) by >0.0005; "
                         "union of kept blocks compared with FS-R+all by CV"}, indent=2))
            log.info(f"  selected FS-E blocks: {selected_blocks}")

        records = []
        cache_r = get_cache(frame, split, None)
        cache_e = get_cache(frame, split, selected_blocks)
        log.info(f"  FS-E matrix: {cache_e.X_train_t.shape[1]} features")

        def run(name, kind, cache, cands, n_jobs=4, threads=1, sampler=None, fixed=None, notes="", fs="FS-E"):
            mk = estimator_factory(kind, cache.y_train, threads=threads, **(fixed or {}))
            r = run_experiment(name, "B_improved", protocol, mk, cands, cache, out_dir, test_groups=test_groups,
                               test_ids=test_ids, n_jobs=n_jobs, sampler=sampler, notes=notes,
                               extra_config={"feature_set": fs, "blocks": selected_blocks if fs == "FS-E" else [],
                                             "split": summ})
            records.append(r)
            return r

        def cands(kind, n=None):
            c = improved_candidates(kind)
            return c[: (n or len(c))] if not args.fast else c[:2]

        # ---------------- B1 tuning effect only (FS-R) ---------------------------
        run("B1_XGB_tuned_FS-R", "xgb", cache_r, cands("xgb"), fs="FS-R",
            notes="random search, FS-R features: isolates the effect of tuning")
        # ---------------- B2 model zoo on FS-E ----------------------------------
        run("B2_LR_regularised", "logreg", cache_e, cands("logreg"), notes="L1/L2, class weights")
        run("B2_RF_tuned", "rf", cache_e, cands("rf"), notes="random search, class weights")
        r_xgb = run("B2_XGB_tuned", "xgb", cache_e, cands("xgb"), notes="random search, scale_pos_weight")
        r_lgb = run("B2_LGBM_tuned", "lgbm", cache_e, cands("lgbm"), notes="random search, balanced weights")
        try:
            run("B2_CatBoost", "catboost", cache_e, cands("catboost"), notes="grid sample, balanced weights")
        except Exception as e:  # noqa: BLE001
            log.error(f"CatBoost failed: {e!r}")
        run("B2_DNN_improved", "dnn", cache_e, cands("dnn"), n_jobs=1, threads=4,
            notes="BatchNorm + dropout + L2, random search")

        # ---------------- B3 imbalance handling (LightGBM, best params) -----------
        best_lgb = {k: v for k, v in r_lgb["best_params"].items()}
        run("B3_LGBM_no_weighting", "lgbm", cache_e, [best_lgb], fixed={"class_weight": None},
            notes="no imbalance handling")
        run("B3_LGBM_SMOTE", "lgbm", cache_e, [best_lgb], fixed={"class_weight": None},
            sampler=SMOTE(random_state=C.SEED), notes="SMOTE applied inside each training fold only")
        # (class-weighted variant = B2_LGBM_tuned)

        # ---------------- B4 ensemble of top-3 B2 models (selected on CV) --------
        b2 = [r for r in records if r["experiment"].startswith("B2_")]
        top = sorted(b2, key=lambda r: r["cv_roc_auc_mean"], reverse=True)[:3]
        oof = np.mean([r["_oof"] for r in top], axis=0)
        pte = np.mean([r["_proba_test"] for r in top], axis=0)
        ytr, yte = cache_e.y_train, cache_e.y_test
        # fold-wise CV AUC of the ensemble from OOF predictions
        fold_aucs = [roc_auc_score(ytr[va], oof[va]) for _, va in split.cv_splits]
        thr = select_threshold(ytr, oof, "f1")
        ens = {
            "experiment": "B4_Ensemble_top3", "leaderboard": "B_improved", "protocol": protocol,
            "notes": "soft vote (mean probability) of " + ", ".join(r["experiment"] for r in top),
            "best_params": {"members": [r["experiment"] for r in top]}, "n_candidates": 1, "n_folds": 5,
            "cv_roc_auc_mean": float(np.mean(fold_aucs)), "cv_roc_auc_std": float(np.std(fold_aucs)),
            "cv_pr_auc_mean": float(np.mean([average_precision_score(ytr[va], oof[va]) for _, va in split.cv_splits])),
            "oof_roc_auc": float(roc_auc_score(ytr, oof)),
            "train_roc_auc_refit": float("nan"),
            "threshold_f1_oof": thr, "threshold_youden_oof": select_threshold(ytr, oof, "youden"),
            "test": ranking_metrics(yte, pte), "test_at_0.5": threshold_metrics(yte, pte, 0.5),
            "test_at_oof_f1_threshold": threshold_metrics(yte, pte, thr),
            "test_roc_auc_95ci": bootstrap_ci(yte, pte, roc_auc_score, n=500, groups=test_groups),
            "test_pr_auc_95ci": bootstrap_ci(yte, pte, average_precision_score, n=500, groups=test_groups),
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "n_features": cache_e.X_train_t.shape[1],
            "total_seconds": 0.0, "_proba_test": pte, "_oof": oof,
        }
        ens["test_at_oof_youden_threshold"] = threshold_metrics(yte, pte, ens["threshold_youden_oof"])
        d = out_dir / "B4_Ensemble_top3"
        d.mkdir(exist_ok=True)
        pd.DataFrame({"row_id": test_ids, "y_true": yte, "proba": pte}).to_csv(d / "test_predictions.csv", index=False)
        (d / "metrics.json").write_text(json.dumps({k: v for k, v in ens.items() if not k.startswith("_")},
                                                   indent=2, default=str))
        log.info(f"   -> ensemble {ens['notes']} | CV AUC {ens['cv_roc_auc_mean']:.4f} | "
                 f"TEST AUC {ens['test']['roc_auc']:.4f}")
        records.append(ens)

        save_leaderboard(records, out_dir / "leaderboard_B")
        save_probas(records, frame, split, out_dir / "test_probas.npz")
        (out_dir / "feature_names_FS-E.json").write_text(json.dumps(cache_e.feature_names))
    log.info(f"Improved experiments finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
