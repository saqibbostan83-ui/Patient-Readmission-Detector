"""Generic, leakage-safe experiment runner.

For one (feature set, split protocol):
  1. FoldCache: for every CV fold the preprocessor is fitted on the fold's
     training part only, then used to transform the fold's validation part.
     The final preprocessor is fitted on the full training set only.
  2. cv_search: every candidate hyper-parameter set is fitted on each fold ->
     mean CV ROC-AUC/PR-AUC and full out-of-fold (OOF) predictions.
  3. The best candidate (highest mean CV ROC-AUC) is refitted on the full
     training set; its OOF predictions choose the operating threshold.
  4. Only then is the untouched test set scored - once.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from . import config as C
from .metrics import bootstrap_ci, ranking_metrics, select_threshold, threshold_metrics

log = C.get_logger()


@dataclass
class FoldCache:
    folds: List[tuple]          # (Xtr_t, ytr, Xva_t, yva, va_idx)
    X_train_t: np.ndarray
    X_test_t: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: object
    feature_names: List[str]
    n_train: int


def build_fold_cache(prep, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame,
                     y_test: np.ndarray, cv_splits) -> FoldCache:
    folds = []
    for tr, va in cv_splits:
        p = clone(prep).fit(X_train.iloc[tr])
        folds.append((p.transform(X_train.iloc[tr]).astype(np.float32), y_train[tr],
                      p.transform(X_train.iloc[va]).astype(np.float32), y_train[va], va))
    final = clone(prep).fit(X_train)
    return FoldCache(folds, final.transform(X_train).astype(np.float32),
                     final.transform(X_test).astype(np.float32), y_train, y_test, final,
                     list(final.get_feature_names_out()), len(y_train))


def _fit_predict(make_est: Callable, params: dict, Xtr, ytr, Xva, sampler=None):
    t0 = time.time()
    est = make_est(**params)
    if sampler is not None:
        Xtr, ytr = clone(sampler).fit_resample(Xtr, ytr)
    est.fit(Xtr, ytr)
    return est.predict_proba(Xva)[:, 1], time.time() - t0


def cv_search(make_est: Callable, candidates: List[dict], cache: FoldCache, n_jobs: int = 4,
              sampler=None) -> tuple:
    tasks = [(ci, fi) for ci in range(len(candidates)) for fi in range(len(cache.folds))]
    if n_jobs == 1:
        outs = [_fit_predict(make_est, candidates[ci], cache.folds[fi][0], cache.folds[fi][1],
                             cache.folds[fi][2], sampler) for ci, fi in tasks]
    else:
        outs = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
            delayed(_fit_predict)(make_est, candidates[ci], cache.folds[fi][0], cache.folds[fi][1],
                                  cache.folds[fi][2], sampler) for ci, fi in tasks)
    oof = np.zeros((len(candidates), cache.n_train), dtype=np.float64)
    rows = []
    fold_auc = {ci: [] for ci in range(len(candidates))}
    fold_ap = {ci: [] for ci in range(len(candidates))}
    secs = {ci: 0.0 for ci in range(len(candidates))}
    for (ci, fi), (p, s) in zip(tasks, outs):
        _, _, _, yva, va = cache.folds[fi]
        oof[ci, va] = p
        fold_auc[ci].append(roc_auc_score(yva, p))
        fold_ap[ci].append(average_precision_score(yva, p))
        secs[ci] += s
    for ci, params in enumerate(candidates):
        rows.append({"candidate": ci, "params": json.dumps(params, default=str),
                     "cv_roc_auc_mean": float(np.mean(fold_auc[ci])),
                     "cv_roc_auc_std": float(np.std(fold_auc[ci])),
                     "cv_pr_auc_mean": float(np.mean(fold_ap[ci])),
                     "fit_seconds_total": secs[ci]})
    return pd.DataFrame(rows), oof


def run_experiment(name: str, leaderboard: str, protocol: str, make_est: Callable,
                   candidates: List[dict], cache: FoldCache, out_dir: Path, test_groups=None,
                   test_ids=None, n_jobs: int = 4, sampler=None, notes: str = "",
                   extra_config: Optional[dict] = None, save_model: bool = True) -> Dict:
    """Tune with CV on train, pick threshold on OOF, refit, evaluate on test once, persist."""
    t_start = time.time()
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{protocol}/{leaderboard}] {name}: {len(candidates)} candidates x {len(cache.folds)} folds")
    cv_df, oof = cv_search(make_est, candidates, cache, n_jobs=n_jobs, sampler=sampler)
    best = int(cv_df["cv_roc_auc_mean"].idxmax())
    best_params = candidates[best]
    oof_best = oof[best]
    thr_f1 = select_threshold(cache.y_train, oof_best, "f1")
    thr_youden = select_threshold(cache.y_train, oof_best, "youden")

    t0 = time.time()
    est = make_est(**best_params)
    Xtr, ytr = cache.X_train_t, cache.y_train
    if sampler is not None:
        Xtr, ytr = clone(sampler).fit_resample(Xtr, ytr)
    est.fit(Xtr, ytr)
    fit_s = time.time() - t0
    p_test = est.predict_proba(cache.X_test_t)[:, 1]
    p_train = est.predict_proba(cache.X_train_t)[:, 1]
    y = cache.y_test

    rec = {
        "experiment": name, "leaderboard": leaderboard, "protocol": protocol, "notes": notes,
        "best_params": best_params, "n_candidates": len(candidates), "n_folds": len(cache.folds),
        "cv_roc_auc_mean": float(cv_df.loc[best, "cv_roc_auc_mean"]),
        "cv_roc_auc_std": float(cv_df.loc[best, "cv_roc_auc_std"]),
        "cv_pr_auc_mean": float(cv_df.loc[best, "cv_pr_auc_mean"]),
        "oof_roc_auc": float(roc_auc_score(cache.y_train, oof_best)),
        "train_roc_auc_refit": float(roc_auc_score(cache.y_train, p_train)),  # for overfitting check only
        "threshold_f1_oof": thr_f1, "threshold_youden_oof": thr_youden,
        "test": ranking_metrics(y, p_test),
        "test_at_0.5": threshold_metrics(y, p_test, 0.5),
        "test_at_oof_f1_threshold": threshold_metrics(y, p_test, thr_f1),
        "test_at_oof_youden_threshold": threshold_metrics(y, p_test, thr_youden),
        "test_roc_auc_95ci": bootstrap_ci(y, p_test, roc_auc_score, n=500, groups=test_groups),
        "test_pr_auc_95ci": bootstrap_ci(y, p_test, average_precision_score, n=500, groups=test_groups),
        "n_train": int(len(cache.y_train)), "n_test": int(len(y)),
        "n_features": int(cache.X_train_t.shape[1]),
        "final_fit_seconds": fit_s, "total_seconds": time.time() - t_start,
        "sampler": type(sampler).__name__ if sampler is not None else None,
    }
    if extra_config:
        rec["config"] = extra_config
    cv_df.to_csv(exp_dir / "cv_results.csv", index=False)
    pd.DataFrame({"row_id": test_ids if test_ids is not None else np.arange(len(y)),
                  "y_true": y, "proba": p_test}).to_csv(exp_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"y_true": cache.y_train, "oof_proba": oof_best}).to_csv(exp_dir / "oof_predictions.csv",
                                                                         index=False)
    (exp_dir / "metrics.json").write_text(json.dumps(rec, indent=2, default=str))
    if save_model:
        mdir = C.MODELS_OUT / protocol
        mdir.mkdir(parents=True, exist_ok=True)
        steps = [("prep", cache.preprocessor), ("clf", est)]
        joblib.dump(Pipeline(steps), mdir / f"{name}.joblib", compress=3)
    m = rec["test"]
    log.info(f"   -> best {best_params} | CV AUC {rec['cv_roc_auc_mean']:.4f}±{rec['cv_roc_auc_std']:.4f} "
             f"| TEST AUC {m['roc_auc']:.4f} PR-AUC {m['pr_auc']:.4f} | thr(F1,OOF)={thr_f1:.3f} "
             f"| {rec['total_seconds']:.0f}s")
    rec["_proba_test"] = p_test
    rec["_oof"] = oof_best
    rec["_estimator"] = est
    return rec


def flat_row(rec: Dict) -> Dict:
    """One leaderboard row (CSV friendly)."""
    t, a, f = rec["test"], rec["test_at_0.5"], rec["test_at_oof_f1_threshold"]
    return {
        "protocol": rec["protocol"], "leaderboard": rec["leaderboard"], "experiment": rec["experiment"],
        "cv_roc_auc": rec["cv_roc_auc_mean"], "cv_roc_auc_std": rec["cv_roc_auc_std"],
        "cv_pr_auc": rec["cv_pr_auc_mean"],
        "test_roc_auc": t["roc_auc"], "test_roc_auc_ci_low": rec["test_roc_auc_95ci"][0],
        "test_roc_auc_ci_high": rec["test_roc_auc_95ci"][1], "test_pr_auc": t["pr_auc"],
        "brier": t["brier"],
        "acc@0.5": a["accuracy"], "prec@0.5": a["precision"], "rec@0.5": a["recall"], "f1@0.5": a["f1"],
        "spec@0.5": a["specificity"],
        "thr_oof_f1": rec["threshold_f1_oof"], "acc@thr": f["accuracy"], "prec@thr": f["precision"],
        "rec@thr": f["recall"], "f1@thr": f["f1"], "spec@thr": f["specificity"],
        "tn@thr": f["tn"], "fp@thr": f["fp"], "fn@thr": f["fn"], "tp@thr": f["tp"],
        "train_auc_refit": rec["train_roc_auc_refit"], "n_features": rec["n_features"],
        "n_test": rec["n_test"], "best_params": json.dumps(rec["best_params"], default=str),
        "seconds": rec["total_seconds"],
    }
