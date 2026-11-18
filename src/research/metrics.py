"""Evaluation metrics for an imbalanced binary problem."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score)


def threshold_metrics(y: np.ndarray, proba: np.ndarray, t: float) -> Dict:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return {
        "threshold": float(t),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "balanced_accuracy": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # paper-style (sklearn classification_report) numbers
        "weighted_precision": float(w_p), "weighted_recall": float(w_r), "weighted_f1": float(w_f),
        "class0_precision": float(p_cls[0]), "class0_recall": float(r_cls[0]), "class0_f1": float(f_cls[0]),
        "class1_precision": float(p_cls[1]), "class1_recall": float(r_cls[1]), "class1_f1": float(f_cls[1]),
    }


def ranking_metrics(y: np.ndarray, proba: np.ndarray) -> Dict:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, np.clip(proba, 0, 1))),
        "prevalence": float(np.mean(y)),
        "mean_predicted": float(np.mean(proba)),
    }


def select_threshold(y: np.ndarray, proba: np.ndarray, rule: str = "f1") -> float:
    """Choose an operating threshold on VALIDATION / out-of-fold predictions only."""
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 197)))
    if rule == "f1":
        scores = [f1_score(y, proba >= t, zero_division=0) for t in grid]
    elif rule == "youden":
        scores = []
        for t in grid:
            pred = proba >= t
            tpr = pred[y == 1].mean()
            tnr = (~pred[y == 0]).mean()
            scores.append(tpr + tnr - 1)
    else:
        raise ValueError(rule)
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y: np.ndarray, proba: np.ndarray, metric=roc_auc_score, n: int = 1000,
                 seed: int = 42, groups: np.ndarray | None = None) -> tuple:
    """95% percentile bootstrap CI. If `groups` (patient ids) is given, whole
    patients are resampled (cluster bootstrap) to respect within-patient correlation."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [[] for _ in range(len(uniq))]
        for i, g in enumerate(inv):
            members[g].append(i)
        members = [np.array(m) for m in members]
    vals = []
    for _ in range(n):
        if groups is None:
            b = rng.choice(idx, size=len(idx), replace=True)
        else:
            gs = rng.integers(0, len(members), size=len(members))
            b = np.concatenate([members[g] for g in gs])
        if y[b].min() == y[b].max():
            continue
        vals.append(metric(y[b], proba[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
