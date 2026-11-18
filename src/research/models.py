"""Model factories and documented hyper-parameter search spaces.

Leaderboard A (replication) uses exhaustive GRID search (paper: "grid search
with five-fold cross-validation"); the paper does not publish its grids, so the
grids below are our documented assumption, kept small enough to execute.
Leaderboard B (improved) uses seeded RANDOM search over wider spaces.
"""
from __future__ import annotations

from typing import Dict, List

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, ParameterSampler

from .config import SEED


def make_model(kind: str, pos_weight: float, threads: int = 1, **params):
    """Return an unfitted estimator. Class weighting is applied consistently:
    'balanced' weights for LR/RF/DNN/LightGBM/CatBoost, scale_pos_weight=neg/pos for XGBoost."""
    if kind == "logreg":
        base = dict(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED)
        base.update(params)
        return LogisticRegression(**base)
    if kind == "rf":
        base = dict(n_estimators=300, class_weight="balanced", n_jobs=threads, random_state=SEED)
        base.update(params)
        return RandomForestClassifier(**base)
    if kind == "xgb":
        from xgboost import XGBClassifier
        base = dict(tree_method="hist", scale_pos_weight=pos_weight, n_jobs=threads,
                    random_state=SEED, eval_metric="logloss")
        base.update(params)
        return XGBClassifier(**base)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        base = dict(class_weight="balanced", n_jobs=threads, random_state=SEED, verbose=-1)
        base.update(params)
        return LGBMClassifier(**base)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        base = dict(auto_class_weights="Balanced", thread_count=threads, random_seed=SEED,
                    verbose=0, allow_writing_files=False)
        base.update(params)
        return CatBoostClassifier(**base)
    if kind == "dnn":
        from .keras_mlp import KerasDNNClassifier
        base = dict(class_weight="balanced", random_state=SEED)
        base.update(params)
        return KerasDNNClassifier(**base)
    raise ValueError(kind)


class ModelFactory:
    """Picklable callable: ModelFactory(kind, pos_weight, threads, fixed)(**params) -> estimator."""

    def __init__(self, kind: str, pos_weight: float, threads: int = 1, fixed: dict | None = None):
        self.kind, self.pos_weight, self.threads, self.fixed = kind, pos_weight, threads, fixed or {}

    def __call__(self, **params):
        return make_model(self.kind, self.pos_weight, self.threads, **{**self.fixed, **params})


# --------------------------------------------------------------------------- #
# Leaderboard A - replication grids (exhaustive grid search, 5-fold CV)
# --------------------------------------------------------------------------- #
REPLICATION_GRIDS: Dict[str, Dict[str, list]] = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "rf": {"max_depth": [10, 20, None], "min_samples_leaf": [1, 10]},
    "xgb": {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    "dnn": {"hidden": [(128, 64), (256, 128, 64)], "dropout": [0.2, 0.4]},
}


def replication_candidates(kind: str) -> List[dict]:
    return list(ParameterGrid(REPLICATION_GRIDS[kind]))


# --------------------------------------------------------------------------- #
# Leaderboard B - improved spaces (seeded random search, 5-fold CV)
# --------------------------------------------------------------------------- #
IMPROVED_SPACES = {
    "logreg": None,  # explicit list below (L1 and L2)
    "rf": {"n_estimators": [400], "max_depth": [12, 16, 20, None],
           "min_samples_leaf": [5, 10, 20, 50], "max_features": ["sqrt", 0.2, 0.33]},
    "xgb": {"n_estimators": randint(300, 1200), "learning_rate": loguniform(0.01, 0.1),
            "max_depth": randint(3, 8), "min_child_weight": [1, 5, 10, 20, 50],
            "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.6),
            "reg_lambda": loguniform(0.5, 20), "gamma": [0, 0.5, 1, 2]},
    "lgbm": {"n_estimators": randint(300, 1200), "learning_rate": loguniform(0.01, 0.1),
             "num_leaves": randint(8, 64), "min_child_samples": [20, 50, 100, 200],
             "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
             "colsample_bytree": uniform(0.4, 0.6), "reg_lambda": loguniform(0.1, 20)},
    "catboost": {"iterations": [800], "learning_rate": [0.03, 0.06], "depth": [4, 6, 8],
                 "l2_leaf_reg": [3, 10]},
    "dnn": {"hidden": [(256, 128), (512, 256, 128)], "dropout": [0.3, 0.5],
            "l2": [1e-5, 1e-4], "batch_norm": [True], "learning_rate": [1e-3, 3e-4]},
}
IMPROVED_N_ITER = {"rf": 8, "xgb": 16, "lgbm": 16, "catboost": 4, "dnn": 4}


def improved_candidates(kind: str) -> List[dict]:
    if kind == "logreg":
        l2 = [{"C": c, "penalty": "l2"} for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]]
        l1 = [{"C": c, "penalty": "l1", "solver": "liblinear"} for c in [0.01, 0.03, 0.1]]
        return l2 + l1
    space = IMPROVED_SPACES[kind]
    if kind == "catboost":
        return list(ParameterSampler(space, n_iter=IMPROVED_N_ITER[kind], random_state=SEED))
    return list(ParameterSampler(space, n_iter=IMPROVED_N_ITER[kind], random_state=SEED))
