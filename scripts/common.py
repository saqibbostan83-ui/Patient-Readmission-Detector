"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research import config as C  # noqa: E402
from src.research.data import (TARGET, build_cohort, encounter_split, load_raw,  # noqa: E402
                               patient_split, verify_files)
from src.research.experiment import build_fold_cache, flat_row, run_experiment  # noqa: E402
from src.research.features import build_preprocessor, feature_lists, make_frame  # noqa: E402
from src.research.models import ModelFactory  # noqa: E402

log = C.get_logger()


def load_cohort(exclude=True, adults=True):
    verify_files()
    raw = load_raw()
    coh = build_cohort(raw, exclude_hospice_expired=exclude, adults_only=adults)
    frame = make_frame(coh.df)
    return coh, frame


def make_split(frame: pd.DataFrame, protocol: str):
    y = frame[TARGET].values
    if protocol == "encounter":
        return encounter_split(y)
    return patient_split(y, frame["patient_nbr"].values)


def split_summary(frame, split) -> dict:
    g = frame["patient_nbr"].values
    y = frame[TARGET].values
    tr_p = set(g[split.train_idx])
    te_g = g[split.test_idx]
    return {
        "protocol": split.protocol,
        "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx)),
        "pos_rate_train": float(y[split.train_idx].mean()), "pos_rate_test": float(y[split.test_idx].mean()),
        "n_pos_test": int(y[split.test_idx].sum()),
        "test_encounters_with_patient_in_train": int(np.isin(te_g, list(tr_p)).sum()),
        "test_encounters_with_patient_in_train_pct": float(np.isin(te_g, list(tr_p)).mean() * 100),
        "n_patients_train": int(len(tr_p)), "n_patients_test": int(len(set(te_g))),
    }


def get_cache(frame, split, blocks=None):
    cols = feature_lists(blocks)
    prep = build_preprocessor(cols)
    X = frame
    y = frame[TARGET].values
    Xtr, Xte = X.iloc[split.train_idx], X.iloc[split.test_idx]
    return build_fold_cache(prep, Xtr, y[split.train_idx], Xte, y[split.test_idx], split.cv_splits)


def estimator_factory(kind, y_train, threads=1, **fixed):
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    return ModelFactory(kind, pos_weight, threads, fixed)


def save_leaderboard(records, path_stem: Path):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([flat_row(r) for r in records])
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    path_stem.with_suffix(".json").write_text(json.dumps(clean, indent=2, default=str))
    return df


def save_probas(records, frame, split, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_test=frame[TARGET].values[split.test_idx],
                        encounter_id=frame["encounter_id"].values[split.test_idx],
                        patient_nbr=frame["patient_nbr"].values[split.test_idx],
                        **{r["experiment"]: r["_proba_test"] for r in records})
