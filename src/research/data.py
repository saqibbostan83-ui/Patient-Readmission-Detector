"""Data loading, integrity verification, cohort definition, label and splits."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from . import config as C

TARGET_RAW = "readmitted"
TARGET = "readmit_30d"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files() -> Dict[str, str]:
    """Check the raw files exist and match the official UCI release."""
    out = {}
    for path, expected in [(C.RAW_CSV, C.SHA256_DIABETIC), (C.IDS_CSV, C.SHA256_IDS)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python scripts/download_data.py` (official UCI source).")
        digest = sha256(path)
        out[path.name] = digest
        if digest != expected:
            raise ValueError(f"{path.name}: SHA-256 {digest} does not match the official UCI file")
    return out


def load_raw() -> pd.DataFrame:
    """Read the CSV treating ONLY '?' as missing.

    Important: pandas' default NA list contains the string 'None', which in this
    dataset is a real category of `max_glu_serum` / `A1Cresult` ('test not
    performed'). keep_default_na=False preserves it.
    """
    df = pd.read_csv(C.RAW_CSV, na_values=["?"], keep_default_na=False, low_memory=False)
    for col in ["max_glu_serum", "A1Cresult"]:
        df[col] = df[col].replace("", "None")
    return df


@dataclass
class Cohort:
    df: pd.DataFrame
    flow: List[Tuple[str, int]] = field(default_factory=list)


def build_cohort(raw: pd.DataFrame, exclude_hospice_expired: bool = True,
                 adults_only: bool = True) -> Cohort:
    """Apply the paper's stated exclusion criteria and build the binary label."""
    df = raw.copy()
    flow = [("All encounters (UCI file)", len(df))]
    if exclude_hospice_expired:
        df = df[~df["discharge_disposition_id"].isin(C.HOSPICE_EXPIRED_IDS)]
        flow.append(("Excluded discharged to hospice / expired", len(df)))
    if adults_only:
        df = df[~df["age"].isin(C.UNDER_ADULT_BINS)]
        flow.append(("Excluded age bins [0-10) and [10-20) (adult approximation)", len(df)))
    df = df[df[TARGET_RAW].isin(["<30", ">30", "NO"])]
    flow.append(("Valid readmission label", len(df)))
    df = df.reset_index(drop=True)
    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)
    return Cohort(df=df, flow=flow)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    protocol: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    cv_splits: List[Tuple[np.ndarray, np.ndarray]]   # indices relative to the TRAIN subset


def encounter_split(y: np.ndarray, seed: int = C.SEED) -> Split:
    """Protocol A: stratified encounter-level 80/20 split (paper methodology)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    tr, te = np.sort(tr), np.sort(te)
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    cv = list(skf.split(np.zeros(len(tr)), y[tr]))
    return Split("encounter", tr, te, cv)


def patient_split(y: np.ndarray, groups: np.ndarray, seed: int = C.SEED) -> Split:
    """Protocol B: patient-grouped split - no patient in both train and test,
    and CV folds inside train are also grouped by patient."""
    sgkf = StratifiedGroupKFold(n_splits=int(round(1 / C.TEST_SIZE)), shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    tr, te = np.sort(tr), np.sort(te)
    inner = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    cv = list(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
    assert not set(groups[tr]) & set(groups[te]), "patient leakage between train and test"
    for a, b in cv:
        assert not set(groups[tr][a]) & set(groups[tr][b]), "patient leakage inside CV"
    return Split("patient", tr, te, cv)
