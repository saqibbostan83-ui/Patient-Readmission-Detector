"""Central configuration: every path is relative to the project root."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
RAW_CSV = DATA_RAW / "diabetic_data.csv"
IDS_CSV = DATA_RAW / "IDs_mapping.csv"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS_OUT = ROOT / "models" / "research"   # new artefacts (legacy files in models/ untouched)
REPORTS = ROOT / "reports"

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.20

# Official UCI release (dataset id 296); SHA-256 verified on 2026-11-18
UCI_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
SHA256_DIABETIC = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
SHA256_IDS = "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e"

# discharge_disposition_id codes meaning expired or hospice (IDs_mapping.csv):
# 11 Expired, 13 Hospice/home, 14 Hospice/medical facility,
# 19/20/21 Expired at home / in a medical facility / place unknown (Medicaid hospice)
HOSPICE_EXPIRED_IDS = (11, 13, 14, 19, 20, 21)
# Paper: "patients aged 18 and older". Ages are 10-year bins; the closest
# conservative approximation is to drop the [0-10) and [10-20) bins.
UNDER_ADULT_BINS = ("[0-10)", "[10-20)")


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "research") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        RESULTS.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(RESULTS / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger
