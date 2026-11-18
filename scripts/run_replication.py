"""LEADERBOARD A - replication of Emi-Johnson & Nkrumah (2025).

Paper protocol as stated: exclude hospice/expired discharges, adults only,
80/20 stratified split, mean imputation + standardisation, one-hot encoding,
class weighting, grid search with 5-fold CV, models LR / RF / XGBoost / DNN.
No SMOTE (the paper explicitly did not use it).

Runs the replication under
  * protocol 'encounter' (the paper's encounter-level stratified split) and
  * protocol 'patient'   (patient-grouped split, realistic generalisation),
plus a sensitivity run on the FULL 101,766-row file (no exclusions), which is
what the paper's reported test-set size (20,354) actually corresponds to.

Usage:  python scripts/run_replication.py [--skip-sensitivity] [--models logreg,rf,xgb,dnn]
"""
from __future__ import annotations

import argparse
import json
import time

from common import (C, estimator_factory, get_cache, load_cohort, log, make_split, run_experiment,
                    save_leaderboard, save_probas, split_summary)
from src.research.models import replication_candidates

KINDS = {"logreg": "LR", "rf": "RF", "xgb": "XGB", "dnn": "DNN"}


def run_protocol(frame, protocol, kinds, out_root, fixed_params=None, tag=""):
    split = make_split(frame, protocol)
    summ = split_summary(frame, split)
    log.info(f"=== Replication | protocol={protocol}{tag} | {json.dumps(summ)}")
    cache = get_cache(frame, split, blocks=None)
    log.info(f"FS-R matrix: {cache.X_train_t.shape[1]} features after one-hot encoding")
    out_dir = out_root / f"{protocol}{tag}"
    test_groups = frame["patient_nbr"].values[split.test_idx]
    records = []
    for kind in kinds:
        cands = [fixed_params[kind]] if fixed_params else replication_candidates(kind)
        threads = 4 if kind in ("dnn",) else 1
        n_jobs = 1 if kind == "dnn" else 4
        make_est = estimator_factory(kind, cache.y_train, threads=threads)
        rec = run_experiment(f"A_{KINDS[kind]}{tag}", "A_replication", protocol + tag, make_est, cands, cache,
                             out_dir, test_groups=test_groups,
                             test_ids=frame["encounter_id"].values[split.test_idx], n_jobs=n_jobs,
                             notes="FS-R, class weighting, grid search 5-fold CV (ROC-AUC)",
                             extra_config={"feature_set": "FS-R", "split": summ})
        records.append(rec)
    save_leaderboard(records, out_dir / "leaderboard_A")
    save_probas(records, frame, split, out_dir / "test_probas.npz")
    (out_dir / "split_summary.json").write_text(json.dumps(summ, indent=2))
    (out_dir / "feature_names.json").write_text(json.dumps(cache.feature_names))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--models", default="logreg,rf,xgb,dnn")
    ap.add_argument("--protocols", default="encounter,patient")
    args = ap.parse_args()
    C.set_seeds()
    kinds = args.models.split(",")
    out_root = C.RESULTS / "replication"
    t0 = time.time()

    coh, frame = load_cohort(exclude=True, adults=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cohort_flow.json").write_text(json.dumps(coh.flow, indent=2))
    log.info(f"Cohort flow: {coh.flow}")

    best = {}
    for protocol in args.protocols.split(","):
        recs = run_protocol(frame, protocol, kinds, out_root)
        if protocol == "encounter":
            best = {k: r["best_params"] for k, r in zip(kinds, recs)}

    if not args.skip_sensitivity and best:
        # Sensitivity: the paper's test size (20,354 = 20% of 101,766) implies no rows were removed.
        coh_all, frame_all = load_cohort(exclude=False, adults=False)
        (out_root / "cohort_flow_noexclusion.json").write_text(json.dumps(coh_all.flow, indent=2))
        run_protocol(frame_all, "encounter", kinds, out_root, fixed_params=best, tag="_noexcl")
    log.info(f"Replication finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
