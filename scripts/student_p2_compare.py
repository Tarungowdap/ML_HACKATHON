"""Classifier-only vs HMM on the 10 STUDENT trips, with P2's current model.

Uses the repo layout (datasets/processed + models/). Everything is scored against schema-1.0 labels
(observations.true_arc_id). Student trips have no routes in the repo yet, so route metrics are skipped.

python scripts/student_p2_compare.py --repo /path/to/ML_HACKATHON
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from hmm import HMMConfig, TripGraph, decode  # noqa: E402
from hmm.data import TripData  # noqa: E402
from hmm.evaluate import evaluate_trip, summarize  # noqa: E402
from hmm.p2_adapter import P2Model, relabel, v0_to_hmm_candidates  # noqa: E402


def load_student(repo: Path):
    P = repo / "datasets" / "processed"
    obs = pd.read_parquet(P / "observations_student.parquet")
    arcs = pd.read_parquet(P / "arcs_student.parquet")
    c0 = pd.read_parquet(P / "candidates_student.parquet")
    model = P2Model(repo / "models" / "final_tuned_raw.joblib", repo / "models" / "features_raw.json")
    c0["p_match"] = model.predict(c0, obs)                    # predicted on the features P2 trained on
    cand = relabel(v0_to_hmm_candidates(c0, arcs), obs)
    trips = []
    for tid, o in obs.groupby("trip_id"):
        a = arcs[arcs.trip_id == tid].sort_values("arc_id").reset_index(drop=True)
        c = cand[cand.trip_id == tid].sort_values(["obs_idx", "cand_rank"]).reset_index(drop=True)
        g = TripGraph(a, n_nodes=int(max(a.from_node.max(), a.to_node.max())) + 1)
        trips.append(TripData(tid, o.sort_values("obs_idx").reset_index(drop=True), c, a, None, None, g))
    return trips


def calibration(trips) -> pd.DataFrame:
    c = pd.concat([t.cand for t in trips])
    bins = np.array([0, .01, .05, .1, .2, .4, .6, .8, .9, .95, .99, 1.0])
    c["bin"] = pd.cut(c.p_match, bins, include_lowest=True)
    return c.groupby("bin", observed=True).agg(n=("label", "size"), mean_p=("p_match", "mean"),
                                                frac_true=("label", "mean")).reset_index()


def run(trips, cfg, use_p=True):
    rows = []
    for td in trips:
        p = td.cand.p_match.to_numpy() if use_p and cfg.emission != "gaussian" else None
        rows.append(evaluate_trip(td, decode(td.obs, td.cand, td.graph, cfg, p_match=p)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "hmm_config.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "hmm"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trips = load_student(Path(a.repo))
    base = HMMConfig.load(a.config)

    cal = calibration(trips); cal.to_csv(out / "p2_calibration_student.csv", index=False)
    print("calibration (student, schema-1.0 labels):\n", cal.round(3).to_string(index=False))

    variants = {
        "gaussian": replace(base, emission="gaussian"),
        "pmatch": replace(base, emission="pmatch"),
    }
    for w in (0.25, 0.5, 0.75):
        variants[f"blend w={w}"] = replace(base, emission="blend", w_pmatch=w)
    for k in (0.0,):
        variants["pmatch, no heading term"] = replace(base, emission="pmatch", heading_kappa=k)

    res, per_trip = [], []
    for name, cfg in variants.items():
        rows = run(trips, cfg)
        s = summarize(rows); s["variant"] = name; res.append(s)
        for r in rows:
            r["variant"] = name; per_trip.append(r)
    df = pd.DataFrame(res)
    # classifier-only P2 = top-1 of p_match (heading term off so it is purely P2's model)
    p2_only = summarize(run(trips, replace(base, emission="pmatch", heading_kappa=0.0)))
    cols = ["variant", "acc_top1", "acc_hmm", "acc_top1_ok", "acc_hmm_ok", "oracle_acc", "cont_top1", "cont_hmm",
            "n_uturns_hmm", "n_breaks"]
    df.to_csv(out / "student_compare.csv", index=False)
    pd.DataFrame(per_trip).to_csv(out / "student_compare_per_trip.csv", index=False)
    print(f"\nP2 classifier-only top-1 (schema-1.0 labels): {p2_only['acc_top1']:.4f}  "
          f"(ok-labels {p2_only['acc_top1_ok']:.4f})")
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
