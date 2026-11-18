"""Staged tuning on TRAIN trips (schema 1.0, rebuilt with P1's code). Student and evaluator are never used.

Stage 1: sigma x beta grid (U-turn and heading at current values)
Stage 2: U-turn penalty x heading weight at the best sigma/beta
Selection: highest point-weighted pointwise accuracy; among configs within --tol of it, lowest MEDIAN
per-trip route edit distance. Trips whose candidate ceiling (oracle) is below --min-oracle are excluded
(currently 00000033 and 00000054: broken candidate coverage, reported separately).

python scripts/tune_train.py --data data --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm import HMMConfig, decode  # noqa: E402
from hmm.data import load_split  # noqa: E402
from hmm.evaluate import evaluate_trip  # noqa: E402

_TRIPS = None


def _init(data_dir, ids):
    global _TRIPS
    _TRIPS = load_split(data_dir, "train", ids)


def _run(cfg_dict):
    cfg = HMMConfig(**cfg_dict)
    rows = []
    for td in _TRIPS:
        rows.append(evaluate_trip(td, decode(td.obs, td.cand, td.graph, cfg)))
        td.graph.clear_cache()
    return cfg_dict, rows


def agg(rows) -> dict:
    d = pd.DataFrame(rows)
    w = d.n_labeled
    out = {}
    for k in ["acc_top1", "acc_hmm", "acc_top1_ok", "acc_hmm_ok", "oracle_acc", "cont_hmm"]:
        m = d[k].notna()
        out[k] = float(np.average(d.loc[m, k], weights=w[m])) if m.any() else float("nan")
    out.update(edit_med=float(d.edit_norm_hmm.median()), f1_med=float(d.route_f1_hmm.median()),
               uturns=int(d.n_uturns_hmm.sum()), breaks=int(d.n_breaks.sum()))
    return out


def select(df, tol):
    near = df[df.acc_hmm >= df.acc_hmm.max() - tol].assign(edit_r=lambda x: x.edit_med.round(3))
    return near.sort_values(["edit_r", "acc_hmm"], ascending=[True, False]).iloc[0]   # rounding: true ties -> accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.003)
    ap.add_argument("--min-oracle", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "hmm"))
    ap.add_argument("--config", default=str(ROOT / "hmm_config.json"))
    ap.add_argument("--refine", action="store_true", help="only run stage 3 around the current config")
    a = ap.parse_args()
    sp = Path(a.data) / "splits.json"
    if not sp.exists():
        sp = ROOT / "datasets" / "splits.json"
    splits = json.loads(sp.read_text())
    meta = pd.read_csv(Path(a.data) / "rebuild_meta_train.csv", dtype={"trip_id": str})
    bad = set(meta.trip_id[(1 - meta.miss_rate) < a.min_oracle])
    ids = [t for t in splits["train"] if t not in bad]
    assert not set(ids) & set(splits.get("evaluator", [])) and not set(ids) & set(splits["student"])
    print(f"tuning on {len(ids)} train trips; excluded (oracle < {a.min_oracle}): {sorted(bad)}", flush=True)

    base = HMMConfig.load(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    with Pool(a.workers, initializer=_init, initargs=(a.data, ids)) as pool:
        def stage(name, grid):
            keys = list(grid)
            cfgs = [asdict(replace(best_cfg, **dict(zip(keys, v)))) for v in itertools.product(*grid.values())]
            t0 = time.time()
            for cfg_d, rows in pool.imap_unordered(_run, cfgs):
                r = {"stage": name, **{k: cfg_d[k] for k in ["sigma_m", "beta_m", "uturn_logp", "heading_kappa"]}, **agg(rows)}
                results.append(r)
                print(f"[{name}] σ={r['sigma_m']:g} β={r['beta_m']:g} ut={r['uturn_logp']:g} κ={r['heading_kappa']:g} "
                      f"acc={r['acc_hmm']:.4f} ok={r['acc_hmm_ok']:.4f} edit_med={r['edit_med']:.3f} "
                      f"uturns={r['uturns']}  ({time.time() - t0:.0f}s)", flush=True)
            df = pd.DataFrame(results); df.to_csv(out / "tuning_train.csv", index=False)
            return select(df, a.tol)

        best_cfg = base
        if a.refine:
            prev = pd.read_csv(out / "tuning_train.csv"); results.extend(prev.to_dict("records"))
            b = stage("3_refine", {"sigma_m": [3, 4.5, 6], "beta_m": [1.5, 2, 3], "uturn_logp": [-10, -20]})
            best_cfg = replace(base, sigma_m=float(b.sigma_m), beta_m=float(b.beta_m), uturn_logp=float(b.uturn_logp))
        else:
            b = stage("1_sigma_beta", {"sigma_m": [6, 9, 12, 16, 22], "beta_m": [1, 2, 4, 8, 16]})
            best_cfg = replace(base, sigma_m=float(b.sigma_m), beta_m=float(b.beta_m))
            b = stage("2_uturn_heading", {"uturn_logp": [0, -3, -6, -10], "heading_kappa": [0, 1, 2, 4]})
            best_cfg = replace(best_cfg, uturn_logp=float(b.uturn_logp), heading_kappa=float(b.heading_kappa))

    df = pd.DataFrame(results)
    final = df[(df.sigma_m == best_cfg.sigma_m) & (df.beta_m == best_cfg.beta_m) &
               (df.uturn_logp == best_cfg.uturn_logp) & (df.heading_kappa == best_cfg.heading_kappa)].iloc[0]
    best_cfg.save(a.config, extra={
        "tuned_on": "train", "n_trips": len(ids), "excluded_trips": sorted(bad),
        "metric": f"max acc_hmm; tie-break (within {a.tol}) lowest median per-trip route edit distance",
        "train_scores": {k: float(final[k]) for k in ["acc_top1", "acc_hmm", "acc_hmm_ok", "oracle_acc",
                                                       "cont_hmm", "edit_med", "f1_med", "uturns", "breaks"]}})
    print("selected:", asdict(best_cfg)); print(final.to_dict())


if __name__ == "__main__":
    main()
