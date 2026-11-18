"""Rebuild schema-1.0 observations/candidates with P1's OWN code (pathsnap.py), from the repo tables.

Why: the repo has schema-1.0 observations, nodes, arcs and routes, but candidates_student is the old v0
file and candidates_train is in Git LFS. Raw track = observations (lon, lat, timestamp in orig_row
order); P1's cleaning dropped 0 rows on every trip, so this reproduces P1's pipeline exactly.

python scripts/rebuild_v1.py --repo /path/to/ML_HACKATHON --splits student train --out data
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hmm.p1 import load_pathsnap  # noqa: E402
ps = load_pathsnap()


def rebuild_split(repo: Path, split: str, out: Path):
    P = repo / "datasets" / "processed"
    obs_all = pd.read_parquet(P / f"observations_{split}.parquet",
                              columns=["trip_id", "orig_row", "lon", "lat", "timestamp"])
    nodes_all = pd.read_parquet(P / f"nodes_{split}.parquet")
    arcs_all = pd.read_parquet(P / f"arcs_{split}.parquet", columns=["trip_id", "arc_id", "from_node", "to_node"])
    routes_all = pd.read_parquet(P / f"routes_{split}.parquet")
    O, C, A, N, R, M = [], [], [], [], [], []
    for tid in sorted(obs_all.trip_id.unique()):
        t0 = time.time()
        track = (obs_all[obs_all.trip_id == tid].sort_values("orig_row")
                 .rename(columns={"lon": "longitude", "lat": "latitude"})[["longitude", "latitude", "timestamp"]])
        nodes = (nodes_all[nodes_all.trip_id == tid].sort_values("node_id")
                 .rename(columns={"lon": "longitude", "lat": "latitude"}).reset_index(drop=True))
        assert np.array_equal(nodes.node_id.to_numpy(), np.arange(len(nodes)))
        arcs = (arcs_all[arcs_all.trip_id == tid].sort_values("arc_id")
                .rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"}).reset_index(drop=True))
        route = routes_all[routes_all.trip_id == tid].sort_values("route_pos").arc_id.to_numpy(np.int64)
        obs, cand, info = ps.build_observations_and_candidates(tid, track, nodes, arcs)
        obs, cand, lab = ps.attach_labels(obs, cand, nodes, arcs, route if len(route) else None, info)
        a, n, r = ps.graph_tables(tid, nodes, arcs, route, info["lat0"], info["lon0"])
        O.append(obs); C.append(cand); A.append(a); N.append(n); R.append(r)
        M.append({"trip_id": tid, "lat0": info["lat0"], "lon0": info["lon0"], "n_obs": info["n_obs"],
                  "n_zero_cand": info["n_zero_cand"], "n_truncated": info["n_truncated"],
                  "n_miss": lab["n_miss"], "miss_rate": lab["n_miss"] / max(lab["n_labeled"], 1),
                  "build_s": round(time.time() - t0, 2)})
        print(f"{split} {tid}: {info['n_obs']} pts, {len(cand)} cands, miss {M[-1]['miss_rate']:.3f}, {M[-1]['build_s']}s",
              flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(O).to_parquet(out / f"observations_{split}.parquet", index=False)
    pd.concat(C).to_parquet(out / f"candidates_{split}.parquet", index=False)
    pd.concat(A).to_parquet(out / f"arcs_{split}.parquet", index=False)
    pd.concat(N).to_parquet(out / f"nodes_{split}.parquet", index=False)
    pd.concat(R).to_parquet(out / f"routes_{split}.parquet", index=False)
    pd.DataFrame(M).to_csv(out / f"rebuild_meta_{split}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT), help="repo root holding datasets/processed (default: this repo)")
    ap.add_argument("--splits", nargs="+", default=["student", "train"])
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    for s in a.splits:
        rebuild_split(Path(a.repo), s, Path(a.out))


if __name__ == "__main__":
    main()
