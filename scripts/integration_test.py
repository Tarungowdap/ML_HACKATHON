"""End-to-end test of P3's part exactly as the API will run it on Laptop 1.

For each trip:  raw .track/.nodes/.arcs  ->  match_trip() (label-free: P1 builder + HMM)  ->  route for the UI
                then, separately, score_trip() reads .route (server-side only) and computes accuracy.

Modes
  --raw <folder>        trip folders with raw files (e.g. evaluator_test/). Decoding only by default:
                        evaluator accuracy is for the final demo, never for choosing anything.
                        Add --score only at the real evaluation.
  --student-parquets    rebuild raw inputs for the 10 student trips from data/*_student.parquet and score them;
                        the scores must equal results/hmm/student_eval.csv (checks the API path = the report path).

python scripts/integration_test.py --raw path/to/evaluator_test
python scripts/integration_test.py --student-parquets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm import HMMConfig  # noqa: E402
from hmm.api import match_trip, score_trip  # noqa: E402


def read_raw(folder: Path, tid: str):
    rd = lambda p, n: pd.read_csv(p, header=None, sep=r"\s+", names=n, usecols=range(len(n)))  # noqa: E731
    track = rd(folder / f"{tid}.track", ["longitude", "latitude", "timestamp"])
    nodes = rd(folder / f"{tid}.nodes", ["longitude", "latitude"])
    arcs = rd(folder / f"{tid}.arcs", ["from_node_id", "to_node_id"])
    rf = folder / f"{tid}.route"
    route = rd(rf, ["arc_id"]).arc_id.to_numpy(np.int64) if rf.exists() and rf.stat().st_size else None
    return track, nodes, arcs, route


def student_inputs():
    D = ROOT / "data"
    obs = pd.read_parquet(D / "observations_student.parquet", columns=["trip_id", "orig_row", "lon", "lat", "timestamp"])
    nodes = pd.read_parquet(D / "nodes_student.parquet", columns=["trip_id", "node_id", "lon", "lat"])
    arcs = pd.read_parquet(D / "arcs_student.parquet", columns=["trip_id", "arc_id", "from_node", "to_node"])
    routes = pd.read_parquet(D / "routes_student.parquet")
    for tid in sorted(obs.trip_id.unique()):
        track = (obs[obs.trip_id == tid].sort_values("orig_row")
                 .rename(columns={"lon": "longitude", "lat": "latitude"})[["longitude", "latitude", "timestamp"]])
        n = nodes[nodes.trip_id == tid].sort_values("node_id").rename(columns={"lon": "longitude", "lat": "latitude"})
        a = (arcs[arcs.trip_id == tid].sort_values("arc_id")
             .rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})[["from_node_id", "to_node_id"]])
        r = routes[routes.trip_id == tid].sort_values("route_pos").arc_id.to_numpy(np.int64)
        yield tid, track.reset_index(drop=True), n[["longitude", "latitude"]].reset_index(drop=True), a.reset_index(drop=True), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--student-parquets", action="store_true")
    ap.add_argument("--score", action="store_true", help="score raw trips against .route (final evaluation only)")
    ap.add_argument("--config", default=str(ROOT / "hmm_config.json"))
    a = ap.parse_args()
    cfg = HMMConfig.load(a.config)
    print(f"config: sigma={cfg.sigma_m} beta={cfg.beta_m} uturn={cfg.uturn_logp} kappa={cfg.heading_kappa} "
          f"emission={cfg.emission}")

    if a.raw:
        root = Path(a.raw)
        trips = [(p.name, *read_raw(p, p.name)) for p in sorted(root.iterdir()) if p.is_dir()]
        score = a.score
    elif a.student_parquets:
        trips = list(student_inputs()); score = True
    else:
        ap.error("pass --raw <folder> or --student-parquets")

    rows = []
    for tid, track, nodes, arcs, route in trips:
        t0 = time.perf_counter()
        out = match_trip(track, nodes, arcs, tid, cfg)               # what the API sends to the UI
        secs = time.perf_counter() - t0
        json.dumps({"polylines": out["polylines"], "geojson": out["geojson"]})   # must be JSON-serialisable
        row = {"trip_id": tid, "points": len(track), "arcs": len(arcs), "seconds": round(secs, 2),
               "matched_pts": float(out["points"].matched.mean()), "route_arcs": len(out["route_arcs"]),
               "polylines": len(out["polylines"]), "segments": out["n_segments"], "has_route_file": route is not None}
        if score and route is not None:
            s = score_trip(track, nodes, arcs, route, tid, out["points"].arc_id.to_numpy(), out["route_arcs"])
            row.update(accuracy=round(s["accuracy"], 4), accuracy_ok=round(s["accuracy_ok_labels"], 4),
                       route_f1=round(s["route_f1"], 4))
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} trips OK, max {df.seconds.max():.2f} s, mean {df.seconds.mean():.2f} s per trip")
    if a.student_parquets:
        ref = pd.read_csv(ROOT / "results" / "hmm" / "student_eval.csv", dtype={"trip_id": str}).set_index("trip_id")
        diff = (df.set_index("trip_id").accuracy - ref.acc_hmm.round(4)).abs().max()
        print(f"API-path accuracy vs report (results/hmm/student_eval.csv): max difference {diff:.4f}")
        assert diff < 1e-3, "API path disagrees with the evaluated pipeline"
        w = df.points
        print(f"student accuracy via API path: {np.average(df.accuracy, weights=w):.4f}")


if __name__ == "__main__":
    main()
