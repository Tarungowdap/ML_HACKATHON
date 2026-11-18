"""One call for P4's API: raw trip files -> matched route. Label-free.

    from hmm.api import match_trip
    out = match_trip(track, nodes, arcs, trip_id, cfg=HMMConfig.load("hmm_config.json"),
                     p_match_fn=p2_model.predict_proba_for)   # optional, P2
    out["polylines"]   # [[ [lat, lon], ... ], ...] for Leaflet
    out["points"]      # per GPS point: obs_idx, lat, lon, arc_id

track: longitude, latitude, timestamp | nodes: longitude, latitude | arcs: from_node_id, to_node_id
(exactly what pathsnap.load_trip(raw_root=...) returns; the route is never used here).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import HMMConfig
from .decode import decode, expand_route
from .graph import TripGraph


def score_trip(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, route, trip_id: str,
               pred_arc_id: np.ndarray, route_arcs: list[int] | None = None) -> dict:
    """SERVER-SIDE scoring for the API (the evaluator's 'accuracy' step). Uses the reference .route, so it
    must run AFTER match_trip and its result must never feed back into decoding.
    Labels come from P1's own attach_labels (monotone alignment), so the API scores exactly like our report."""
    from .p1 import load_pathsnap
    ps = load_pathsnap()
    from .evaluate import levenshtein, route_prf

    arcs = arcs.copy()
    if "arc_id" not in arcs:
        arcs["arc_id"] = np.arange(len(arcs), dtype=np.int32)
    obs, cand, info = ps.build_observations_and_candidates(trip_id, track, nodes, arcs)
    obs, _, _ = ps.attach_labels(obs, cand, nodes, arcs, np.asarray(route, dtype=np.int64), info)
    true = obs.true_arc_id.to_numpy()
    pred = np.asarray(pred_arc_id)
    ok = (obs.label_quality == "ok").to_numpy()
    out = {"trip_id": trip_id, "n_points": int(len(obs)),
           "accuracy": float((pred == true).mean()),
           "accuracy_ok_labels": float((pred[ok] == true[ok]).mean()) if ok.any() else None,
           "n_ok_labels": int(ok.sum())}
    if route_arcs is not None:
        p, r, f = route_prf(route_arcs, list(route))
        out.update(route_precision=p, route_recall=r, route_f1=f,
                   route_edit_norm=levenshtein(route_arcs, list(route)) / max(len(route), 1))
    return out


def match_trip(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, trip_id: str,
               cfg: HMMConfig, p_match_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None) -> dict:
    from .p1 import load_pathsnap
    ps = load_pathsnap()                        # P1's shared module (schema 1.0)
    from viz.coords import arcs_to_polylines, route_geojson

    arcs = arcs.copy()
    if "arc_id" not in arcs:
        arcs["arc_id"] = np.arange(len(arcs), dtype=np.int32)
    obs, cand, info = ps.build_observations_and_candidates(trip_id, track, nodes, arcs)
    arcs_tbl, _, _ = ps.graph_tables(trip_id, nodes, arcs, None, info["lat0"], info["lon0"])
    g = TripGraph(arcs_tbl, n_nodes=len(nodes))

    use_p2 = p_match_fn is not None and cfg.emission in ("pmatch", "blend")
    p = p_match_fn(obs, cand) if use_p2 else None
    res = decode(obs, cand, g, cfg, p_match=p)
    route = expand_route(res.arc_id, g)
    return {
        "trip_id": trip_id,
        "points": pd.DataFrame({"obs_idx": obs.obs_idx, "lat": obs.lat, "lon": obs.lon,
                                "arc_id": res.arc_id, "matched": res.matched}),
        "route_arcs": route,
        "polylines": arcs_to_polylines(route, arcs_tbl),
        "geojson": route_geojson(route, arcs_tbl, {"trip_id": trip_id}),
        "n_segments": int(res.segment.max() + 1) if res.matched.any() else 0,
        "breaks": res.breaks,
        "seconds_decode": res.seconds,
        "build_info": info,
    }
