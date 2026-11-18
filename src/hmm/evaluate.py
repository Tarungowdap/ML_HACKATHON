"""Evaluation (train/student data only; never imported by the API path).

Metrics
  pointwise accuracy      share of labelled points whose predicted arc == true_arc_id
  accuracy_ok             same, only on points with label_quality == 'ok'
  oracle (ceiling)        share of labelled points whose true arc is among the candidates (1 - miss rate)
  continuity              share of consecutive predicted arcs (after collapsing repeats) that are connected
  gt_continuity           same for the reference .route (continuity ceiling)
  route P/R/F1            arc sets of the expanded predicted route vs .route
  edit_dist_norm          Levenshtein(predicted route, .route) / len(.route)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decode import DecodeResult, collapse, expand_route
from .graph import TripGraph


def continuity(seq, g: TripGraph) -> float:
    s = collapse(seq)
    if len(s) < 2:
        return 1.0
    return float(np.mean([g.connected(a, b) for a, b in zip(s[:-1], s[1:])]))


def n_uturns(seq, g: TripGraph) -> int:
    s = collapse(seq)
    return int(sum(g.from_node[a] == g.to_node[b] and g.to_node[a] == g.from_node[b] for a, b in zip(s[:-1], s[1:])))


def levenshtein(a, b) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return int(Levenshtein.distance(list(a), list(b)))
    except ImportError:
        pass
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def route_prf(pred, ref) -> tuple[float, float, float]:
    P, R = set(pred), set(ref)
    if not P or not R:
        return 0.0, 0.0, 0.0
    tp = len(P & R)
    p, r = tp / len(P), tp / len(R)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def pointwise(pred: np.ndarray, obs: pd.DataFrame, mask: np.ndarray | None = None) -> float:
    lab = obs.true_arc_id.to_numpy() >= 0
    if mask is not None:
        lab &= mask
    return float((pred[lab] == obs.true_arc_id.to_numpy()[lab]).mean()) if lab.any() else float("nan")


def error_types(pred: np.ndarray, obs: pd.DataFrame, g: TripGraph) -> dict:
    """Classify wrong points: reverse direction of the true road / neighbouring arc (shares a node) /
    other (e.g. parallel street) / unmatched."""
    true = obs.true_arc_id.to_numpy()
    wrong = (true >= 0) & (pred != true)
    out = {"reverse_dir": 0, "adjacent_arc": 0, "other": 0, "unmatched": 0}
    for p, t in zip(pred[wrong], true[wrong]):
        if p < 0:
            out["unmatched"] += 1
        elif g.from_node[p] == g.to_node[t] and g.to_node[p] == g.from_node[t]:
            out["reverse_dir"] += 1
        elif len({g.from_node[p], g.to_node[p]} & {g.from_node[t], g.to_node[t]}):
            out["adjacent_arc"] += 1
        else:
            out["other"] += 1
    return out


def evaluate_trip(td, res: DecodeResult) -> dict:
    obs, g = td.obs, td.graph
    ok = (obs.label_quality == "ok").to_numpy()
    hit = td.cand.groupby("obs_idx").label.max().reindex(obs.obs_idx).fillna(0).to_numpy() > 0
    lab = obs.true_arc_id.to_numpy() >= 0
    route = list(td.route) if td.route is not None else []
    exp_hmm = expand_route(res.arc_id, g)
    out = {
        "trip_id": td.trip_id, "n_obs": len(obs), "n_labeled": int(lab.sum()),
        "acc_top1": pointwise(res.top1_arc, obs), "acc_hmm": pointwise(res.arc_id, obs),
        "acc_top1_ok": pointwise(res.top1_arc, obs, ok), "acc_hmm_ok": pointwise(res.arc_id, obs, ok),
        "oracle_acc": float(hit[lab].mean()) if lab.any() else float("nan"),
        "cont_top1": continuity(res.top1_arc, g), "cont_hmm": continuity(res.arc_id, g),
        "cont_hmm_expanded": continuity(exp_hmm, g),
        "n_uturns_hmm": n_uturns(res.arc_id, g),
        "gt_continuity": continuity(route, g) if route else float("nan"),
        "n_segments": int(res.segment.max() + 1) if (res.segment >= 0).any() else 0,
        "n_breaks": len(res.breaks), "n_unmatched_pts": int((~res.matched).sum()),
        "seconds": res.seconds,
    }
    if route:
        p, r, f = route_prf(exp_hmm, route)
        p1, r1, f1 = route_prf(expand_route(res.top1_arc, g), route)
        out.update(route_p_hmm=p, route_r_hmm=r, route_f1_hmm=f, route_f1_top1=f1,
                   edit_norm_hmm=levenshtein(exp_hmm, route) / len(route),
                   edit_norm_top1=levenshtein(collapse(res.top1_arc), route) / len(route))
    for k, v in error_types(res.arc_id, obs, g).items():
        out[f"err_hmm_{k}"] = v
    for k, v in error_types(res.top1_arc, obs, g).items():
        out[f"err_top1_{k}"] = v
    return out


def summarize(rows: list[dict]) -> dict:
    """Point-weighted averages across trips."""
    df = pd.DataFrame(rows)
    w = df.n_labeled.clip(lower=1)
    num = [c for c in df.columns if c not in ("trip_id",) and pd.api.types.is_numeric_dtype(df[c])]
    out = {c: float(np.average(df[c].fillna(0), weights=w)) for c in num if c.startswith(("acc_", "oracle", "cont_", "gt_", "route_", "edit_"))}
    out.update({c: int(df[c].sum()) for c in num if c.startswith(("n_", "err_"))})
    out["n_trips"] = len(df)
    return out
