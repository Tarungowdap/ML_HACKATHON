"""decode(): the function the API calls. Label-free.

Inputs are P1's tables for ONE trip:
  obs  : observations rows (obs_idx, x, y, timestamp, dt_prev, n_candidates, heading_valid, speed ...)
  cand : candidates rows (obs_idx, arc_id, dist_to_arc_m, offset_m, bearing_diff_deg, cand_rank ...)
  graph: TripGraph built from that trip's arcs table
  p_match (optional): P2's probabilities aligned with cand rows
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import HMMConfig
from .emission import emission_logp
from .graph import TripGraph
from .transition import road_distance, search_limit, transition_logp, uturn_mask
from .viterbi import viterbi


@dataclass
class DecodeResult:
    arc_id: np.ndarray                 # per obs (all points, in obs order); -1 = unmatched
    matched: np.ndarray                # bool per obs: decoded from its own candidates
    segment: np.ndarray                # segment id per obs (-1 for points without candidates)
    breaks: list = field(default_factory=list)   # obs_idx where a new segment started
    top1_arc: np.ndarray | None = None # emission-only top-1 per obs ("classifier-only" baseline)
    seconds: float = 0.0
    n_dijkstra_calls: int = 0

    def to_frame(self, obs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"trip_id": obs.trip_id.to_numpy(), "obs_idx": obs.obs_idx.to_numpy(),
                             "pred_arc_id": self.arc_id, "top1_arc_id": self.top1_arc,
                             "matched": self.matched, "segment": self.segment})


def top1_from_scores(obs: pd.DataFrame, cand: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Best-scoring candidate per point; -1 for points without candidates."""
    c = pd.DataFrame({"obs_idx": cand.obs_idx.to_numpy(), "arc_id": cand.arc_id.to_numpy(), "s": score})
    best = c.loc[c.groupby("obs_idx").s.idxmax()].set_index("obs_idx").arc_id
    return best.reindex(obs.obs_idx).fillna(-1).to_numpy(np.int64)


def _fill(arc: np.ndarray, matched: np.ndarray, how: str) -> np.ndarray:
    if how == "none" or matched.all():
        return arc
    s = pd.Series(np.where(matched, arc, np.nan))
    s = s.ffill().bfill() if how == "prev" else s.bfill().ffill()
    return s.fillna(-1).to_numpy(np.int64)


def decode(obs: pd.DataFrame, cand: pd.DataFrame, graph: TripGraph, cfg: HMMConfig,
           p_match: np.ndarray | None = None) -> DecodeResult:
    t0 = time.perf_counter()
    obs = obs.sort_values("obs_idx").reset_index(drop=True)
    order = np.lexsort((cand.cand_rank.to_numpy(), cand.obs_idx.to_numpy()))
    cand = cand.iloc[order].reset_index(drop=True)
    if p_match is not None:
        p_match = np.asarray(p_match)[order]

    le_all = emission_logp(cand, obs, cfg, p_match)
    top1 = top1_from_scores(obs, cand, le_all)

    # group candidate rows per observation; keep only points that have candidates
    oi = cand.obs_idx.to_numpy()
    bounds = np.flatnonzero(np.diff(oi)) + 1
    starts = np.r_[0, bounds]; ends = np.r_[bounds, len(oi)]
    steps = oi[starts] if len(oi) else np.empty(0, np.int64)       # obs_idx of decoded points
    pos = pd.Series(np.arange(len(obs)), index=obs.obs_idx.to_numpy())
    rows = pos.reindex(steps).to_numpy()                             # row in obs for each step
    x = obs.x.to_numpy(float)[rows]; y = obs.y.to_numpy(float)[rows]
    ts = obs.timestamp.to_numpy(float)[rows]
    arc_c = cand.arc_id.to_numpy(np.int64); off_c = cand.offset_m.to_numpy(np.float64)

    log_e = [le_all[s:e] for s, e in zip(starts, ends)]
    sl = [slice(s, e) for s, e in zip(starts, ends)]

    def dt_of(t):            # time since the previous DECODED point (bridges points with 0 candidates)
        return float(ts[t] - ts[t - 1])

    def log_t(t):
        dt = dt_of(t)
        straight = float(np.hypot(x[t] - x[t - 1], y[t] - y[t - 1]))
        road = road_distance(graph, arc_c[sl[t - 1]], off_c[sl[t - 1]], arc_c[sl[t]], off_c[sl[t]],
                             search_limit(dt, cfg), cfg)
        ut = uturn_mask(graph, arc_c[sl[t - 1]], arc_c[sl[t]])
        return transition_logp(road, straight, dt, cfg, ut)

    def broken(T):
        return bool(np.all(T <= cfg.log_unreachable))

    def restart(t):
        return cfg.split_on_gap and dt_of(t) > cfg.gap_s

    st, sg, br = viterbi(log_e, log_t, broken, restart)

    n = len(obs)
    arc = np.full(n, -1, np.int64); seg = np.full(n, -1, np.int64); matched = np.zeros(n, bool)
    if len(steps):
        arc[rows] = arc_c[starts + st]; seg[rows] = sg; matched[rows] = True
    arc = _fill(arc, matched, cfg.fill_missing)
    return DecodeResult(arc_id=arc, matched=matched, segment=seg,
                        breaks=[int(steps[b]) for b in br], top1_arc=top1,
                        seconds=time.perf_counter() - t0, n_dijkstra_calls=graph.n_dijkstra_calls)


def collapse(arcs) -> list[int]:
    """Drop consecutive repeats and -1: per-point arcs -> visited arc sequence."""
    out: list[int] = []
    for a in arcs:
        a = int(a)
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


def expand_route(arc_seq, graph: TripGraph, limit: float = 5000.0) -> list[int]:
    """Make a collapsed arc sequence fully connected by inserting the shortest-path arcs between
    consecutive arcs that do not touch (e.g. short arcs crossed between two 1-s GPS points)."""
    seq = collapse(arc_seq)
    if not seq:
        return []
    out = [seq[0]]
    for b in seq[1:]:
        a = out[-1]
        if not graph.connected(a, b):
            mid = graph.arc_path(a, b, limit)
            if mid:
                out.extend(mid)
        out.append(b)
    return out
