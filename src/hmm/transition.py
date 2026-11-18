"""Transition log-probabilities (Newson & Krumm 2009): "is this move physically possible?"

log P(a -> b) = -| road_distance(a, b) - straight_distance(z_{t-1}, z_t) | / beta_eff

road_distance:
  * same arc, moving forward (or backwards by less than GPS noise): |offset_b - offset_a|
  * otherwise: (rest of arc a) + shortest path(to_node(a) -> from_node(b)) + (distance into b)
    (if b starts where a ends the shortest path is 0)
"""
from __future__ import annotations

import numpy as np

from .config import HMMConfig
from .graph import TripGraph


def search_limit(dt: float, cfg: HMMConfig) -> float:
    return float(min(cfg.search_max_m, cfg.search_base_m + cfg.vmax_mps * max(dt, 1.0)))


def road_distance(g: TripGraph, arc_p, off_p, arc_c, off_c, limit: float, cfg: HMMConfig) -> np.ndarray:
    """(k_prev, k_cur) matrix of road distances in metres (inf = no path within `limit`)."""
    arc_p = np.asarray(arc_p, np.int64); arc_c = np.asarray(arc_c, np.int64)
    off_p = np.asarray(off_p, np.float64); off_c = np.asarray(off_c, np.float64)
    rest_p = np.maximum(g.length[arc_p] - off_p, 0.0)
    S = g.dist_from(g.to_node[arc_p], limit, g.from_node[arc_c])             # node -> node
    road = rest_p[:, None] + S + off_c[None, :]
    fwd = off_c[None, :] - off_p[:, None]
    same = (arc_p[:, None] == arc_c[None, :]) & (fwd >= -cfg.same_arc_back_tol_m)
    road = np.where(same, np.abs(fwd), road)
    return road


def uturn_mask(g: TripGraph, arc_p, arc_c) -> np.ndarray:
    """True where arc_c is the reverse twin of arc_p (same road, opposite direction)."""
    arc_p = np.asarray(arc_p, np.int64); arc_c = np.asarray(arc_c, np.int64)
    return ((g.to_node[arc_p][:, None] == g.from_node[arc_c][None, :])
            & (g.from_node[arc_p][:, None] == g.to_node[arc_c][None, :]))


def transition_logp(road: np.ndarray, straight: float, dt: float, cfg: HMMConfig,
                    uturn: np.ndarray | None = None) -> np.ndarray:
    beta = cfg.beta_m * (max(dt, 1.0) if cfg.beta_dt_scale else 1.0)
    lp = -np.abs(road - straight) / beta
    if uturn is not None and cfg.uturn_logp:
        lp = lp + np.where(uturn, cfg.uturn_logp, 0.0)
    lp = np.maximum(lp, cfg.log_floor)
    return np.where(np.isfinite(road), lp, cfg.log_unreachable)
