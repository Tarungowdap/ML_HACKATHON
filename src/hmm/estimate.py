"""Data-driven starting values for sigma and beta (Newson & Krumm 2009, section 4), from LABELLED trips.

sigma = 1.4826 * median(GPS-to-true-arc distance)            (robust std of GPS noise)
beta  = median(|road_distance - straight_distance|) / ln 2   over consecutive points on the true path
"""
from __future__ import annotations

import numpy as np

from .config import HMMConfig
from .transition import road_distance, search_limit


def estimate_sigma(trips) -> float:
    d = np.concatenate([t.obs.true_dist_m.to_numpy() for t in trips])
    return float(1.4826 * np.nanmedian(d))


def estimate_beta(trips, cfg: HMMConfig | None = None, max_dt: float = 1.5) -> tuple[float, np.ndarray]:
    """Uses only consecutive pairs with dt <= max_dt where both points have their true arc as a candidate."""
    cfg = cfg or HMMConfig()
    diffs = []
    for td in trips:
        true_c = td.cand[td.cand.label == 1].set_index("obs_idx")
        o = td.obs.set_index("obs_idx")
        idx = o.index.to_numpy()
        for i0, i1 in zip(idx[:-1], idx[1:]):
            if i0 not in true_c.index or i1 not in true_c.index:
                continue
            dt = o.timestamp[i1] - o.timestamp[i0]
            if dt > max_dt or dt <= 0:
                continue
            a, b = true_c.loc[i0], true_c.loc[i1]
            road = road_distance(td.graph, [a.arc_id], [a.offset_m], [b.arc_id], [b.offset_m],
                                 search_limit(dt, cfg), cfg)[0, 0]
            straight = np.hypot(o.x[i1] - o.x[i0], o.y[i1] - o.y[i0])
            if np.isfinite(road):
                diffs.append(abs(road - straight))
    diffs = np.asarray(diffs)
    return float(np.median(diffs) / np.log(2)), diffs
