"""Emission log-probabilities: "does this GPS point fit this arc?".

All functions return one log-score per candidate row (same order as `cand`).
Only relative values within a point matter to Viterbi, so constants are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HMMConfig


def log_gaussian(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """log N(d; 0, sigma) up to a constant: -d^2 / (2 sigma^2)."""
    d = np.asarray(dist_m, dtype=np.float64)
    return -0.5 * (d / sigma_m) ** 2


def log_heading(cand: pd.DataFrame, obs: pd.DataFrame, cfg: HMMConfig) -> np.ndarray:
    """kappa*(cos(bearing_diff) - 1): 0 when aligned, -2*kappa when opposite.
    Only applied where the point's own heading is trusted and it is moving.
    Separates the two directions of the same road, which have identical distance."""
    if cfg.heading_kappa <= 0:
        return np.zeros(len(cand))
    o = obs.set_index("obs_idx")
    use = (o.heading_valid & (o.speed >= cfg.heading_min_speed)).reindex(cand.obs_idx).to_numpy()
    bd = np.radians(cand.bearing_diff_deg.to_numpy(np.float64))
    return np.where(use, cfg.heading_kappa * (np.cos(bd) - 1.0), 0.0)


def emission_logp(cand: pd.DataFrame, obs: pd.DataFrame, cfg: HMMConfig,
                  p_match: np.ndarray | None = None) -> np.ndarray:
    """Main entry. p_match: P2's probabilities aligned with `cand` rows (needed for 'pmatch'/'blend')."""
    lg = log_gaussian(cand.dist_to_arc_m.to_numpy(), cfg.sigma_m)
    if cfg.emission == "gaussian":
        out = lg
    else:
        if p_match is None:
            raise ValueError(f"emission='{cfg.emission}' needs p_match from P2's model")
        lp = np.log(np.clip(np.asarray(p_match, dtype=np.float64), cfg.pmatch_floor, 1.0))
        if cfg.emission == "pmatch":
            out = lp
        elif cfg.emission == "blend":
            out = cfg.w_pmatch * lp + (1.0 - cfg.w_pmatch) * lg
        else:
            raise ValueError(cfg.emission)
    return out + log_heading(cand, obs, cfg)
