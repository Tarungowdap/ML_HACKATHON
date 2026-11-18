"""Bridge between P2's model and the HMM.

Two jobs:
1. `v0_to_hmm_candidates`: TEMPORARY. Rebuild the schema-1.0 columns the HMM needs from P1's old v0
   candidate table (which P2's current model was trained on), using the trip's arcs table:
     dist_to_arc_m    = perp_dist_m
     offset_m         = proj_ratio_t * arc_length_m
     from_node/to_node from arcs
     cand_rank        = order by (perp_dist_m, arc_id) per point
     bearing_diff_deg = degrees(heading_diff_rad)
   Delete once P1 ships schema-1.0 candidates for every split and P2 retrains on them.
2. `attach_p_match`: join P2's predictions onto candidate rows by (trip_id, obs_idx, arc_id), never by
   row order, and fail loudly if any candidate is left without a probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["trip_id", "obs_idx", "arc_id"]


def v0_to_hmm_candidates(cand_v0: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    a = arcs[["trip_id", "arc_id", "from_node", "to_node", "length_m"]]
    c = cand_v0.merge(a, on=["trip_id", "arc_id"], how="left", validate="many_to_one")
    if c.from_node.isna().any():
        raise ValueError("v0 candidates reference arcs missing from the arcs table")
    c["dist_to_arc_m"] = c.perp_dist_m.astype(np.float32)
    c["offset_m"] = (c.proj_ratio_t * c.length_m).astype(np.float32)
    c["arc_length_m"] = c.length_m.astype(np.float32)
    c["bearing_diff_deg"] = np.degrees(c.heading_diff_rad).astype(np.float32)
    c = c.sort_values(["trip_id", "obs_idx", "dist_to_arc_m", "arc_id"], kind="stable")
    c["cand_rank"] = c.groupby(["trip_id", "obs_idx"]).cumcount().astype(np.int16)
    c["from_node"] = c.from_node.astype(np.int32); c["to_node"] = c.to_node.astype(np.int32)
    return c.reset_index(drop=True)


def relabel(cand: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Overwrite `label` with the schema-1.0 reference arc (obs.true_arc_id) so every method is scored
    against the same labels."""
    t = obs[["trip_id", "obs_idx", "true_arc_id"]]
    c = cand.drop(columns=["label"], errors="ignore").merge(t, on=["trip_id", "obs_idx"], how="left")
    c["label"] = (c.arc_id == c.true_arc_id).astype(np.int8)
    return c.drop(columns=["true_arc_id"])


class P2Model:
    """Wraps P2's saved model + feature list. predict(cand, obs) -> probabilities aligned with cand rows."""

    def __init__(self, model_path: str | Path, feats_path: str | Path):
        import joblib
        self.model = joblib.load(model_path)
        self.feats = json.loads(Path(feats_path).read_text())

    def predict(self, cand: pd.DataFrame, obs: pd.DataFrame | None = None, eps: float = 1e-6) -> np.ndarray:
        X = cand
        missing = [f for f in self.feats if f not in X.columns]
        if missing and obs is not None:
            extra = [f for f in missing if f in obs.columns]
            X = X.merge(obs[["trip_id", "obs_idx"] + extra], on=["trip_id", "obs_idx"], how="left")
        still = [f for f in self.feats if f not in X.columns]
        if still:
            raise KeyError(f"model needs features not in the candidate table: {still}")
        p = self.model.predict_proba(X[self.feats].to_numpy(np.float32))[:, 1]
        return np.clip(p, eps, 1 - eps)


def attach_p_match(cand: pd.DataFrame, preds: pd.DataFrame) -> np.ndarray:
    """preds: DataFrame with trip_id, obs_idx, arc_id, p_match (e.g. P2's pmatch_*.parquet).
    Returns p_match aligned with cand rows. Raises if any candidate has no prediction."""
    m = cand[KEY].merge(preds[KEY + ["p_match"]], on=KEY, how="left", validate="one_to_one")
    if m.p_match.isna().any():
        n = int(m.p_match.isna().sum())
        raise ValueError(f"{n} candidate rows have no p_match (different candidate set / schema?)")
    return m.p_match.to_numpy(np.float64)


def normalise_per_point(cand: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """Turn independent binary probabilities into a distribution over each point's candidates."""
    s = pd.Series(p).groupby(cand.obs_idx.to_numpy()).transform("sum").to_numpy()
    return p / np.maximum(s, 1e-12)
