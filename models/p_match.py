"""P2 -> P3/P4 inference helper: candidate match probabilities, label-free.

p_match_fn(obs, cand) returns one probability in (0, 1) per row of `cand`, in the same row order,
so `cand['p_match'] = p_match_fn(obs, cand)` joins on (trip_id, obs_idx, arc_id) trivially.
Uses only the feature columns listed in models/features_raw.json; never reads label / true_arc_id.
"""
import json, os
import numpy as np
import joblib

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, 'models', 'final_tuned_raw.joblib')
FEATS_PATH = os.path.join(_DIR, 'models', 'features_raw.json')
EPS = 1e-6                       # clip so log(p) and log(1-p) are always finite for the HMM

_model = _feats = None
def load(model_path=MODEL_PATH, feats_path=FEATS_PATH):
    """Load once (call at API startup)."""
    global _model, _feats
    _model = joblib.load(model_path)
    _feats = json.load(open(feats_path))
    return _model, _feats

def p_match_fn(obs, cand):
    """obs: observations DataFrame or None (only used if a feature lives there, joined on trip_id, obs_idx).
    cand: candidates DataFrame from P1's builder. Returns np.ndarray (len(cand),) of P(candidate is the true arc)."""
    if _model is None: load()
    X = cand
    missing = [f for f in _feats if f not in X.columns]
    if missing and obs is not None:
        extra = [c for c in missing if c in obs.columns]
        X = X.merge(obs[['trip_id', 'obs_idx'] + extra], on=['trip_id', 'obs_idx'], how='left')
    still = [f for f in _feats if f not in X.columns]
    if still: raise KeyError(f'candidate table is missing model features: {still}')
    p = _model.predict_proba(X[_feats].to_numpy(dtype='float32'))[:, 1]
    return np.clip(p, EPS, 1 - EPS)
