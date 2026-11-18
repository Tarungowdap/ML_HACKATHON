"""P2 scorer for the API: probability that each candidate row is the true arc.

Loads P2's tuned LightGBM (models/final_tuned_raw.joblib) once, at import time, and
uses exactly the raw feature columns listed in models/features_raw.json, in that order.
Never reads `label` or any reference-route information.

Same logic as models/p_match.py (P2), with the model paths resolved from the repo root.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "final_tuned_raw.joblib"
FEATS_PATH = REPO_ROOT / "models" / "features_raw.json"
EPS = 1e-6  # keeps log(p) finite for the HMM

MODEL = joblib.load(MODEL_PATH)
FEATURES = json.loads(FEATS_PATH.read_text())


def score_candidates(cands: pd.DataFrame) -> np.ndarray:
    missing = [f for f in FEATURES if f not in cands.columns]
    if missing:
        raise KeyError(f"candidate table is missing model features: {missing}")
    p = MODEL.predict_proba(cands[FEATURES].to_numpy(dtype="float32"))[:, 1]
    return np.clip(p, EPS, 1 - EPS)
