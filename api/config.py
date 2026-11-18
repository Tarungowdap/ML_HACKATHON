"""Runtime configuration for the PathSnap API (Laptop 1).

Everything is overridable with environment variables so the same code runs
on the dev machine and on the demo laptop without edits.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Folder holding one sub-folder per trip: <EVAL_DIR>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
# Point this at student_test during development and at evaluator_test for the final demo.
EVAL_DIR = Path(os.getenv("PATHSNAP_EVAL_DIR", REPO_ROOT / "data" / "evaluator_test"))

# P3's tuned HMM settings (emission gaussian | pmatch | blend, sigma, beta, ...).
HMM_CONFIG = Path(os.getenv("PATHSNAP_HMM_CONFIG", REPO_ROOT / "hmm_config.json"))

# Placeholder-mode plug-in points, used only if P1's/P3's modules are missing.
# Written as "module.path:function_name".
# If a module cannot be imported, the API falls back to a simple built-in
# placeholder and flags it loudly in both the logs and the response.
FEATURES_FN = os.getenv("PATHSNAP_FEATURES_FN", "src.features:build_candidates")   # P1
SCORER_FN = os.getenv("PATHSNAP_SCORER_FN", "src.model:score_candidates")          # P2
DECODER_FN = os.getenv("PATHSNAP_DECODER_FN", "src.hmm:decode")                     # P3

# Optional observation-level labelling (P1's labelling logic, server-side only).
LABELS_FN = os.getenv("PATHSNAP_LABELS_FN", "src.features:label_observations")

# Allowed UI origins (Laptop 2). "*" is fine on a closed demo network.
CORS_ORIGINS = os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",")
