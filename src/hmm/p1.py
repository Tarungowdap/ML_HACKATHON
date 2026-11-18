"""Find P1's shared module (pathsnap.py) wherever the repo keeps it.

Supported layouts: src/pathsnap/pathsnap.py (repo README) or src/pathsnap.py.
P3 does not ship its own copy, so the API always runs P1's code.
"""
from __future__ import annotations

import importlib


def load_pathsnap():
    for name in ("pathsnap.pathsnap", "pathsnap"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "build_observations_and_candidates"):
            return m
    raise ImportError("P1's pathsnap.py not found: expected src/pathsnap/pathsnap.py or src/pathsnap.py "
                      "(with src/ on sys.path)")
