"""Glue between the team's modules and the API.

Interfaces the team modules must implement:

  P1  build_candidates(track, nodes, arcs) -> DataFrame
        one row per (obs_idx, arc_id) candidate, plus every feature column the model needs.
        track: lon, lat, t | nodes: lon, lat | arcs: from_node, to_node (row index = id)

  P2  score_candidates(cands: DataFrame) -> np.ndarray
        probability that each candidate row is the true arc. Loads the saved model and
        preprocessor itself (once, at import time).

  P3  decode(cands_with_prob: DataFrame, track, nodes, arcs) -> np.ndarray
        one arc_id per obs_idx (len(track) values, -1 if a point is unmatched).
        cands has a "prob" column added by the API.

  P1  label_observations(track, nodes, arcs, route) -> np.ndarray   (optional, server-side only)
        true arc_id per obs_idx (-1 if none) for observation-level Top-1 accuracy.
"""
import importlib
import logging
import time

import numpy as np

from . import config, fallbacks

log = logging.getLogger("pathsnap")


def _load(spec: str, fallback, stage: str):
    mod_name, fn_name = spec.split(":")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name)
        log.info("%-9s -> %s", stage, spec)
        return fn, False
    except (ImportError, AttributeError) as e:
        if fallback is None:
            log.info("%-9s -> not available (%s)", stage, e)
            return None, False
        log.warning("%-9s -> PLACEHOLDER in use (%s not importable: %s)", stage, spec, e)
        return fallback, True


class Pipeline:
    def __init__(self):
        self.features, f1 = _load(config.FEATURES_FN, fallbacks.build_candidates, "features")
        self.scorer, f2 = _load(config.SCORER_FN, fallbacks.score_candidates, "scorer")
        self.decoder, f3 = _load(config.DECODER_FN, fallbacks.decode, "decoder")
        self.labeller, _ = _load(config.LABELS_FN, None, "labels")
        if f1 and not f2:
            # The trained model needs P1's full feature set; the placeholder builder only makes 3 columns.
            log.warning("scorer    -> PLACEHOLDER in use (trained model needs P1's real features)")
            self.scorer, f2 = fallbacks.score_candidates, True
        self.placeholders = {"features": f1, "scorer": f2, "decoder": f3}

    @property
    def uses_placeholders(self) -> bool:
        return any(self.placeholders.values())

    def run(self, trip):
        """Returns (hmm_pred, classifier_only_pred, cands, timings); preds hold one arc_id per GPS point."""
        timings = {}
        t0 = time.perf_counter()
        cands = self.features(trip.track, trip.nodes, trip.arcs)
        timings["features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cands = cands.copy()
        cands["prob"] = np.asarray(self.scorer(cands), dtype=float)
        timings["scoring_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = np.asarray(self.decoder(cands, trip.track, trip.nodes, trip.arcs), dtype=np.int64)
        timings["decoding_s"] = time.perf_counter() - t0

        if len(pred) != len(trip.track):
            raise ValueError(f"decoder returned {len(pred)} arcs for {len(trip.track)} GPS points")
        timings["num_candidates"] = int(len(cands))
        return pred, classifier_only(cands, len(trip.track)), cands, timings


def classifier_only(cands, n_obs: int) -> np.ndarray:
    """Highest-probability candidate per GPS point, no HMM (the report's baseline)."""
    best = cands.loc[cands.groupby("obs_idx")["prob"].idxmax(), ["obs_idx", "arc_id"]]
    out = np.full(n_obs, -1, dtype=np.int64)
    out[best.obs_idx.values.astype(int)] = best.arc_id.values
    return out
