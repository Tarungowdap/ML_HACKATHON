"""Glue between the team's modules and the API.

Two backends, chosen automatically at startup:

1. TEAM pipeline (used whenever P1's and P3's modules are importable):
     P1  src/pathsnap/pathsnap.py   schema-1.0 observations + candidates, and server-side labels
     P1  src/data/candidates.py     the feature builder P2's model was trained on (19 raw features)
     P2  src/model.py               LightGBM probability per candidate
     P3  src/hmm/api.py             match_trip(): HMM/Viterbi route, label-free
   The classifier-only baseline is P2's highest-probability candidate per GPS point.
   Which emission the HMM uses (gaussian | pmatch | blend) is set in hmm_config.json.

2. PLACEHOLDER pipeline (api/fallbacks.py), only so the API and UI run before the
   team modules exist. Every response says so ("uses_placeholders": true).

Label-free rule: run() never sees the reference route. score() is the only place that
reads it, and only numbers leave score().
"""
import importlib
import logging
import sys
import time

import numpy as np
import pandas as pd

from . import config, fallbacks, metrics

log = logging.getLogger("pathsnap")

# P1's and P3's modules import each other as top-level packages (pathsnap, hmm, viz).
_SRC = str(config.REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


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


def _try_team_modules():
    """Import P1 + P2 + P3. Returns a dict of handles, or None (with the reason logged)."""
    try:
        from hmm import HMMConfig
        from hmm.api import match_trip
        from hmm.evaluate import route_prf
        from hmm.p1 import load_pathsnap
        ps = load_pathsnap()
    except Exception as e:  # noqa: BLE001 - any import problem means "not ready"
        log.warning("team pipeline not available (%s: %s)", type(e).__name__, e)
        return None
    cfg_path = config.HMM_CONFIG
    if not cfg_path.exists():
        log.warning("team pipeline not available (%s missing)", cfg_path)
        return None
    cfg = HMMConfig.load(str(cfg_path))
    try:
        from src.data.candidates import build_trip_candidates
        from src.model import score_candidates
    except Exception as e:  # noqa: BLE001
        log.warning("P2 classifier not available (%s: %s)", type(e).__name__, e)
        build_trip_candidates = score_candidates = None
    return dict(cfg=cfg, match_trip=match_trip, route_prf=route_prf, ps=ps,
                build_v0=build_trip_candidates, p2=score_candidates)


class Pipeline:
    def __init__(self):
        self.team = _try_team_modules()
        if self.team:
            cfg, has_p2 = self.team["cfg"], self.team["p2"] is not None
            if cfg.emission in ("pmatch", "blend") and not has_p2:
                raise RuntimeError(f"hmm_config.json uses emission={cfg.emission!r} but P2's model is not loadable")
            self.placeholders = {"features": False, "scorer": not has_p2, "decoder": False}
            self.emission = cfg.emission
            log.info("pipeline  -> TEAM: P1 candidates, P2 %s, P3 HMM (emission=%s, sigma=%s m, beta=%s m)",
                     "classifier" if has_p2 else "NOT LOADED", cfg.emission, cfg.sigma_m, cfg.beta_m)
            return
        self.emission = None
        self.features, f1 = _load(config.FEATURES_FN, fallbacks.build_candidates, "features")
        self.scorer, f2 = _load(config.SCORER_FN, fallbacks.score_candidates, "scorer")
        self.labeller, _ = _load(config.LABELS_FN, None, "labels")
        if f1 and not f2:
            # The trained model needs P1's full feature set; the placeholder builder only makes 3 columns.
            log.warning("scorer    -> PLACEHOLDER in use (trained model needs P1's real features)")
            self.scorer, f2 = fallbacks.score_candidates, True
        self.decoder, f3 = fallbacks.decode, True
        log.warning("decoder   -> PLACEHOLDER in use (per-point argmax, no HMM)")
        self.placeholders = {"features": f1, "scorer": f2, "decoder": f3}

    @property
    def uses_placeholders(self) -> bool:
        return any(self.placeholders.values())

    # ------------------------------------------------------------------ run (label-free)
    def run(self, trip) -> dict:
        """Returns per-GPS-point predictions plus map geometry. Never reads the reference route."""
        return self._run_team(trip) if self.team else self._run_placeholder(trip)

    def _run_team(self, trip) -> dict:
        T, timings = self.team, {}
        track = trip.track.rename(columns={"lon": "longitude", "lat": "latitude", "t": "timestamp"})
        nodes = trip.nodes.rename(columns={"lon": "longitude", "lat": "latitude"})
        arcs = trip.arcs.rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})

        # P1 (feature builder P2 was trained on) + P2 scoring; keyed by raw track row (= v0 obs_idx).
        v0 = None
        if T["p2"] is not None:
            t0 = time.perf_counter()
            v0, _ = T["build_v0"]({"trip_id": trip.trip_id, "track": track, "nodes": nodes, "arcs": arcs,
                                   "route": None}, radius_m=75.0, max_candidates=15, include_labels=False)
            timings["features_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            v0["prob"] = np.asarray(T["p2"](v0), dtype=float)
            timings["scoring_s"] = time.perf_counter() - t0
            timings["num_candidates"] = int(len(v0))

        # schema-1.0 obs order -> raw track row, using P1's own cleaning step (same one match_trip runs)
        orig_row = T["ps"].clean_track(track)[0]["orig_row"].to_numpy(np.int64)

        def p_match_fn(obs, cand):
            """P2 probability for each schema-1.0 candidate, joined by (raw row, arc_id), never by position."""
            key = pd.DataFrame({"raw": obs.set_index("obs_idx").orig_row.reindex(cand.obs_idx).to_numpy(np.int64),
                                "arc_id": cand.arc_id.to_numpy(np.int64)})
            p = key.merge(v0.rename(columns={"obs_idx": "raw"})[["raw", "arc_id", "prob"]].astype({"raw": np.int64, "arc_id": np.int64}),
                          on=["raw", "arc_id"], how="left").prob.to_numpy(float, copy=True)
            missing = np.isnan(p)
            if missing.any():   # candidate P1's schema-1.0 builder found but the 15-nearest v0 builder did not
                p[missing] = T["cfg"].pmatch_floor
            return np.clip(p, T["cfg"].pmatch_floor, 1.0)

        t0 = time.perf_counter()
        out = T["match_trip"](track, nodes, arcs, trip.trip_id, T["cfg"],
                              p_match_fn=p_match_fn if v0 is not None else None)
        timings["decoding_s"] = time.perf_counter() - t0
        pts = out["points"]
        hmm_pred = pts["arc_id"].to_numpy(np.int64)

        # classifier-only baseline: P2's best candidate per GPS point (no HMM)
        if v0 is not None:
            best = v0.loc[v0.groupby("obs_idx")["prob"].idxmax()].set_index("obs_idx")["arc_id"]
            clf_pred = best.reindex(orig_row).fillna(-1).to_numpy(np.int64)
        else:
            clf_pred = np.full(len(hmm_pred), -1, dtype=np.int64)

        return {"gps_points": np.round(pts[["lat", "lon"]].to_numpy(float), 6).tolist(),
                "gps_times": track["timestamp"].to_numpy(float)[orig_row].tolist(),
                "hmm_pred": hmm_pred, "clf_pred": clf_pred,
                "route_arcs": [int(a) for a in out["route_arcs"]],
                "route_lines": out["polylines"],
                "timings": timings,
                "hmm_info": {"segments": out["n_segments"], "breaks": len(out["breaks"])}}

    def _run_placeholder(self, trip) -> dict:
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
        timings["num_candidates"] = int(len(cands))
        route = metrics.collapse(pred)
        n, a = trip.nodes, trip.arcs
        f, t = a.from_node.values[route], a.to_node.values[route]
        lines = [[[float(n.lat[x]), float(n.lon[x])], [float(n.lat[y]), float(n.lon[y])]] for x, y in zip(f, t)]
        return {"gps_points": np.round(trip.track[["lat", "lon"]].values, 6).tolist(),
                "gps_times": trip.track["t"].tolist(),
                "hmm_pred": pred, "clf_pred": classifier_only(cands, len(trip.track)),
                "route_arcs": route, "route_lines": lines, "timings": timings, "_cands": cands}

    # ------------------------------------------------------------------ score (server-side only)
    def score(self, trip, ref, res) -> tuple[dict, dict]:
        """Metrics for the HMM output and for the classifier-only baseline. Only numbers are returned."""
        labels, cand_keys, ok = None, None, None
        if ref is not None and self.team:
            ps = self.team["ps"]
            track = trip.track.rename(columns={"lon": "longitude", "lat": "latitude", "t": "timestamp"})
            nodes = trip.nodes.rename(columns={"lon": "longitude", "lat": "latitude"})
            arcs = trip.arcs.rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})
            arcs["arc_id"] = np.arange(len(arcs), dtype=np.int32)
            obs, cand, info = ps.build_observations_and_candidates(trip.trip_id, track, nodes, arcs)
            obs, cand, _ = ps.attach_labels(obs, cand, nodes, arcs, np.asarray(ref, dtype=np.int64), info)
            labels = obs.true_arc_id.to_numpy(np.int64)        # P1's monotone alignment, same as the report
            ok = (obs.label_quality == "ok").to_numpy()
            cand_keys = set(zip(cand.obs_idx.to_numpy(int).tolist(), cand.arc_id.to_numpy(int).tolist()))
        elif ref is not None and getattr(self, "labeller", None) is not None:
            labels = np.asarray(self.labeller(trip.track, trip.nodes, trip.arcs, ref))
            c = res["_cands"]
            cand_keys = set(zip(c.obs_idx.to_numpy(int).tolist(), c.arc_id.to_numpy(int).tolist()))

        out = []
        for pred in (res["hmm_pred"], res["clf_pred"]):
            s = metrics.evaluate(pred, ref, trip.nodes, trip.arcs, None)
            if labels is not None:
                # every labelled point counts; a point whose true arc was never a candidate counts as wrong
                v = labels >= 0
                s["top1_accuracy"] = float(np.mean(pred[v] == labels[v])) if v.any() else None
                if ok is not None and ok.any():
                    s["top1_accuracy_ok_labels"] = float(np.mean(pred[ok] == labels[ok]))
            out.append(s)
        hmm, clf = out
        if labels is not None and cand_keys is not None:
            idx = np.nonzero(labels >= 0)[0]
            hmm["candidate_miss_rate"] = float(np.mean([(int(i), int(labels[i])) not in cand_keys for i in idx])) if idx.size else None
        if ref is not None and self.team:
            p, r, f = self.team["route_prf"](res["route_arcs"], list(ref))
            hmm.update(route_precision=p, route_recall=r, route_f1=f)   # P3's connected route vs reference
        return hmm, clf


def classifier_only(cands, n_obs: int) -> np.ndarray:
    """Highest-probability candidate per GPS point, no HMM (the report's baseline)."""
    best = cands.loc[cands.groupby("obs_idx")["prob"].idxmax(), ["obs_idx", "arc_id"]]
    out = np.full(n_obs, -1, dtype=np.int64)
    out[best.obs_idx.values.astype(int)] = best.arc_id.values
    return out
