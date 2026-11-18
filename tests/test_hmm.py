"""Edge-case tests. Run: python tests/test_hmm.py   (or pytest tests/)

Synthetic network: a straight two-way road 0-1-2-3-4 (100 m arcs, x axis), a parallel road 15 m north
(nodes 5-6-7-8-9, one-way east) not connected to the main road, and a far-away disconnected arc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm import HMMConfig, TripGraph, decode, expand_route, load_trip_tables  # noqa: E402
from hmm.evaluate import continuity  # noqa: E402

NODES = np.array([[0, 0], [100, 0], [200, 0], [300, 0], [400, 0],
                  [0, 15], [100, 15], [200, 15], [300, 15], [400, 15],
                  [5000, 5000], [5100, 5000]], float)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 0), (2, 1), (3, 2), (4, 3),     # arcs 0-7 main road
         (5, 6), (6, 7), (7, 8), (8, 9),                                     # arcs 8-11 parallel
         (10, 11)]                                                           # arc 12 far away


def make_graph():
    f = np.array([e[0] for e in EDGES]); t = np.array([e[1] for e in EDGES])
    L = np.hypot(*(NODES[t] - NODES[f]).T)
    arcs = pd.DataFrame({"arc_id": np.arange(len(EDGES)), "from_node": f, "to_node": t, "length_m": L})
    return TripGraph(arcs, n_nodes=len(NODES)), arcs


def tables(xs, ys, ts, g, radius=30.0, drop=()):
    """Build obs/cand like P1 does (simplified: all arcs within radius)."""
    n = len(xs)
    obs = pd.DataFrame({"trip_id": "T", "obs_idx": np.arange(n), "x": xs, "y": ys, "timestamp": ts,
                        "dt_prev": np.r_[0, np.diff(ts)], "heading_valid": True, "speed": 10.0})
    rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i in drop:
            continue
        A = NODES[g.from_node]; B = NODES[g.to_node]; d = B - A
        tt = np.clip(((x - A[:, 0]) * d[:, 0] + (y - A[:, 1]) * d[:, 1]) / (d ** 2).sum(1), 0, 1)
        S = A + tt[:, None] * d; dist = np.hypot(x - S[:, 0], y - S[:, 1])
        bear = (90 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
        bd = np.abs((90 - bear + 180) % 360 - 180)            # vehicle heading east (90 deg)
        k = np.flatnonzero(dist <= radius); k = k[np.argsort(dist[k], kind="stable")]
        for r, a in enumerate(k):
            rows.append(dict(trip_id="T", obs_idx=i, arc_id=a, dist_to_arc_m=dist[a], offset_m=tt[a] * g.length[a],
                             bearing_diff_deg=bd[a], cand_rank=r))
    obs["n_candidates"] = pd.Series([r["obs_idx"] for r in rows]).value_counts().reindex(obs.obs_idx).fillna(0).to_numpy()
    return obs, pd.DataFrame(rows)


def test_parallel_jump_removed():
    g, _ = make_graph()
    xs = np.arange(5, 400, 10.0); ys = np.full(len(xs), 2.0)
    ys[10] = 13.0                                           # one noisy point is closer to the parallel road
    obs, cand = tables(xs, ys, np.arange(len(xs), dtype=float), g)
    r = decode(obs, cand, g, HMMConfig(sigma_m=5, beta_m=3, heading_kappa=2))
    assert r.top1_arc[10] in (8, 9, 10, 11), "classifier-only should jump to the parallel road"
    assert set(r.arc_id) <= {0, 1, 2, 3}, f"HMM should stay on the main road eastbound, got {set(r.arc_id)}"
    assert continuity(r.arc_id, g) == 1.0 and not r.breaks


def test_missing_candidates_filled():
    g, _ = make_graph()
    xs = np.arange(5, 400, 10.0); ys = np.full(len(xs), 2.0)
    obs, cand = tables(xs, ys, np.arange(len(xs), dtype=float), g, drop={12, 13})
    r = decode(obs, cand, g, HMMConfig(heading_kappa=2))
    assert not r.matched[12] and not r.matched[13]
    assert r.arc_id[12] == r.arc_id[11] and r.arc_id[12] >= 0        # filled with previous arc
    r2 = decode(obs, cand, g, HMMConfig(heading_kappa=2, fill_missing="none"))
    assert r2.arc_id[12] == -1


def test_broken_path_restarts_segment():
    g, _ = make_graph()
    xs = np.r_[np.arange(5, 200, 10.0), np.arange(5010, 5100, 10.0)]
    ys = np.r_[np.full(20, 2.0), np.full(9, 5001.0)]
    obs, cand = tables(xs, ys, np.arange(len(xs), dtype=float), g)
    r = decode(obs, cand, g, HMMConfig(heading_kappa=2))
    assert r.breaks == [20], r.breaks                      # no road path from main road to arc 12
    assert r.segment.max() == 1 and (r.arc_id[20:] == 12).all()


def test_gap_loosens_and_splits():
    g, _ = make_graph()
    xs = np.r_[np.arange(5, 100, 10.0), np.arange(305, 400, 10.0)]
    ys = np.full(len(xs), 2.0)
    ts = np.r_[np.arange(10.0), np.arange(10.0) + 60]        # 51 s gap after point 9
    obs, cand = tables(xs, ys, ts, g)
    r = decode(obs, cand, g, HMMConfig(heading_kappa=2))
    assert not r.breaks and set(r.arc_id) == {0, 3}
    assert expand_route(r.arc_id, g) == [0, 1, 2, 3]         # gap filled by shortest path
    r2 = decode(obs, cand, g, HMMConfig(heading_kappa=2, split_on_gap=True))
    assert r2.breaks == [10]


def test_sample_trip_label_free_and_pmatch():
    data = ROOT / "data"
    if not (data / "observations_sample.parquet").exists():
        return
    td = load_trip_tables("00000000", data, "sample")
    cfg = HMMConfig.load(ROOT / "hmm_config.json")
    r = decode(td.obs, td.cand, td.graph, cfg)
    obs_nl = td.obs.drop(columns=["true_arc_id", "label_quality", "true_dist_m"])
    r_nl = decode(obs_nl, td.cand.drop(columns=["label"]), td.graph, cfg)
    assert np.array_equal(r.arc_id, r_nl.arc_id), "decode must not depend on labels"
    # a perfect p_match must give (near) oracle accuracy -> plumbing for P2 works
    p = np.where(td.cand.label == 1, 0.99, 0.01)
    rp = decode(td.obs, td.cand, td.graph, HMMConfig(emission="pmatch", beta_m=cfg.beta_m), p_match=p)
    acc = (rp.arc_id == td.obs.true_arc_id.to_numpy()).mean()
    assert acc > 0.995, acc


def test_attach_p_match_joins_by_key_and_fails_on_gaps():
    from hmm.p2_adapter import attach_p_match
    cand = pd.DataFrame({"trip_id": "T", "obs_idx": [0, 0, 1], "arc_id": [5, 6, 5]})
    preds = pd.DataFrame({"trip_id": "T", "obs_idx": [1, 0, 0], "arc_id": [5, 6, 5], "p_match": [.3, .2, .9]})
    assert np.allclose(attach_p_match(cand, preds), [.9, .2, .3])          # shuffled rows still align
    try:
        attach_p_match(cand, preds.iloc[:2]); raise AssertionError("should fail on missing rows")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
