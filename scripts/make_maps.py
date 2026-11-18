"""Folium maps for the report: full trip (good case) + zoomed failure windows.

python scripts/make_maps.py --trip 00000000 --tag sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm import HMMConfig, decode, load_trip_tables  # noqa: E402
from hmm.evaluate import evaluate_trip  # noqa: E402
from viz import save, trip_map  # noqa: E402


def worst_window(wrong: np.ndarray, width: int = 60) -> tuple[int, int]:
    """Window of `width` points with the most HMM errors."""
    c = np.convolve(wrong.astype(int), np.ones(width, int), mode="valid")
    s = int(np.argmax(c))
    return s, s + width - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="00000000")
    ap.add_argument("--tag", default="sample")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--config", default=str(ROOT / "hmm_config.json"))
    ap.add_argument("--out", default=str(ROOT / "maps"))
    a = ap.parse_args()

    td = load_trip_tables(a.trip, a.data, a.tag)
    cfg = HMMConfig.load(a.config)
    r = decode(td.obs, td.cand, td.graph, cfg)
    e = evaluate_trip(td, r)
    title = (f"Trip {a.trip}: HMM {e['acc_hmm']:.1%} vs classifier-only {e['acc_top1']:.1%} pointwise | "
             f"route F1 {e['route_f1_hmm']:.2f}")
    m = trip_map(td.obs, td.arcs, td.route, r.top1_arc, r.arc_id, title=title)
    print(save(m, Path(a.out) / f"trip_{a.trip}_full.html"))

    wrong = (r.arc_id != td.obs.true_arc_id.to_numpy()) & (td.obs.true_arc_id.to_numpy() >= 0)
    lo, hi = worst_window(wrong)
    m = trip_map(td.obs[(td.obs.obs_idx >= lo) & (td.obs.obs_idx <= hi)], td.arcs,
                 [x for x in td.route if x in set(td.obs.true_arc_id[lo:hi + 1]) | set(r.arc_id[lo:hi + 1])],
                 r.top1_arc[lo:hi + 1], r.arc_id[lo:hi + 1],
                 title=f"Trip {a.trip} failure window obs {lo}-{hi}: {int(wrong[lo:hi + 1].sum())} HMM errors")
    print(save(m, Path(a.out) / f"trip_{a.trip}_failure_{lo}_{hi}.html"))

    # second window: largest classifier-only mistakes the HMM fixed (shows the "jumps")
    fixed = (r.top1_arc != td.obs.true_arc_id.to_numpy()) & ~wrong
    lo, hi = worst_window(fixed, 80)
    m = trip_map(td.obs[(td.obs.obs_idx >= lo) & (td.obs.obs_idx <= hi)], td.arcs,
                 [x for x in td.route if x in set(td.obs.true_arc_id[lo:hi + 1])],
                 r.top1_arc[lo:hi + 1], r.arc_id[lo:hi + 1],
                 title=f"Trip {a.trip} obs {lo}-{hi}: {int(fixed[lo:hi + 1].sum())} classifier-only errors fixed by the HMM")
    print(save(m, Path(a.out) / f"trip_{a.trip}_hmm_fixes_{lo}_{hi}.html"))


if __name__ == "__main__":
    main()
