"""Final-style evaluation on the 10 STUDENT trips (schema 1.0 candidates rebuilt with P1's code).
No tuning happens here. Writes results/student_eval.csv (per trip), results/student_summary.json
and the good-case / failure-case maps.

python scripts/eval_student.py [--config hmm_config.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm import HMMConfig, decode  # noqa: E402
from hmm.data import load_split  # noqa: E402
from hmm.evaluate import evaluate_trip  # noqa: E402
from viz import save, trip_map  # noqa: E402


def wavg(d, k):
    m = d[k].notna()
    return float(np.average(d.loc[m, k], weights=d.n_labeled[m])) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--config", default=str(ROOT / "hmm_config.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "hmm"))
    ap.add_argument("--maps", default=str(ROOT / "maps"))
    a = ap.parse_args()
    cfg = HMMConfig.load(a.config)
    trips = load_split(a.data, "student")
    rows, res = [], {}
    for td in trips:
        r = decode(td.obs, td.cand, td.graph, cfg)
        res[td.trip_id] = r
        rows.append(evaluate_trip(td, r))
    d = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "student_eval.csv", index=False)
    summ = {k: wavg(d, k) for k in ["acc_top1", "acc_hmm", "acc_top1_ok", "acc_hmm_ok", "oracle_acc",
                                    "cont_top1", "cont_hmm", "cont_hmm_expanded", "gt_continuity"]}
    summ.update({k: float(d[k].median()) for k in ["route_p_hmm", "route_r_hmm", "route_f1_hmm", "route_f1_top1",
                                                   "edit_norm_hmm", "edit_norm_top1"]})
    summ.update({k: int(d[k].sum()) for k in d.columns if k.startswith(("n_breaks", "n_uturns", "err_"))})
    summ["mean_decode_s"] = float(d.seconds.mean()); summ["config"] = json.loads(Path(a.config).read_text())
    (out / "student_summary.json").write_text(json.dumps(summ, indent=2))
    print(d[["trip_id", "n_obs", "acc_top1", "acc_hmm", "acc_hmm_ok", "oracle_acc", "route_f1_hmm",
             "edit_norm_hmm", "n_breaks", "n_uturns_hmm"]].round(3).to_string(index=False))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items() if k != "config"}, indent=1))

    # maps: best trip (highest accuracy, >= 1000 points) and worst trip
    big = d[d.n_obs >= 1000]
    good = big.sort_values("acc_hmm").iloc[-1].trip_id; bad = d.sort_values("acc_hmm").iloc[0].trip_id
    for tag, tid in (("good", good), ("failure", bad)):
        td = next(t for t in trips if t.trip_id == tid); r = res[tid]; e = d[d.trip_id == tid].iloc[0]
        title = (f"Student trip {tid} ({tag}): HMM {e.acc_hmm:.1%} vs classifier-only {e.acc_top1:.1%} | "
                 f"route F1 {e.route_f1_hmm:.2f}")
        print(save(trip_map(td.obs, td.arcs, td.route, r.top1_arc, r.arc_id, title=title),
                   Path(a.maps) / f"student_{tag}_{tid}.html"))


if __name__ == "__main__":
    main()
