# PathSnap P3: HMM / Viterbi route reconstruction

`data/` is generated (`python scripts/rebuild_v1.py`, ~35 s) and not committed. Outputs go to `results/hmm/` and `maps/`.

Turns per-point candidate scores into one connected route. Final model: Gaussian emission, tuned on 78 train trips
(`hmm_config.json`). P2's model can be swapped in with `"emission": "pmatch"` or `"blend"` plus `p_match`; nothing
else changes.

```
src/pathsnap/          P1's shared module (not part of P3; found via src/hmm/p1.py)
src/hmm/
  config.py            HMMConfig  <->  hmm_config.json
  graph.py             TripGraph: per-trip directed graph, bounded + cached Dijkstra, arc paths
  emission.py          Gaussian on dist_to_arc_m | log p_match | blend; optional heading term
  transition.py        Newson–Krumm transition + U-turn penalty
  viterbi.py           log-space Viterbi with segment restarts
  decode.py            decode()  (label-free, used by the API), collapse(), expand_route()
  api.py               match_trip() -> points, route arcs, polylines, GeoJSON; score_trip() server-side (for P4)
  p1.py                locates P1's pathsnap module
  p2_adapter.py        joins P2's p_match by (trip_id, obs_idx, arc_id); temporary v0 -> schema-1.0 bridge
  data.py              load P1's parquet tables (one trip or a whole split)
  estimate.py          Newson–Krumm estimates of sigma and beta from labelled trips
  evaluate.py          pointwise / continuity / route P-R-F1 / edit distance / error types   (never used by the API)
src/viz/
  coords.py            arcs_to_polylines() [[lat, lon], ...], route_geojson()
  maps.py              Folium map: raw GPS, ground truth, classifier-only, HMM (toggleable layers)
scripts/
  rebuild_v1.py        schema-1.0 observations/candidates for all 90 trips with P1's code -> data/
  tune_train.py        staged tuning on train trips only (refuses student/evaluator IDs) -> hmm_config.json
  eval_student.py      held-out student metrics + good/failure maps
  student_p2_compare.py  P2's model inside the HMM (p_match vs Gaussian vs blend)
  integration_test.py  the API path end to end (label-free decode, then server-side scoring)
  make_maps.py         full-trip / failure / "HMM fixes" maps for one trip
  plots_train.py       sensitivity plots from results/hmm/tuning_train.csv
tests/test_hmm.py      parallel-road jump, missing candidates, broken path, GPS gap, label-free, p_match plumbing
```

## Run

```bash
pip install -r requirements.txt
python tests/test_hmm.py
python scripts/rebuild_v1.py                           # schema-1.0 data for all 90 trips (~35 s)
python scripts/tune_train.py --workers 2               # ~45 min on 2 cores
python scripts/eval_student.py                         # student metrics + maps
python scripts/integration_test.py --student-parquets  # API path == report path (must print difference 0.0000)
python scripts/integration_test.py --raw <evaluator_test folder>   # label-free run of all 10 demo trips
python scripts/plots_train.py
```

## Design (report §5 notes)

- **States** at step t are the candidate arcs of point t (P1: ≤ 20 within 75 m), not the whole map.
  Points with 0 candidates are skipped in decoding (transitions bridge across them using the real dt)
  and filled afterwards with the previous arc (`fill_missing`).
- **Emission**: `−d²/(2σ²)` on `dist_to_arc_m`, plus optionally `κ·(cos Δbearing − 1)` when the point's own
  heading is valid and it moves ≥ 2 m/s. The heading term matters because every road has two directed arcs
  with *identical* distance; without it, classifier-only top-1 is 53% on the sample trip (direction is a coin flip).
  P2 version: `log p_match` or `w·log p_match + (1−w)·log Gaussian`.
- **Transition** (Newson–Krumm): `−|road − straight| / β_eff`, `β_eff = β·max(dt, 1)`.
  Road distance = rest of arc a + shortest path(to(a) → from(b)) + offset into b; same arc forward = offset difference
  (backwards < 15 m treated as GPS noise). Dijkstra is capped at `200 m + 60 m/s·dt` (max 5 km) and cached per source node.
  Reverse twin of the same road gets an extra `uturn_logp` penalty. Reachable moves are floored at −40, unreachable = −60.
- **Viterbi** in log space, O(n·k²). If no candidate at t is reachable from any at t−1, the current segment is
  back-tracked and a new one starts (logged in `breaks`). Optional `split_on_gap` restarts after dt > 30 s;
  otherwise gaps just loosen β and the search cap.
- **Route output**: collapse repeats, then `expand_route` inserts shortest-path arcs between non-adjacent arcs
  (short arcs crossed between two 1-s fixes), giving a fully connected route for the UI.
- **Tuning rule**: among configs within 0.5 pp of the best pointwise accuracy, pick the lowest route edit distance.
  Pure accuracy rewards flip-flopping between road directions on ambiguous labels (22 spurious U-turns vs 4).

## Results (train-tuned, schema 1.0 data rebuilt with P1's own pathsnap.py)

Data: `scripts/rebuild_v1.py --repo <ML_HACKATHON>` regenerates schema-1.0 observations/candidates for all 90
trips from the repo's observations, nodes, arcs and routes (the repo's candidates_student is the old v0 file;
candidates_train is in Git LFS). Verified: identical to P1's sample in every column, and identical labels to the
repo observations on all 227,577 points.

Tuning: `scripts/tune_train.py` on 78 train trips (00000033 and 00000054 excluded: their candidate ceiling is
37% / 10%, a P1 data issue). 57 configs in 3 stages. Selected **σ = 6 m, β = 3 m, U-turn −20, heading κ = 0**.

| | train (78 trips) | student (10 trips, never used for tuning) |
|---|---|---|
| HMM pointwise accuracy | 92.2% | **92.4%** |
| HMM accuracy on `ok` labels | 96.4% | **97.5%** |
| distance-only top-1 | 55.2% | 53.8% |
| candidate ceiling | 99.1% | 99.9% |
| route F1 (median per trip) | 0.971 | **0.973** |
| route edit distance (median) | 0.079 | 0.083 |
| U-turns / path breaks | 35 / 17 | 4 / 0 |

With P2's current model (old v0 candidates, sample-tuned σ/β): P2 alone 77.3% → HMM 87.3% on student
(`scripts/student_p2_compare.py`). Re-run once P2 retrains on schema 1.0.

Findings for §6:
- The heading term HURTS on train once U-turns are penalised (κ 0 > 1 > 2 > 4 at every penalty): noisy 1-Hz
  GPS headings mislead more than they help; the road network settles direction by itself.
- The U-turn penalty is the single most important addition: 1,650 → 75 spurious U-turns (−10) and +1.1 pp accuracy.
- Accuracy is flat for σ 4.5–6 and β 1.5–3 (±0.2 pp); σ = 3 keeps accuracy but worsens routes.
- Raw per-point continuity is 89% (short arcs crossed between 1-s fixes); after expand_route it is 99.99%.
- Failure case: student trip 00000077 (85.2%; `ok` labels 93.7%) has many ambiguous labels, consistent with an
  out-and-back route on the same road. Sample trip end: a parked car with GPS jitter.

Maps: `maps/student_good_00000040.html`, `maps/student_failure_00000077.html`.
Plots: `results/hmm/train_sens_sigma_beta.png`, `results/hmm/train_sens_uturn_heading.png`, `results/hmm/train_sens_uturns.png`.

## Earlier results (sample trip only)

| | classifier-only (nearest + heading) | HMM |
|---|---|---|
| pointwise accuracy | 91.7% | **98.1%** |
| accuracy on `ok` labels | n/a | 99.6% |
| route F1 (vs .route) | 0.69 | **0.97** |
| normalised edit distance | 1.18 | **0.11** |
| ceiling: candidate recall / GT continuity | 99.96% / 100% | |

Selected: σ = 12 m, β = 4 m, U-turn −3, κ = 2 (NK estimates from labels: σ ≈ 9.0 m, β ≈ 0.7 m).
Decode time ≈ 0.3 s per trip; end-to-end from raw files ≈ 0.7 s.

Raw continuity of per-point arcs is 94% because consecutive 1-s fixes skip short arcs (median 8 m); after
`expand_route` it is 100%. Not an error.

**Failure case (map `trip_00000000_failure_*.html`)**: last ~60 points, the car is parked. ~8 m GPS jitter looks like
5–13 m/s movement and the HMM wanders onto the adjacent short arcs; the reference route ends on a 12 m arc.
Possible fixes: treat speed < ~1 m/s (smoothed) as stationary and force "stay on arc", or use P2's p_match.

## API integration (for P4)

```python
from hmm import HMMConfig
from hmm.api import match_trip, score_trip
cfg = HMMConfig.load('hmm_config.json')            # load once at startup
out = match_trip(track, nodes, arcs, trip_id, cfg)  # label-free -> out['polylines'], out['geojson'], out['points']
score = score_trip(track, nodes, arcs, route, trip_id, out['points'].arc_id.to_numpy(), out['route_arcs'])  # server-side only
```
Verified: student accuracy through this path = 92.43%, identical to the report; all 10 evaluator trips decode in <= 1.72 s.

## Open items

- P2: a schema-1.0 retrain plus out-of-fold train `p_match` would allow a fair blend test (currently Gaussian is final).
- P1: duplicate arcs and disconnected network pieces; trips 00000033 / 00000054 have broken candidate coverage.
- Stationary vehicles (parked-car GPS jitter) are not modelled.
