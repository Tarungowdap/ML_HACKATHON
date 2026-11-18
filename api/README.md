# PathSnap API (Laptop 1)

FastAPI backend that runs the full prediction pipeline for a selected trip and scores it
against the reference route **on the server**. The reference route never leaves the API.

## Run

```bash
pip install -r api/requirements.txt

# from the repo root
# Windows (PowerShell):  $env:PATHSNAP_EVAL_DIR="data\evaluator_test"
# macOS / Linux:         export PATHSNAP_EVAL_DIR=data/evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000** in a browser. The API serves the evaluator UI (`ui/`) at `/`,
so one command runs the whole system. The UI still fetches `/trips` and `/predict` over HTTP, so
every request appears in the terminal log; this is the API log the evaluators ask to see.
Keep that terminal visible during the demo.

Interactive API docs: `http://127.0.0.1:8000/docs`.

Leaflet is bundled in `ui/vendor/`, so the UI works offline; only the OpenStreetMap background
tiles need internet (without it the GPS points and route still draw on a grey background).

**Two-laptop setup (if ever needed):** start the API with `--host 0.0.0.0` (as above), find the
API laptop's IP (`ipconfig` / `ip a`), open `http://<that-ip>:8000` on the other laptop, or type
that address into the UI's API box. Both machines must be on the same network, and Windows
Firewall must allow Python on port 8000.

## Data layout

```
data/evaluator_test/<trip_id>/<trip_id>.track   lon lat t
                              <trip_id>.nodes   lon lat
                              <trip_id>.arcs    from_node to_node
                              <trip_id>.route   arc_id per line (server-side only)
```
`data/` is git-ignored; copy the trip folders onto Laptop 1.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, and which pipeline stages are real vs placeholder |
| GET | `/trips` | `{"trips": [...]}`, the trip IDs found in `PATHSNAP_EVAL_DIR` |
| GET | `/model` | P2's final model summary read from `results/` (tuned vs untuned, model comparison, best params) |
| POST | `/predict` | body `{"trip_id": "00000000"}`; returns GPS points, predicted route, metrics, timings |

`/predict` response fields used by the UI:
- `gps_points`: `[[lat, lon], ...]`, one per GPS observation
- `predicted_route_lines`: `[[[lat, lon], [lat, lon]], ...]`, one segment per predicted arc, in order
- `predicted_arc_per_point`: arc id chosen for each GPS point
- `metrics`: `route_continuity`, `arc_precision`, `arc_recall`, `arc_f1`, `length_precision`,
  `length_recall`, `on_route_rate`, and `top1_accuracy` once P1's labeller is plugged in
- `metrics_classifier_only`: the same metrics for the per-point argmax of the classifier, without the HMM
  (the report's classifier-only vs classifier + HMM comparison)
- `metrics.candidate_miss_rate`: share of GPS points whose true arc is not in their candidate set
  (needs `label_observations`)
- `timings`: seconds per stage + `total_s`
- `uses_placeholders`: **must be `false` before the demo**

## Pipeline

At startup the API loads the team's code and logs which pipeline it uses:

```
pipeline  -> TEAM: P1 candidates, P2 classifier, P3 HMM (emission=gaussian, sigma=6.0 m, beta=3.0 m)
```

| Stage | Code | Owner |
|---|---|---|
| Candidates + features (P2's 19 features) | `src/data/candidates.py:build_trip_candidates` | P1 |
| Candidates for the HMM (schema 1.0) + server-side labels | `src/pathsnap/pathsnap.py` | P1 |
| Classifier probability per candidate | `src/model.py:score_candidates` (LightGBM) | P2 |
| HMM / Viterbi route | `src/hmm/api.py:match_trip`, settings in `hmm_config.json` | P3 |

- **Classifier only** = P2's highest-probability candidate per GPS point, no HMM.
- **+ HMM** = P3's Viterbi route. `hmm_config.json` `"emission"` chooses what the HMM uses per point:
  `gaussian` (distance only), `pmatch` (P2's probability) or `blend` (both, weight `w_pmatch`).
  P2's probabilities are joined onto the HMM's candidates by (raw track row, arc_id), never by row order.
- **Top-1 accuracy** uses P1's `attach_labels` (the same labels as the report), counted over every labelled
  GPS point; a point whose true arc is not among its candidates counts as wrong.
- `run()` never sees the `.route` file; `score()` is the only code that reads it, and only numbers leave it.

If P1's or P3's modules can't be imported, the API falls back to the simple placeholders in
`api/fallbacks.py`, logs `PLACEHOLDER in use`, and every response carries `"uses_placeholders": true`.
