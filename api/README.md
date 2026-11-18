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

## Plugging in the team's modules

The API imports these functions at startup (paths can be changed with the env vars in `config.py`):

| Owner | Default import | Signature |
|---|---|---|
| P1 | `src.features:build_candidates` | `(track, nodes, arcs) -> DataFrame` with `obs_idx`, `arc_id` + feature columns |
| P2 | `src.model:score_candidates` | `(cands) -> np.ndarray` probability per row. **Done**: `src/model.py` loads `models/final_tuned_raw.joblib` and the 19 columns in `models/features_raw.json` |
| P3 | `src.hmm:decode` | `(cands_with_prob, track, nodes, arcs) -> np.ndarray` arc_id per GPS point (`-1` = unmatched) |
| P1 | `src.features:label_observations` | `(track, nodes, arcs, route) -> np.ndarray` true arc per GPS point (optional) |

The trained scorer needs P1's full feature set, so while P1's builder is missing the API also uses the
placeholder scorer (logged as `trained model needs P1's real features`).

Any stage that can't be imported falls back to a simple placeholder in `api/fallbacks.py`
(nearest arcs / hand-set score / per-point argmax). This keeps the API and UI testable early.
It is **not** the submitted method: the startup log prints `PLACEHOLDER in use` and every
response carries `"uses_placeholders": true` until all three real stages are in.
