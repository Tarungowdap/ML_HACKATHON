"""PathSnap FastAPI backend (Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /                  evaluator UI (static files in ui/)
    GET  /health            liveness + which pipeline stages are real vs placeholder
    GET  /trips             trip IDs available for evaluation
    GET  /model             P2's final model summary (from results/), for the UI's model card
    POST /predict           {"trip_id": "..."} -> predicted route + server-side metrics
"""
import csv
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, data_io, metrics
from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

pipeline = Pipeline()
log.info("Evaluation data folder: %s (%d trips)", config.EVAL_DIR, len(data_io.list_trips(config.EVAL_DIR)))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path,
             response.status_code, (time.perf_counter() - t0) * 1000)
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "placeholders": pipeline.placeholders, "hmm_emission": pipeline.emission,
            "uses_placeholders": pipeline.uses_placeholders, "eval_dir": str(config.EVAL_DIR)}


@app.get("/trips")
def trips():
    return {"trips": data_io.list_trips(config.EVAL_DIR)}


def _read_csv(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v or None


@app.get("/model")
def model_info():
    """Read-only summary of P2's experiment outputs. Contains no evaluator-trip information."""
    res = config.REPO_ROOT / "results"
    params = res / "best_params.json"
    feats = config.REPO_ROOT / "models" / "features_raw.json"
    return {
        "final_model": "LightGBM (Optuna-tuned)",
        "best_params": json.loads(params.read_text()) if params.exists() else None,
        "num_features": len(json.loads(feats.read_text())) if feats.exists() else None,
        "tuned_vs_untuned": _read_csv(res / "tuned_vs_untuned.csv"),
        "model_comparison": _read_csv(res / "model_comparison.csv"),
    }


UI_DIR = config.REPO_ROOT / "ui"
if UI_DIR.is_dir():
    # The UI is plain static files; it still talks to /trips and /predict over HTTP like any client.
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in data_io.list_trips(config.EVAL_DIR):
        raise HTTPException(404, f"Unknown trip_id {req.trip_id!r}")

    t_start = time.perf_counter()
    trip = data_io.load_trip(config.EVAL_DIR, req.trip_id)
    log.info("trip %s: %d GPS points, %d nodes, %d arcs", trip.trip_id, len(trip.track), len(trip.nodes), len(trip.arcs))

    res = pipeline.run(trip)          # label-free: never sees the reference route
    timings = res["timings"]

    # --- server-side evaluation (reference route never leaves this block) ---
    ref = data_io.load_reference_route(config.EVAL_DIR, req.trip_id)
    scores, scores_clf = pipeline.score(trip, ref, res)
    del ref

    timings["total_s"] = time.perf_counter() - t_start
    fmt = lambda d: ", ".join(f"{k}={v:.3f}" for k, v in d.items() if isinstance(v, float))  # noqa: E731
    log.info("trip %s: HMM %s | %.2fs total%s", trip.trip_id, fmt(scores), timings["total_s"],
             " | PLACEHOLDER STAGES IN USE" if pipeline.uses_placeholders else "")
    log.info("trip %s: classifier-only %s", trip.trip_id, fmt(scores_clf))

    return {
        "trip_id": trip.trip_id,
        "uses_placeholders": pipeline.uses_placeholders,
        "placeholders": pipeline.placeholders,
        "hmm_emission": pipeline.emission,
        "metrics": scores,                       # classifier + HMM/Viterbi (final output)
        "metrics_classifier_only": scores_clf,   # P2's best candidate per GPS point, no HMM
        "timings": timings,
        "gps_points": res["gps_points"],
        "gps_times": res["gps_times"],
        "predicted_route_arcs": res["route_arcs"],
        "predicted_route_lines": res["route_lines"],
        "predicted_arc_per_point": res["hmm_pred"].tolist(),
    }
