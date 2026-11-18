# PathSnap: Distributed Map-Matching

ML hackathon project (PES University, UE24CS352A). Given a noisy 1 Hz GPS track, match each point to a road-network arc, then reconstruct the full route with an HMM/Viterbi decoder.

Final demo: FastAPI backend (Laptop 1) + UI (Laptop 2), evaluated on 10 hidden trips whose ground truth stays server-side.

## Pipeline

```
GPS track ──► candidate arcs + features (P1) ──► ML scorer (P2) ──► HMM / Viterbi (P3) ──► route
                                                                         │
                                                              FastAPI + UI (P4)
```

1. **Candidate generation (P1):** for each GPS point, find nearby road arcs and compute numeric features (distance, heading difference, etc.).
2. **Candidate scoring (P2):** a classifier outputs the probability that a candidate arc is the true arc. This feeds the HMM as the emission score.
3. **Route decoding (P3):** Viterbi over candidates with a transition model; the result is shown on a Folium map.
4. **Serving (P4):** FastAPI endpoint and UI; the evaluator scores the returned routes.

## Data

Per trip: `.track` (lon, lat, timestamp), `.nodes` (lon, lat), `.arcs` (from_node, to_node; row index = arc_id), `.route` (ground-truth arc_id sequence).

| Split | Trips | Use |
|---|---|---|
| Train | 80 | model training, group CV |
| Student test | 10 | model comparison |
| Evaluator | 10 | final demo only, never used in development |

Splits are always **by trip, never by row**.

## Repository layout

| Folder | Contents |
|---|---|
| `datasets/` | raw trip files |
| `processed_datasets/` | P1's candidate tables (`candidates_train.parquet`, `candidates_student.parquet`) |
| `notebooks/` | `P2_models_bayesopt.ipynb` |
| `models/` | trained models and feature list |
| `results/` | comparison tables, plots, Optuna outputs |

## P2: Model experimentation and Bayesian optimisation

Notebook: `notebooks/P2_models_bayesopt.ipynb` (runs on Google Colab).

**Input:** candidate tables with `trip_id, obs_idx, arc_id, label` (label = 1 if the candidate is the reference arc) plus numeric features. Ids, absolute coordinates (`snap_x`, `snap_y`, `from_node`, `to_node`) and pre-scaled duplicates (`*_scaled`) are excluded from the features.

**Models (5 families):**

| Model | Family |
|---|---|
| Logistic Regression | linear |
| Decision Tree | single tree |
| Random Forest | bagging |
| LightGBM | boosting |
| MLP | neural network |

**Evaluation**
- GroupKFold CV by trip on the training trips (current config: 3 folds on a fixed-seed 30-trip subset; update here if rerun on all 80).
- Metrics: candidate-level macro F1, precision/recall, log loss (threshold 0.5); observation-level Top-1 (a missing reference arc counts as wrong); route continuity (needs per-trip arc adjacency, otherwise reported as N/A); inference time; train vs CV vs test gap.
- Final comparison on the 10 student-test trips.
- Plots: model comparison bars, confusion matrices, MLP loss curve (only the MLP has an epoch history, so no curves for other models).

**Bayesian optimisation:** Optuna TPE on LightGBM, objective = mean CV macro F1 (current config: 20 trials, 15-minute timeout; the actual number completed is in `results/optuna_summary.json`). The tuned model is compared against the untuned one on the student-test trips and refit on all 80 training trips for the final model.

### Results

Numbers are produced by the notebook and saved in `results/`. See:

- `results/model_comparison.csv`
- `results/tuned_vs_untuned.csv`
- `results/optuna_trials.csv`, `results/best_params.json`, `results/optuna_summary.json`

| Model | CV macro F1 | Student Top-1 | Student macro F1 | Log loss |
|---|---|---|---|---|
| _fill in from `model_comparison.csv`_ | | | | |

| | CV macro F1 | Student Top-1 | Student macro F1 | Log loss |
|---|---|---|---|---|
| Untuned LightGBM | _from `tuned_vs_untuned.csv`_ | | | |
| Tuned LightGBM | | | | |

> Only paste values from real runs. The tuned CV score is optimistic because Optuna chose its parameters on the same folds.

### Using the final model (for P3 / P4)

Files: `models/final_tuned_raw.joblib` (tuned LightGBM, refit on all 80 train trips) and `models/features_raw.json` (exact feature order).

```python
import joblib, json, numpy as np

model = joblib.load("models/final_tuned_raw.joblib")
FEATURES = json.load(open("models/features_raw.json"))

p = model.predict_proba(df[FEATURES].values.astype(np.float32))[:, 1]
emission = np.log(np.clip(p, 1e-7, 1.0))   # HMM emission score per candidate
```

Requirements: columns must be in `FEATURES` order, unscaled, produced by the same candidate generator used for training. Use the same `lightgbm` and `scikit-learn` versions as the training environment (see `requirements_p2.txt`).

### Reproducing

1. Open the notebook in Colab.
2. Upload `candidates_train.parquet` and `candidates_student.parquet` when prompted (or place them next to the notebook).
3. Run all cells. Outputs are written to `results/` and `models/`, and zipped as `p2_outputs.zip`.

Seed is fixed (`SEED = 42`), and the subset trips and config are saved in `results/run_config.json`.

## P1: Data and candidate generation

_To be filled in by P1: candidate radius, feature definitions, miss rate._

## P3: HMM / Viterbi

_To be filled in by P3: emission and transition model, parameters, route continuity._

## P4: API, UI and deployment

_To be filled in by P4: endpoints, how to run the backend and UI, deployment steps._

## Team

| Role | Member |
|---|---|
| P1 Data / candidates / features | |
| P2 Models + Bayesian optimisation | |
| P3 HMM / Viterbi + Folium | |
| P4 API / UI / deploy / report | |

## Rules followed

- All reported metrics come from real runs; nothing is fabricated.
- Splits are by trip; evaluator trips are never loaded during development.
- No loss curves for models without epoch histories.
