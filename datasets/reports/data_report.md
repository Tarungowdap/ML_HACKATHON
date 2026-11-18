# PathSnap P1 data report (schema 1.0)
Trips: {'train': 80, 'student': 10}. Search radius 75 m, max 20 candidates per point (closest first).

## Loader, splits, conventions
- `pathsnap.load_trip(trip_id, raw_root=...)` or `load_trip(trip_id, data_dir=...)` returns track, nodes, arcs, route. `splits.json` lists train / student / evaluator trip ids (evaluator list is empty if no such folder was in the zip).
- Order is (lon, lat) in every file. Verified by bounding box on 90/90 trips; a swapped reading failed (see audit_trips.csv). Typical coordinates: lon 11.039, lat 47.693.
- Projection: Local equirectangular tangent plane in metres: x = east, y = north, origin = (lon0, lat0) = mean of the cleaned GPS track of that trip (stored in trips_meta_*.csv). Input degrees are assumed WGS84 lon/lat. Helpers in pathsnap.py: project, unproject, haversine_m. Max relative error vs haversine on steps > 10 m: 8.89e-03.
- IDs: arc_id = row index in that trip's .arcs, node_id = row index in .nodes. Every trip has its own graph (90 unique node files for 90 trips), so IDs are PER TRIP; the global key is (trip_id, arc_id).
- Direction: arcs are directed (from_node -> to_node). On average 92.2% of arcs have a reverse arc.

## Graph and route health
- Consecutive .route arcs: 34,709 connected of 34,723 pairs (99.96%); repeated arc 0; U-turn 0; disconnected 14; invalid ids 0.
- Self-loops 24, zero-length arcs 106, duplicate arcs 1903, isolated nodes 0, bad node refs 0.
- Trips with more than one weakly connected component: 89; with more than one strongly connected component: 90. Longest arc 9308 m (arcs > 500 m are checked against every point).

## GPS timing and cleaning
- Gaps > 1 s: 7,550; > 5 s: 1,260; > 30 s: 135; largest gap 153064 s (per trip in audit_trips.csv: dt_max).
- Cleaning on 227,577 raw rows: invalid coordinates dropped 0; out-of-order steps 0 (stable sort by timestamp if any); duplicate timestamps 0 (kept, dt_prev = 0); duplicate rows 0 (kept). `orig_row` maps back to the raw file.

## Labels
- Per-point labels come from a monotone Viterbi alignment of the GPS points to the .route arcs (position never moves backwards; cost = point-to-arc distance + 0.5 m per route position skipped forward). `true_arc_id` is the aligned arc; `label` = 1 for that arc in the candidates.
- label_quality: far = true arc > 50 m away; ambiguous = another route arc within 5 m of the assigned one; gap = point after a gap > 30 s; ok = none of these.
- Counts (student + train): ok 138,274 (60.76%), ambiguous 65,961, far 23,286, gap 56.
- GPS to true arc distance: mean 20.95 m, 95th percentile 138.88 m.
- The raw .route is kept in routes_*.parquet (trip_id, route_pos, arc_id) for sequence-level metrics.

## Candidate coverage
- Miss rate = share of GPS points whose true arc is not among the candidates. Train 10.922%, student 0.120%. Per trip: miss_rate_per_trip_*.csv (split into radius misses and cap misses).
- Recall vs radius, train:
           radius_m   recall
                 20 0.855516
                 30 0.874924
                 50 0.888402
                 75 0.897340
                100 0.901457
75 (actual, cap 20) 0.890781
- Recall vs radius, student:
           radius_m   recall
                 20 0.971190
                 30 0.992328
                 50 0.998591
                 75 0.998904
                100 0.999113
75 (actual, cap 20) 0.998800

## Functions
- `build_observations_and_candidates(trip_id, track, nodes, arcs)` is label-free and API-safe.
- `attach_labels(obs, cand, nodes, arcs, route, info)` is for train/student data and server-side scoring only.
- Rows sorted by trip_id, obs_idx, cand_rank. Any change to schema or radius: see CHANGELOG.md, announced before files are replaced.
