"""PathSnap shared helpers, schema 1.0. All coordinates are (lon, lat) WGS84 degrees."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SCHEMA_VERSION = "1.0"
RADIUS_M = 75.0          # candidate search radius (point-to-segment distance)
MAX_CANDIDATES = 20      # max candidates kept per GPS point (closest first)
GAP_S = 30.0             # dt_prev > GAP_S  -> is_gap
LONG_ARC_M = 500.0       # arcs longer than this are checked against every point directly
AMBIG_M = 5.0            # another route arc within 5 m of the assigned arc -> 'ambiguous'
FAR_M = 50.0             # assigned arc farther than this -> 'far'
ALIGN_LAMBDA = 0.5       # metres of penalty per route position skipped forward
RECALL_RADII = (20, 30, 50, 75, 100)
M_PER_DEG_LAT = 111132.95
M_PER_DEG_LON_EQ = 111319.49
PROJECTION = ("Local equirectangular tangent plane in metres: x = east, y = north, origin = (lon0, lat0) "
              "= mean of the cleaned GPS track of that trip (stored in trips_meta_*.csv). "
              "Input degrees are assumed WGS84 lon/lat.")

OBS_COLS_LABELED = ["trip_id", "obs_idx", "orig_row", "timestamp", "dt_prev", "is_gap", "lon", "lat", "x", "y",
                    "speed", "heading", "n_candidates", "true_arc_id", "label_quality", "heading_valid", "true_dist_m"]
CAND_COLS = ["trip_id", "obs_idx", "arc_id", "from_node", "to_node", "dist_to_arc_m", "snap_x", "snap_y",
             "offset_m", "arc_length_m", "bearing_diff_deg", "cand_rank", "label"]

# ---------------- geometry helpers ----------------
def project(lon, lat, lat0, lon0):
    lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
    return ((lon - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)), (lat - lat0) * M_PER_DEG_LAT)

def unproject(x, y, lat0, lon0):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    return (lon0 + x / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))), lat0 + y / M_PER_DEG_LAT)

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def point_segment_dist(px, py, ax, ay, bx, by):
    """Distance (m) from every point (N) to every segment (M) -> (N, M)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    l2s = np.where(l2 < 1e-9, 1.0, l2)
    t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / l2s, 0.0, 1.0)
    return np.hypot(px[:, None] - (ax + t * dx), py[:, None] - (ay + t * dy))

def _bearing_deg(dx, dy):
    """Compass bearing, degrees clockwise from north, in [0, 360)."""
    return (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0

# ---------------- loading ----------------
def _read(path, names):
    return pd.read_csv(path, header=None, sep=r"\s+", names=names, usecols=range(len(names)))

def find_trip_dir(trip_id, raw_root):
    raw_root = Path(raw_root)
    for sub in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if (sub / trip_id).is_dir():
            return sub / trip_id
    if (raw_root / trip_id).is_dir():
        return raw_root / trip_id
    raise FileNotFoundError(trip_id)

def load_trip(trip_id, raw_root=None, data_dir=None):
    """Return {'trip_id','track','nodes','arcs','route'}.
    raw_root: folder containing train/ student_test/ ... (raw .track/.nodes/.arcs/.route files)
    data_dir: folder with the delivered parquets (works without the raw zip)
    track: longitude, latitude, timestamp (+orig_row in parquet mode); nodes: longitude, latitude;
    arcs: from_node_id, to_node_id, arc_id (= row index in .arcs, PER TRIP); route: int array of arc_id or None."""
    if raw_root is not None:
        d = find_trip_dir(trip_id, raw_root)
        track = _read(d / f"{trip_id}.track", ["longitude", "latitude", "timestamp"])
        nodes = _read(d / f"{trip_id}.nodes", ["longitude", "latitude"])
        arcs = _read(d / f"{trip_id}.arcs", ["from_node_id", "to_node_id"])
        arcs["arc_id"] = np.arange(len(arcs), dtype=np.int32)
        rf = d / f"{trip_id}.route"
        route = _read(rf, ["arc_id"])["arc_id"].to_numpy(np.int64) if rf.exists() and rf.stat().st_size > 0 else None
        return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
    if data_dir is not None:
        data_dir = Path(data_dir)
        for split in ("train", "student"):
            fo = data_dir / f"observations_{split}.parquet"
            if not fo.exists():
                continue
            flt = [("trip_id", "==", trip_id)]
            obs = pd.read_parquet(fo, filters=flt)
            if len(obs) == 0:
                continue
            arcs = pd.read_parquet(data_dir / f"arcs_{split}.parquet", filters=flt)
            nodes = pd.read_parquet(data_dir / f"nodes_{split}.parquet", filters=flt)
            rt = pd.read_parquet(data_dir / f"routes_{split}.parquet", filters=flt).sort_values("route_pos")
            track = obs[["orig_row", "lon", "lat", "timestamp"]].rename(columns={"lon": "longitude", "lat": "latitude"})
            return {"trip_id": trip_id, "track": track.reset_index(drop=True),
                    "nodes": nodes.rename(columns={"lon": "longitude", "lat": "latitude"})[["node_id", "longitude", "latitude"]].reset_index(drop=True),
                    "arcs": arcs.rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})[["arc_id", "from_node_id", "to_node_id"]].reset_index(drop=True),
                    "route": rt["arc_id"].to_numpy(np.int64) if len(rt) else None}
        raise FileNotFoundError(trip_id)
    raise ValueError("pass raw_root= or data_dir=")

# ---------------- cleaning ----------------
def clean_track(track):
    """Drops invalid coordinates, stable-sorts by timestamp if out of order. Duplicate timestamps are KEPT
    (dt_prev == 0). orig_row = 0-based row in the raw .track file."""
    t = track[["longitude", "latitude", "timestamp"]].copy().reset_index(drop=True)
    t["orig_row"] = np.arange(len(t), dtype=np.int64)
    for c in ("longitude", "latitude", "timestamp"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    bad = (t.isna().any(axis=1) | (t.longitude.abs() > 180) | (t.latitude.abs() > 90)
           | ((t.longitude == 0) & (t.latitude == 0)))
    log = {"n_raw_rows": int(len(t)), "n_invalid_coord": int(bad.sum())}
    t = t.loc[~bad].reset_index(drop=True)
    log["n_out_of_order"] = int((np.diff(t.timestamp.to_numpy(float)) < 0).sum())
    if log["n_out_of_order"]:
        t = t.sort_values("timestamp", kind="stable").reset_index(drop=True)
    log["n_dup_timestamp"] = int((np.diff(t.timestamp.to_numpy(float)) == 0).sum())
    log["n_dup_rows"] = int(t.duplicated(["longitude", "latitude", "timestamp"]).sum())
    return t, log

# ---------------- label-free build (API-safe) ----------------
def build_observations_and_candidates(trip_id, track, nodes, arcs, radius_m=RADIUS_M,
                                      max_candidates=MAX_CANDIDATES, gap_s=GAP_S):
    """LABEL-FREE. Needs only track, nodes, arcs. Returns (obs_df, cand_df, info)."""
    t, clog = clean_track(track)
    n = len(t)
    if n == 0:
        raise ValueError(f"{trip_id}: no valid GPS rows")
    lon = t.longitude.to_numpy(float); lat = t.latitude.to_numpy(float); ts = t.timestamp.to_numpy(float)
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    px, py = project(lon, lat, lat0, lon0)
    nx, ny = project(nodes.longitude.to_numpy(float), nodes.latitude.to_numpy(float), lat0, lon0)
    fr = arcs.from_node_id.to_numpy(np.int64); to = arcs.to_node_id.to_numpy(np.int64)
    ax, ay, bx, by = nx[fr], ny[fr], nx[to], ny[to]
    dx, dy = bx - ax, by - ay
    alen = np.hypot(dx, dy)
    abear = _bearing_deg(dx, dy)

    dtp = np.diff(ts, prepend=ts[0]).astype(float)            # 0 for the first point
    is_gap = dtp > gap_s
    sdx = np.diff(px, prepend=px[0]); sdy = np.diff(py, prepend=py[0])
    step = np.hypot(sdx, sdy)
    speed = step / np.maximum(dtp, 1.0)                       # dt floored at 1 s (handles dt = 0)
    valid = step >= 1.0; valid[0] = False                     # heading only trusted if moved >= 1 m
    head = np.where(valid, _bearing_deg(sdx, sdy), np.nan)
    head = pd.Series(head).ffill().bfill().fillna(0.0).to_numpy()   # carry forward when stationary

    pts = np.column_stack([px, py])
    mid = np.column_stack([0.5 * (ax + bx), 0.5 * (ay + by)])
    is_long = alen > LONG_ARC_M
    short_ids = np.where(~is_long)[0]; long_ids = np.where(is_long)[0]
    if len(short_ids):
        tree = cKDTree(mid[short_ids])
        near = tree.query_ball_point(pts, r=radius_m + 0.5 * float(alen[short_ids].max()))
    else:
        near = [[] for _ in range(n)]

    C = {k: [] for k in ("obs_idx", "arc_id", "dist", "sx", "sy", "off", "bd", "rank")}
    ncand = np.zeros(n, dtype=np.int16); n_trunc = 0
    for i in range(n):
        ids = short_ids[np.asarray(near[i], dtype=np.int64)] if len(short_ids) else np.empty(0, np.int64)
        if long_ids.size:
            ids = np.concatenate([ids, long_ids])
        if ids.size == 0:
            continue
        l2 = alen[ids] ** 2
        tt = np.clip(((px[i] - ax[ids]) * dx[ids] + (py[i] - ay[ids]) * dy[ids]) / np.where(l2 < 1e-9, 1.0, l2), 0.0, 1.0)
        sx = ax[ids] + tt * dx[ids]; sy = ay[ids] + tt * dy[ids]
        d = np.hypot(px[i] - sx, py[i] - sy)
        m = d <= radius_m
        if not m.any():
            continue
        ids, tt, sx, sy, d = ids[m], tt[m], sx[m], sy[m], d[m]
        if len(ids) > max_candidates:
            n_trunc += 1
        o = np.lexsort((ids, d))[:max_candidates]           # closest first, ties by arc_id
        ids, tt, sx, sy, d = ids[o], tt[o], sx[o], sy[o], d[o]
        k = len(ids); ncand[i] = k
        C["obs_idx"].append(np.full(k, i, np.int32)); C["arc_id"].append(ids.astype(np.int32))
        C["dist"].append(d.astype(np.float32)); C["sx"].append(sx.astype(np.float32)); C["sy"].append(sy.astype(np.float32))
        C["off"].append((tt * alen[ids]).astype(np.float32))
        C["bd"].append((np.abs((head[i] - abear[ids] + 180.0) % 360.0 - 180.0)).astype(np.float32))
        C["rank"].append(np.arange(k, dtype=np.int16))

    cat = lambda k, dt: np.concatenate(C[k]) if C[k] else np.empty(0, dt)
    aid = cat("arc_id", np.int32)
    cand = pd.DataFrame({
        "trip_id": trip_id, "obs_idx": cat("obs_idx", np.int32), "arc_id": aid,
        "from_node": fr[aid].astype(np.int32), "to_node": to[aid].astype(np.int32),
        "dist_to_arc_m": cat("dist", np.float32), "snap_x": cat("sx", np.float32), "snap_y": cat("sy", np.float32),
        "offset_m": cat("off", np.float32), "arc_length_m": alen[aid].astype(np.float32),
        "bearing_diff_deg": cat("bd", np.float32), "cand_rank": cat("rank", np.int16)})
    obs = pd.DataFrame({
        "trip_id": trip_id, "obs_idx": np.arange(n, dtype=np.int32), "orig_row": t["orig_row"].to_numpy(np.int32),
        "timestamp": ts, "dt_prev": dtp.astype(np.float32), "is_gap": is_gap, "lon": lon, "lat": lat,
        "x": px, "y": py, "speed": speed.astype(np.float32), "heading": head.astype(np.float32),
        "n_candidates": ncand, "heading_valid": valid})
    info = {"trip_id": trip_id, "lat0": lat0, "lon0": lon0, "n_obs": n, "cleaning": clog,
            "n_zero_cand": int((ncand == 0).sum()), "n_truncated": int(n_trunc),
            "radius_m": radius_m, "max_candidates": max_candidates}
    return obs, cand, info

# ---------------- labels (train/student/server-side only) ----------------
def _monotone_align(D, lam=ALIGN_LAMBDA):
    """Viterbi over route positions: position never decreases; moving forward by k positions costs lam*k.
    D: (N points, M route arcs) distances. Returns route position per point."""
    N, M = D.shape
    idx = np.arange(M)
    dp = D[0].copy()
    back = np.zeros((N, M), dtype=np.int32)
    for i in range(1, N):
        tmp = dp - lam * idx
        cm = np.minimum.accumulate(tmp)
        back[i] = np.maximum.accumulate(np.where(tmp <= cm, idx, 0))
        dp = D[i] + lam * idx + cm
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(dp))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos

def attach_labels(obs, cand, nodes, arcs, route, info, radius_m=RADIUS_M, recall_radii=RECALL_RADII):
    """Adds true_arc_id, label_quality, true_dist_m to obs and label to cand. Returns (obs, cand, lab_info)."""
    obs = obs.copy(); cand = cand.copy(); n = len(obs)
    empty = {"labeled": False, "n_labeled": 0, "d_true": np.empty(0), "n_hit": 0, "n_miss": 0,
             "n_miss_radius": 0, "n_miss_cap": 0, "quality": {}, "recall_counts": {r: 0 for r in recall_radii},
             "n_route_invalid": 0}
    if route is None or len(route) == 0 or n == 0:
        obs["true_arc_id"] = np.int32(-1); obs["label_quality"] = "unlabeled"; obs["true_dist_m"] = np.float32(np.nan)
        cand["label"] = np.int8(0)
        return obs[OBS_COLS_LABELED], cand[CAND_COLS], empty
    route = np.asarray(route, dtype=np.int64)
    ok = (route >= 0) & (route < len(arcs))
    ra = route[ok]
    nx, ny = project(nodes.longitude.to_numpy(float), nodes.latitude.to_numpy(float), info["lat0"], info["lon0"])
    fr = arcs.from_node_id.to_numpy(np.int64); to = arcs.to_node_id.to_numpy(np.int64)
    D = point_segment_dist(obs.x.to_numpy(float), obs.y.to_numpy(float), nx[fr[ra]], ny[fr[ra]], nx[to[ra]], ny[to[ra]])
    pos = _monotone_align(D)
    true_arc = ra[pos]
    ar = np.arange(n)
    d_true = D[ar, pos]
    D2 = D.copy(); D2[ra[None, :] == true_arc[:, None]] = np.inf
    d_alt = D2.min(axis=1)
    quality = np.select([d_true > FAR_M, (d_alt - d_true) < AMBIG_M, obs.is_gap.to_numpy()],
                        ["far", "ambiguous", "gap"], default="ok")
    obs["true_arc_id"] = true_arc.astype(np.int32)
    obs["label_quality"] = quality
    obs["true_dist_m"] = d_true.astype(np.float32)
    cand["label"] = (cand.arc_id.to_numpy() == true_arc[cand.obs_idx.to_numpy()]).astype(np.int8)
    hit = np.zeros(n, bool)
    hit[cand.obs_idx.to_numpy()[cand.label.to_numpy() == 1]] = True
    miss = ~hit; miss_r = miss & (d_true > radius_m); miss_c = miss & ~miss_r
    lab = {"labeled": True, "n_labeled": n, "d_true": d_true, "n_hit": int(hit.sum()), "n_miss": int(miss.sum()),
           "n_miss_radius": int(miss_r.sum()), "n_miss_cap": int(miss_c.sum()),
           "quality": {q: int((quality == q).sum()) for q in ("ok", "ambiguous", "far", "gap")},
           "recall_counts": {r: int((d_true <= r).sum()) for r in recall_radii},
           "n_route_invalid": int((~ok).sum())}
    return obs[OBS_COLS_LABELED], cand[CAND_COLS], lab

# ---------------- per-trip graph tables / audits ----------------
def graph_tables(trip_id, nodes, arcs, route, lat0, lon0):
    nlon = nodes.longitude.to_numpy(float); nlat = nodes.latitude.to_numpy(float)
    nx, ny = project(nlon, nlat, lat0, lon0)
    f = arcs.from_node_id.to_numpy(np.int64); t = arcs.to_node_id.to_numpy(np.int64)
    a = pd.DataFrame({"trip_id": trip_id, "arc_id": arcs.arc_id.to_numpy(np.int32),
                      "from_node": f.astype(np.int32), "to_node": t.astype(np.int32),
                      "from_lon": nlon[f], "from_lat": nlat[f], "to_lon": nlon[t], "to_lat": nlat[t],
                      "length_m": np.hypot(nx[t] - nx[f], ny[t] - ny[f]).astype(np.float32)})
    nd = pd.DataFrame({"trip_id": trip_id, "node_id": np.arange(len(nodes), dtype=np.int32),
                       "lon": nlon, "lat": nlat, "x": nx, "y": ny})
    rt = pd.DataFrame({"trip_id": trip_id, "route_pos": np.arange(0 if route is None else len(route), dtype=np.int32),
                       "arc_id": np.asarray([] if route is None else route, dtype=np.int32)})
    return a, nd, rt

def lonlat_order_check(track, nodes, pad=0.05):
    """Fraction of GPS points inside the node bounding box as (lon,lat) vs swapped (lat,lon)."""
    def inside(a, b, A, B):
        return float(((a >= A.min() - pad) & (a <= A.max() + pad) & (b >= B.min() - pad) & (b <= B.max() + pad)).mean())
    tl, tb = track.longitude.to_numpy(float), track.latitude.to_numpy(float)
    nl, nb = nodes.longitude.to_numpy(float), nodes.latitude.to_numpy(float)
    return inside(tl, tb, nl, nb), inside(tl, tb, nb, nl)

def graph_audit(nodes, arcs, route):
    n_nodes, n_arcs = len(nodes), len(arcs)
    f = arcs.from_node_id.to_numpy(np.int64); t = arcs.to_node_id.to_numpy(np.int64)
    out = {"n_nodes": n_nodes, "n_arcs": n_arcs,
           "arcs_bad_node_ref": int(((f < 0) | (f >= n_nodes) | (t < 0) | (t >= n_nodes)).sum())}
    fc, tc = np.clip(f, 0, n_nodes - 1), np.clip(t, 0, n_nodes - 1)
    lat0, lon0 = float(nodes.latitude.mean()), float(nodes.longitude.mean())
    nx, ny = project(nodes.longitude.to_numpy(float), nodes.latitude.to_numpy(float), lat0, lon0)
    length = np.hypot(nx[tc] - nx[fc], ny[tc] - ny[fc])
    out.update(self_loops=int((f == t).sum()), zero_len_arcs=int((length < 0.01).sum()),
               duplicate_arcs=int(pd.Series(fc * n_nodes + tc).duplicated().sum()),
               arc_len_p50=float(np.median(length)), arc_len_p99=float(np.percentile(length, 99)), arc_len_max=float(length.max()))
    key = fc * n_nodes + tc
    ns = f != t
    out["frac_arcs_with_reverse"] = float(np.isin(tc[ns] * n_nodes + fc[ns], key).mean()) if ns.any() else 0.0
    used = np.zeros(n_nodes, bool); used[fc] = True; used[tc] = True
    out["isolated_nodes"] = int((~used).sum())
    g = coo_matrix((np.ones(n_arcs), (fc, tc)), shape=(n_nodes, n_nodes))
    for nm, kw in (("weak", dict(directed=False)), ("strong", dict(directed=True, connection="strong"))):
        nc, lab = connected_components(g, **kw)
        sizes = np.bincount(lab[used], minlength=nc); sizes = sizes[sizes > 0]
        out[f"n_{nm}_components"] = int(len(sizes)); out[f"largest_{nm}_frac"] = float(sizes.max() / sizes.sum())
    z = dict(route_len=0, route_pairs=0, route_repeat=0, route_connected=0, route_uturn=0, route_disconnected=0, route_invalid_ids=0)
    if route is not None and len(route) > 1:
        r = np.asarray(route, dtype=np.int64); okr = (r >= 0) & (r < n_arcs)
        both = okr[:-1] & okr[1:]; a, b = r[:-1][both], r[1:][both]
        rep = a == b; con = (~rep) & (t[a] == f[b])
        ut = (~rep) & (~con) & (t[b] == f[a]) & (f[b] == t[a])
        z = dict(route_len=int(len(r)), route_pairs=int(len(a)), route_repeat=int(rep.sum()), route_connected=int(con.sum()),
                 route_uturn=int(ut.sum()), route_disconnected=int((~rep & ~con & ~ut).sum()), route_invalid_ids=int((~okr).sum()))
    out.update(z)
    return out

def route_continuity(arcs_tbl, arc_seq):
    """Fraction of consecutive predicted arcs that are the same arc or directly connected (to_node == next from_node).
    arcs_tbl: arcs_{split}.parquet rows of ONE trip (needs arc_id, from_node, to_node)."""
    tb = arcs_tbl.set_index("arc_id"); seq = list(arc_seq)
    ok = [(a == b) or (tb.at[a, "to_node"] == tb.at[b, "from_node"]) for a, b in zip(seq[:-1], seq[1:])]
    return sum(ok) / max(len(ok), 1)
