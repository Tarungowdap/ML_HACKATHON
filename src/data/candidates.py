import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .loader import latlon_to_local_xy
from .label_alignment import align_track_to_route

def build_trip_candidates(trip_data: dict, radius_m: float = 75.0, max_candidates: int = 15, include_labels: bool = True):
    tid = trip_data["trip_id"]
    track = trip_data["track"].copy()
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]
    lat0 = float(track["latitude"].mean())
    lon0 = float(track["longitude"].mean())
    px, py = latlon_to_local_xy(track["longitude"].to_numpy(), track["latitude"].to_numpy(), lat0, lon0)
    nx, ny = latlon_to_local_xy(nodes["longitude"].to_numpy(), nodes["latitude"].to_numpy(), lat0, lon0)
    from_ids = arcs["from_node_id"].to_numpy()
    to_ids = arcs["to_node_id"].to_numpy()
    ax, ay = nx[from_ids], ny[from_ids]
    bx, by = nx[to_ids], ny[to_ids]
    arc_dx = bx - ax
    arc_dy = by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_heading = np.arctan2(arc_dy, arc_dx)
    mid_x = 0.5 * (ax + bx)
    mid_y = 0.5 * (ay + by)
    n_nodes = len(nodes)
    out_deg = np.bincount(from_ids, minlength=n_nodes)
    in_deg = np.bincount(to_ids, minlength=n_nodes)
    arc_from_out_deg = out_deg[from_ids]
    arc_to_out_deg = out_deg[to_ids]
    arc_from_in_deg = in_deg[from_ids]
    ts = track["timestamp"].to_numpy(dtype=np.float64)
    dt = np.diff(ts, prepend=ts[0])
    dt[0] = 1.0
    dt = np.where(dt <= 0, 1.0, dt)
    step_dx = np.diff(px, prepend=px[0])
    step_dy = np.diff(py, prepend=py[0])
    if len(px) > 1:
        step_dx[0] = px[1] - px[0]
        step_dy[0] = py[1] - py[0]
    step_dist = np.hypot(step_dx, step_dy)
    speed_mps = step_dist / dt
    accel_mps2 = np.diff(speed_mps, prepend=speed_mps[0]) / dt
    gps_heading = np.arctan2(step_dy, step_dx)
    true_arc_per_obs = None
    if include_labels and route is not None:
        true_arc_per_obs = align_track_to_route(track, nodes, arcs, route)
    max_half_len = float(np.percentile(arc_len, 99) * 0.5) + 50.0
    tree = cKDTree(np.column_stack([mid_x, mid_y]))
    nearby_lists = tree.query_ball_point(np.column_stack([px, py]), r=radius_m + max_half_len)
    rows = []
    missed_obs = 0
    total_obs = len(track)
    for i in range(total_obs):
        cand_ids = np.asarray(nearby_lists[i], dtype=np.int32)
        if len(cand_ids) == 0:
            _, cand_ids = tree.query([px[i], py[i]], k=min(5, len(arcs)))
            cand_ids = np.atleast_1d(cand_ids).astype(np.int32)
        c_ax, c_ay = ax[cand_ids], ay[cand_ids]
        c_dx, c_dy = arc_dx[cand_ids], arc_dy[cand_ids]
        c_len_sq = np.where(arc_len[cand_ids] ** 2 < 1e-9, 1.0, arc_len[cand_ids] ** 2)
        t = ((px[i] - c_ax) * c_dx + (py[i] - c_ay) * c_dy) / c_len_sq
        t_clamped = np.clip(t, 0.0, 1.0)
        proj_x = c_ax + t_clamped * c_dx
        proj_y = c_ay + t_clamped * c_dy
        perp_dist = np.hypot(px[i] - proj_x, py[i] - proj_y)
        keep_mask = perp_dist <= radius_m
        if not np.any(keep_mask):
            order = np.argsort(perp_dist)[:max_candidates]
        else:
            valid_idx = np.where(keep_mask)[0]
            order = valid_idx[np.argsort(perp_dist[valid_idx])[:max_candidates]]
        sel_arcs = cand_ids[order]
        sel_dist = perp_dist[order]
        sel_t = t_clamped[order]
        if true_arc_per_obs is not None:
            if true_arc_per_obs[i] not in sel_arcs:
                missed_obs += 1
        angle_diff = np.abs(np.arctan2(np.sin(gps_heading[i] - arc_heading[sel_arcs]), np.cos(gps_heading[i] - arc_heading[sel_arcs])))
        cos_heading_diff = np.cos(angle_diff)
        sin_heading_diff = np.sin(angle_diff)
        min_d = sel_dist.min()
        dist_rank = np.arange(len(sel_arcs), dtype=np.float32)
        dist_diff_from_best = sel_dist - min_d
        dist_ratio_to_best = (sel_dist + 0.1) / (min_d + 0.1)
        gaussian_emission = np.exp(-0.5 * (sel_dist / 10.0) ** 2)
        for k in range(len(sel_arcs)):
            aid = int(sel_arcs[k])
            row = {"trip_id": tid, "obs_idx": int(i), "arc_id": aid}
            if true_arc_per_obs is not None:
                row["label"] = int(aid == true_arc_per_obs[i])
            row.update({
                "perp_dist_m": float(sel_dist[k]), "proj_ratio_t": float(sel_t[k]),
                "arc_length_m": float(arc_len[aid]), "gaussian_emission": float(gaussian_emission[k]),
                "heading_diff_rad": float(angle_diff[k]), "cos_heading_diff": float(cos_heading_diff[k]),
                "sin_heading_diff": float(sin_heading_diff[k]), "dt_sec": float(dt[i]),
                "step_dist_m": float(step_dist[i]), "speed_mps": float(speed_mps[i]),
                "accel_mps2": float(accel_mps2[i]), "is_stationary": int(speed_mps[i] < 0.5),
                "from_out_deg": int(arc_from_out_deg[aid]), "to_out_deg": int(arc_to_out_deg[aid]),
                "from_in_deg": int(arc_from_in_deg[aid]), "dist_rank": float(dist_rank[k]),
                "dist_diff_from_best_m": float(dist_diff_from_best[k]), "dist_ratio_to_best": float(dist_ratio_to_best[k]),
                "num_candidates": int(len(sel_arcs)),
            })
            rows.append(row)
    stats = {"trip_id": tid, "total_obs": total_obs, "missed_obs": missed_obs, "candidate_miss_rate": missed_obs / max(total_obs, 1), "avg_candidates_per_obs": len(rows) / max(total_obs, 1)}
    return pd.DataFrame(rows), stats
