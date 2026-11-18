import numpy as np
from .loader import latlon_to_local_xy

def point_to_segments_dist_m(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(len_sq < 1e-9, 1.0, len_sq)
    t = ((px[:, None] - ax[None, :]) * dx[None, :] + (py[:, None] - ay[None, :]) * dy[None, :]) / safe_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax[None, :] + t * dx[None, :]
    proj_y = ay[None, :] + t * dy
    dist = np.hypot(px[:, None] - proj_x, py[:, None] - proj_y)
    return dist, t

def align_track_to_route(track_df, nodes_df, arcs_df, route_arcs: np.ndarray) -> np.ndarray:
    lat0 = float(track_df["latitude"].mean())
    lon0 = float(track_df["longitude"].mean())
    px, py = latlon_to_local_xy(track_df["longitude"].to_numpy(), track_df["latitude"].to_numpy(), lat0, lon0)
    nx, ny = latlon_to_local_xy(nodes_df["longitude"].to_numpy(), nodes_df["latitude"].to_numpy(), lat0, lon0)
    from_ids = arcs_df["from_node_id"].to_numpy()[route_arcs]
    to_ids = arcs_df["to_node_id"].to_numpy()[route_arcs]
    ax, ay = nx[from_ids], ny[from_ids]
    bx, by = nx[to_ids], ny[to_ids]
    dist_mat, _ = point_to_segments_dist_m(px, py, ax, ay, bx, by)
    N, M = dist_mat.shape
    dp = np.full((N, M), np.inf, dtype=np.float64)
    back = np.zeros((N, M), dtype=np.int32)
    dp[0] = dist_mat[0]
    for i in range(1, N):
        prefix_min = np.minimum.accumulate(dp[i - 1])
        best_k = np.zeros(M, dtype=np.int32)
        cur_idx = 0
        for j in range(1, M):
            if dp[i - 1, j] < dp[i - 1, cur_idx]:
                cur_idx = j
            best_k[j] = cur_idx
        dp[i] = dist_mat[i] + prefix_min + 0.5 * (np.arange(M) - best_k)
        back[i] = best_k
    seq_idx = np.zeros(N, dtype=np.int32)
    seq_idx[-1] = int(np.argmin(dp[-1]))
    for i in range(N - 2, -1, -1):
        seq_idx[i] = back[i + 1, seq_idx[i + 1]]
    return route_arcs[seq_idx]
