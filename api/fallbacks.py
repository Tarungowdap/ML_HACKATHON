"""PLACEHOLDER pipeline stages.

These exist only so the API and UI can be built and tested before the real
P1 / P2 / P3 modules are plugged in. They are deliberately simple and are
NOT the team's submitted method. The API marks every response that used one
of these with "uses_placeholders": true.
"""
import numpy as np
import pandas as pd

EARTH_R = 6_371_000.0


def to_local_xy(lon, lat, lon0, lat0):
    """Equirectangular projection to metres around (lon0, lat0); fine at city scale."""
    k = np.pi / 180.0
    x = (np.asarray(lon) - lon0) * k * EARTH_R * np.cos(lat0 * k)
    y = (np.asarray(lat) - lat0) * k * EARTH_R
    return x, y


def build_candidates(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                     k: int = 15, radius_m: float = 150.0) -> pd.DataFrame:
    """Nearest-k directed arcs per GPS point, with distance and heading difference."""
    lon0, lat0 = track.lon.mean(), track.lat.mean()
    px, py = to_local_xy(track.lon.values, track.lat.values, lon0, lat0)
    nx, ny = to_local_xy(nodes.lon.values, nodes.lat.values, lon0, lat0)

    f, t = arcs.from_node.values, arcs.to_node.values
    ax, ay, bx, by = nx[f], ny[f], nx[t], ny[t]
    # keep only arcs near the trajectory's bounding box
    m = ((np.maximum(ax, bx) > px.min() - radius_m) & (np.minimum(ax, bx) < px.max() + radius_m) &
         (np.maximum(ay, by) > py.min() - radius_m) & (np.minimum(ay, by) < py.max() + radius_m))
    arc_ids = np.nonzero(m)[0]
    ax, ay, bx, by = ax[m], ay[m], bx[m], by[m]
    dx, dy = bx - ax, by - ay
    seg_len2 = np.maximum(dx * dx + dy * dy, 1e-9)
    arc_heading = np.arctan2(dy, dx)

    # GPS heading from consecutive points (forward difference, last copies previous)
    hx, hy = np.diff(px, append=px[-1]), np.diff(py, append=py[-1])
    if len(px) > 1:
        hx[-1], hy[-1] = hx[-2], hy[-2]
    gps_heading = np.arctan2(hy, hx)

    rows = []
    chunk = max(1, 2_000_000 // max(len(arc_ids), 1))
    for s in range(0, len(px), chunk):
        e = min(s + chunk, len(px))
        qx, qy = px[s:e, None], py[s:e, None]
        tt = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg_len2, 0.0, 1.0)
        d = np.hypot(qx - (ax + tt * dx), qy - (ay + tt * dy))
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for r in range(e - s):
            j = idx[r][d[r, idx[r]] <= radius_m]
            if j.size == 0:
                j = idx[r][[np.argmin(d[r, idx[r]])]]
            hd = np.abs((gps_heading[s + r] - arc_heading[j] + np.pi) % (2 * np.pi) - np.pi)
            rows.append(pd.DataFrame({
                "obs_idx": s + r, "arc_id": arc_ids[j],
                "perp_dist_m": d[r, j], "proj_ratio_t": tt[r, j], "heading_diff_rad": hd,
            }))
    return pd.concat(rows, ignore_index=True)


def score_candidates(cands: pd.DataFrame) -> np.ndarray:
    """Hand-set score: Gaussian on distance x heading agreement. Not a trained model."""
    sigma = 10.0
    return np.exp(-0.5 * (cands.perp_dist_m.values / sigma) ** 2) * (1 + np.cos(cands.heading_diff_rad.values)) / 2


def decode(cands: pd.DataFrame, track, nodes, arcs) -> np.ndarray:
    """Per-point argmax, no HMM. Returns one arc_id per obs_idx (-1 where no candidate)."""
    best = cands.loc[cands.groupby("obs_idx")["prob"].idxmax(), ["obs_idx", "arc_id"]]
    out = np.full(len(track), -1, dtype=np.int64)
    out[best.obs_idx.values] = best.arc_id.values
    return out
