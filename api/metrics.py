"""Server-side evaluation. The reference route is read here and nowhere else;
only the resulting numbers are returned to the UI."""
import numpy as np


def collapse(seq) -> list[int]:
    """Remove consecutive duplicates and missing (-1) entries."""
    out = []
    for a in seq:
        a = int(a)
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


def arc_lengths_m(nodes, arcs) -> np.ndarray:
    k = np.pi / 180.0
    lat0 = nodes.lat.mean()
    x = nodes.lon.values * k * 6_371_000 * np.cos(lat0 * k)
    y = nodes.lat.values * k * 6_371_000
    f, t = arcs.from_node.values, arcs.to_node.values
    return np.hypot(x[t] - x[f], y[t] - y[f])


def route_continuity(route: list[int], arcs) -> float | None:
    """Share of consecutive predicted arcs that are actually connected (to_node == next from_node)."""
    if len(route) < 2:
        return None
    to_n = arcs.to_node.values[route[:-1]]
    from_n = arcs.from_node.values[route[1:]]
    return float(np.mean(to_n == from_n))


def candidate_miss_rate(cands, obs_labels) -> float | None:
    """Share of labelled GPS points whose true arc is not in their candidate set."""
    labels = np.asarray(obs_labels)
    idx = np.nonzero(labels >= 0)[0]
    if not idx.size:
        return None
    have = set(zip(cands.obs_idx.values.astype(int).tolist(), cands.arc_id.values.astype(int).tolist()))
    return float(np.mean([(int(i), int(labels[i])) not in have for i in idx]))


def evaluate(pred_per_obs, ref_route, nodes, arcs, obs_labels=None) -> dict:
    pred_route = collapse(pred_per_obs)
    res = {
        "num_observations": int(len(pred_per_obs)),
        "num_predicted_arcs": len(pred_route),
        "route_continuity": route_continuity(pred_route, arcs),
    }
    if ref_route is None:
        return res

    lengths = arc_lengths_m(nodes, arcs)
    P, R = set(pred_route), set(ref_route)
    tp = P & R
    prec = len(tp) / len(P) if P else 0.0
    rec = len(tp) / len(R) if R else 0.0
    lp, lr, ltp = lengths[list(P)].sum(), lengths[list(R)].sum(), lengths[list(tp)].sum()
    res.update({
        "num_reference_arcs": len(collapse(ref_route)),
        "arc_precision": prec,
        "arc_recall": rec,
        "arc_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "length_precision": float(ltp / lp) if lp else 0.0,
        "length_recall": float(ltp / lr) if lr else 0.0,
        # share of GPS points whose predicted arc lies on the reference route
        "on_route_rate": float(np.mean([int(a) in R for a in pred_per_obs])),
    })
    if obs_labels is not None:
        labels = np.asarray(obs_labels)
        valid = labels >= 0
        res["top1_accuracy"] = float(np.mean(np.asarray(pred_per_obs)[valid] == labels[valid])) if valid.any() else None
    return res
