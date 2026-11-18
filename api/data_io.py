"""Reading raw trip files.

Layout: <eval_dir>/<trip_id>/<trip_id>.track  (lon lat t, whitespace, no header)
                             <trip_id>.nodes  (lon lat, row index = node_id)
                             <trip_id>.arcs   (from_node to_node, row index = arc_id)
                             <trip_id>.route  (arc_id per line, ground truth - SERVER SIDE ONLY)
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame   # columns: lon, lat, t
    nodes: pd.DataFrame   # columns: lon, lat
    arcs: pd.DataFrame    # columns: from_node, to_node


def _read(path: Path, names, dtypes) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes, engine="c")


def list_trips(eval_dir: Path) -> list[str]:
    if not eval_dir.is_dir():
        return []
    return sorted(p.name for p in eval_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def load_trip(eval_dir: Path, trip_id: str) -> Trip:
    d = eval_dir / trip_id
    if not d.is_dir():
        raise FileNotFoundError(trip_id)
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], {"lon": float, "lat": float, "t": float})
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], {"lon": float, "lat": float})
    arcs = _read(d / f"{trip_id}.arcs", ["from_node", "to_node"], {"from_node": int, "to_node": int})
    return Trip(trip_id, track, nodes, arcs)


def load_reference_route(eval_dir: Path, trip_id: str) -> list[int] | None:
    """Ground-truth arc sequence. Used ONLY by metrics.py; never passed to the pipeline or the UI."""
    p = eval_dir / trip_id / f"{trip_id}.route"
    if not p.exists():
        return None
    return _read(p, ["arc_id"], {"arc_id": int})["arc_id"].tolist()
