from pathlib import Path
import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111132.95

def load_trip(trip_dir: Path) -> dict:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = pd.read_csv(trip_dir / f"{tid}.track", header=None, sep=r"\s+", names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(trip_dir / f"{tid}.nodes", header=None, sep=r"\s+", names=["longitude", "latitude"])
    arcs = pd.read_csv(trip_dir / f"{tid}.arcs", header=None, sep=r"\s+", names=["from_node_id", "to_node_id"])
    arcs["arc_id"] = np.arange(len(arcs), dtype=np.int32)
    route_file = trip_dir / f"{tid}.route"
    route = None
    if route_file.exists():
        route = pd.read_csv(route_file, header=None, sep=r"\s+", names=["arc_id"])["arc_id"].to_numpy(dtype=np.int32)
    return {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs, "route": route}

def latlon_to_local_xy(lon: np.ndarray, lat: np.ndarray, lat0: float, lon0: float):
    m_per_deg_lon = 111319.49 * np.cos(np.radians(lat0))
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * M_PER_DEG_LAT
    return x, y
