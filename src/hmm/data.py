"""Load P1's delivered tables for one trip (schema 1.0).

Files expected in data_dir: observations_{tag}.parquet, candidates_{tag}.parquet, arcs_{tag}.parquet,
nodes_{tag}.parquet, routes_{tag}.parquet, with tag in {"sample", "student", "train"}.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import TripGraph

TAGS = ("sample", "student", "train")


@dataclass
class TripData:
    trip_id: str
    obs: pd.DataFrame
    cand: pd.DataFrame
    arcs: pd.DataFrame
    nodes: pd.DataFrame
    route: np.ndarray | None
    graph: TripGraph


def _read(path: Path, trip_id: str) -> pd.DataFrame:
    return pd.read_parquet(path, filters=[("trip_id", "==", trip_id)])


def available_trips(data_dir: str | Path) -> dict[str, list[str]]:
    data_dir = Path(data_dir)
    out = {}
    for tag in TAGS:
        f = data_dir / f"observations_{tag}.parquet"
        if f.exists():
            out[tag] = sorted(pd.read_parquet(f, columns=["trip_id"]).trip_id.unique().tolist())
    return out


def load_trip_tables(trip_id: str, data_dir: str | Path, tag: str | None = None) -> TripData:
    data_dir = Path(data_dir)
    tags = [tag] if tag else list(TAGS)
    for t in tags:
        fo = data_dir / f"observations_{t}.parquet"
        if not fo.exists():
            continue
        obs = _read(fo, trip_id)
        if len(obs) == 0:
            continue
        cand = _read(data_dir / f"candidates_{t}.parquet", trip_id)
        arcs = _read(data_dir / f"arcs_{t}.parquet", trip_id).sort_values("arc_id").reset_index(drop=True)
        nodes = _read(data_dir / f"nodes_{t}.parquet", trip_id).sort_values("node_id").reset_index(drop=True)
        rt = _read(data_dir / f"routes_{t}.parquet", trip_id).sort_values("route_pos")
        route = rt.arc_id.to_numpy(np.int64) if len(rt) else None
        return TripData(trip_id, obs.sort_values("obs_idx").reset_index(drop=True),
                        cand.sort_values(["obs_idx", "cand_rank"]).reset_index(drop=True),
                        arcs, nodes, route, TripGraph(arcs, n_nodes=len(nodes)))
    raise FileNotFoundError(f"{trip_id} not found in {data_dir}")


def load_split(data_dir: str | Path, tag: str, trip_ids=None) -> list[TripData]:
    """Load every trip of a split at once (much faster than one filtered read per trip)."""
    data_dir = Path(data_dir)
    obs = pd.read_parquet(data_dir / f"observations_{tag}.parquet")
    cand = pd.read_parquet(data_dir / f"candidates_{tag}.parquet")
    arcs = pd.read_parquet(data_dir / f"arcs_{tag}.parquet")
    nodes = pd.read_parquet(data_dir / f"nodes_{tag}.parquet", columns=["trip_id", "node_id"])
    routes = pd.read_parquet(data_dir / f"routes_{tag}.parquet")
    ids = sorted(obs.trip_id.unique()) if trip_ids is None else list(trip_ids)
    g_obs, g_cand, g_arcs = dict(tuple(obs.groupby("trip_id"))), dict(tuple(cand.groupby("trip_id"))), dict(tuple(arcs.groupby("trip_id")))
    n_nodes = nodes.groupby("trip_id").size().to_dict()
    g_rt = dict(tuple(routes.groupby("trip_id")))
    out = []
    for tid in ids:
        o = g_obs[tid].sort_values("obs_idx").reset_index(drop=True)
        c = g_cand.get(tid, cand.iloc[:0]).sort_values(["obs_idx", "cand_rank"]).reset_index(drop=True)
        a = g_arcs[tid].sort_values("arc_id").reset_index(drop=True)
        rt = g_rt.get(tid)
        route = rt.sort_values("route_pos").arc_id.to_numpy(np.int64) if rt is not None and len(rt) else None
        out.append(TripData(tid, o, c, a, None, route, TripGraph(a, n_nodes=int(n_nodes[tid]))))
    return out
