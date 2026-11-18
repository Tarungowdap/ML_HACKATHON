"""Arcs -> lat/long lines. Used by the maps here and by P4's API to send routes to the UI.

Output uses [lat, lon] order (Leaflet / Folium convention). Input tables use lon/lat (schema 1.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def arc_lines(arc_ids, arcs: pd.DataFrame) -> list[list[list[float]]]:
    """One 2-point line [[lat, lon], [lat, lon]] per arc id (from_node -> to_node). -1 is skipped.
    arcs: arcs_{split}.parquet rows of one trip (needs arc_id, from_lat, from_lon, to_lat, to_lon)."""
    a = arcs.set_index("arc_id")
    out = []
    for i in arc_ids:
        i = int(i)
        if i < 0:
            continue
        r = a.loc[i]
        out.append([[float(r.from_lat), float(r.from_lon)], [float(r.to_lat), float(r.to_lon)]])
    return out


def arcs_to_polylines(arc_ids, arcs: pd.DataFrame) -> list[list[list[float]]]:
    """Merge a sequence of arcs into as few polylines as possible: consecutive connected arcs
    (to_node == next from_node) are joined; a new polyline starts wherever the sequence breaks.
    Consecutive repeats are collapsed first. Returns [[ [lat, lon], ... ], ...]."""
    a = arcs.set_index("arc_id")
    seq = []
    for i in arc_ids:
        i = int(i)
        if i >= 0 and (not seq or seq[-1] != i):
            seq.append(i)
    lines: list[list[list[float]]] = []
    prev_to = None
    for i in seq:
        r = a.loc[i]
        p0 = [float(r.from_lat), float(r.from_lon)]; p1 = [float(r.to_lat), float(r.to_lon)]
        if lines and prev_to == int(r.from_node):
            lines[-1].append(p1)
        else:
            lines.append([p0, p1])
        prev_to = int(r.to_node)
    return lines


def route_geojson(arc_ids, arcs: pd.DataFrame, properties: dict | None = None) -> dict:
    """GeoJSON MultiLineString (coordinates in [lon, lat], as GeoJSON requires) for the API."""
    lines = arcs_to_polylines(arc_ids, arcs)
    return {"type": "Feature", "properties": properties or {},
            "geometry": {"type": "MultiLineString",
                         "coordinates": [[[lon, lat] for lat, lon in ln] for ln in lines]}}
