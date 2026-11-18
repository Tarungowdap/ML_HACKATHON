"""Folium maps for one trip: raw GPS, ground truth, classifier-only (top-1) and HMM routes as toggleable layers."""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd

from .coords import arcs_to_polylines

COLORS = {"gps": "#555555", "truth": "#2b8a3e", "top1": "#e03131", "hmm": "#1c7ed6", "wrong": "#f08c00"}


def _lines(layer, lines, color, weight, opacity=0.9, dash=None, tooltip=None):
    for ln in lines:
        folium.PolyLine(ln, color=color, weight=weight, opacity=opacity, dash_array=dash, tooltip=tooltip).add_to(layer)


def trip_map(obs: pd.DataFrame, arcs: pd.DataFrame, route=None, top1=None, hmm=None,
             title: str = "", focus: tuple[int, int] | None = None, point_step: int = 1) -> folium.Map:
    """obs: observations of one trip (lon, lat, obs_idx, true_arc_id optional).
    route: reference arc sequence; top1 / hmm: per-point predicted arcs (same order as obs).
    focus: (first_obs, last_obs) to zoom on a window (e.g. a failure case)."""
    o = obs.sort_values("obs_idx").reset_index(drop=True)
    sel = o if focus is None else o[(o.obs_idx >= focus[0]) & (o.obs_idx <= focus[1])]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[sel.lat.min(), sel.lon.min()], [sel.lat.max(), sel.lon.max()]])
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:6px 10px;'
            f'border-radius:4px;font:14px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">{title}</div>'))

    if route is not None and len(route):
        fg = folium.FeatureGroup("Ground truth (.route)")
        _lines(fg, arcs_to_polylines(route, arcs), COLORS["truth"], 9, 0.45, tooltip="ground truth")
        fg.add_to(m)
    if top1 is not None:
        fg = folium.FeatureGroup("Classifier-only (top-1 per point)", show=True)
        _lines(fg, arcs_to_polylines(top1, arcs), COLORS["top1"], 2, 0.8, tooltip="top-1")
        fg.add_to(m)
    if hmm is not None:
        fg = folium.FeatureGroup("HMM route")
        _lines(fg, arcs_to_polylines(hmm, arcs), COLORS["hmm"], 4, 0.9, tooltip="HMM")
        fg.add_to(m)

    fg = folium.FeatureGroup("Raw GPS points")
    has_truth = "true_arc_id" in o and hmm is not None
    wrong = (np.asarray(hmm) != o.true_arc_id.to_numpy()) & (o.true_arc_id.to_numpy() >= 0) if has_truth else np.zeros(len(o), bool)
    for i in range(0, len(o), point_step):
        r = o.iloc[i]
        tip = f"obs {int(r.obs_idx)} | t={r.timestamp:.0f}"
        if hmm is not None:
            tip += f" | hmm={int(hmm[i])}"
        if top1 is not None:
            tip += f" | top1={int(top1[i])}"
        if has_truth:
            tip += f" | true={int(r.true_arc_id)}"
        folium.CircleMarker([r.lat, r.lon], radius=2.5 if not wrong[i] else 4,
                            color=COLORS["wrong"] if wrong[i] else COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=1, tooltip=tip).add_to(fg)
    fg.add_to(m)
    if has_truth:
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;bottom:20px;left:10px;z-index:9999;background:white;padding:6px 10px;'
            'border-radius:4px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<span style="color:{COLORS["truth"]}">━</span> ground truth &nbsp; '
            f'<span style="color:{COLORS["top1"]}">━</span> classifier-only &nbsp; '
            f'<span style="color:{COLORS["hmm"]}">━</span> HMM &nbsp; '
            f'<span style="color:{COLORS["wrong"]}">●</span> HMM wrong point</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save(m: folium.Map, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path
