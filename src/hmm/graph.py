"""Per-trip directed road graph + bounded, cached shortest paths.

Arc and node IDs are PER TRIP (schema 1.0), so one TripGraph is built per trip.
Edge weight = arc length in metres. Duplicate (from, to) arcs keep the shortest length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class TripGraph:
    """Directed graph of one trip.

    arcs: DataFrame with arc_id, from_node, to_node, length_m (arcs_{split}.parquet rows of ONE trip).
    n_nodes: number of nodes in the trip (defaults to max node id + 1).
    """

    def __init__(self, arcs: pd.DataFrame, n_nodes: int | None = None):
        a = arcs.sort_values("arc_id").reset_index(drop=True)
        if not np.array_equal(a.arc_id.to_numpy(), np.arange(len(a))):
            raise ValueError("arc_id must be 0..n_arcs-1 (row index of .arcs)")
        self.arcs = a
        self.from_node = a.from_node.to_numpy(np.int64)
        self.to_node = a.to_node.to_numpy(np.int64)
        self.length = a.length_m.to_numpy(np.float64)
        self.n_nodes = int(n_nodes if n_nodes is not None else max(self.from_node.max(), self.to_node.max()) + 1)

        # csr_matrix SUMS duplicate entries, so collapse duplicates to their min length first.
        e = pd.DataFrame({"f": self.from_node, "t": self.to_node, "w": self.length})
        e = e[e.f != e.t].groupby(["f", "t"], as_index=False).w.min()
        w = np.maximum(e.w.to_numpy(), 1e-3)            # 0 would be dropped as "no edge" by csgraph
        self.csr = csr_matrix((w, (e.f.to_numpy(), e.t.to_numpy())), shape=(self.n_nodes, self.n_nodes))
        # (from, to) -> arc_id, used to turn node paths back into arc paths (shortest duplicate wins)
        best = (pd.DataFrame({"f": self.from_node, "t": self.to_node, "w": self.length, "id": np.arange(len(a))})
                .sort_values(["w", "id"]).drop_duplicates(["f", "t"]))
        self._edge_arc = {(int(f), int(t)): int(i) for f, t, i in zip(best.f, best.t, best.id)}
        self._cache: dict[int, tuple[float, np.ndarray]] = {}
        self.n_dijkstra_calls = 0

    # ------------------------------------------------------------------
    def dist_from(self, sources: np.ndarray, limit: float, targets: np.ndarray) -> np.ndarray:
        """Road distance (m) from each source node to each target node, searched up to `limit` metres.
        Returns (len(sources), len(targets)); inf where not reachable within the limit.
        The cache keeps only the nodes reached (sparse), per source, and is reused whenever the cached
        limit is large enough. Memory stays small even on 100k-node graphs."""
        sources = np.asarray(sources, dtype=np.int64); targets = np.asarray(targets, dtype=np.int64)
        todo = [int(s) for s in np.unique(sources) if s not in self._cache or self._cache[s][0] < limit]
        if todo:
            self.n_dijkstra_calls += 1
            D = dijkstra(self.csr, directed=True, indices=np.asarray(todo), limit=limit)
            for s, row in zip(todo, np.atleast_2d(D)):
                idx = np.flatnonzero(np.isfinite(row))
                self._cache[s] = (limit, idx, row[idx].astype(np.float32))
        out = np.full((len(sources), len(targets)), np.inf)
        for i, s in enumerate(sources):
            _, idx, d = self._cache[int(s)]
            if len(idx) == 0:
                continue
            p = np.clip(np.searchsorted(idx, targets), 0, len(idx) - 1)
            hit = idx[p] == targets
            out[i, hit] = d[p[hit]]
        return np.where(out <= limit, out, np.inf)

    def clear_cache(self):
        self._cache.clear()

    def node_path(self, src: int, dst: int, limit: float = np.inf) -> list[int] | None:
        """Shortest node path src -> dst, or None."""
        if src == dst:
            return [src]
        _, pred = dijkstra(self.csr, directed=True, indices=src, limit=limit, return_predecessors=True)
        if pred[dst] < 0:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(int(pred[path[-1]]))
        return path[::-1]

    def arc_path(self, arc_a: int, arc_b: int, limit: float = 5000.0) -> list[int] | None:
        """Arcs strictly between arc_a and arc_b on the shortest road path (empty list if directly connected)."""
        p = self.node_path(int(self.to_node[arc_a]), int(self.from_node[arc_b]), limit)
        if p is None:
            return None
        return [self._edge_arc[(u, v)] for u, v in zip(p[:-1], p[1:])]

    def connected(self, a: int, b: int) -> bool:
        return a == b or self.to_node[a] == self.from_node[b]
