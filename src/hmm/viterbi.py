"""Log-space Viterbi with automatic segment restarts.

delta_1(j) = logE_1(j)
delta_t(j) = max_i [ delta_{t-1}(i) + logT_t(i, j) ] + logE_t(j)     (back-pointer = argmax i)

A new segment starts at t when `restart(t)` is True (e.g. a GPS gap with split_on_gap) or when no
candidate at t is reachable from any candidate at t-1 (the path is broken). Each segment is
back-tracked independently, so one bad step never drags the rest of the trip with it.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def viterbi(log_e: Sequence[np.ndarray],
            log_t: Callable[[int], np.ndarray],
            broken: Callable[[np.ndarray], bool],
            restart: Callable[[int], bool] = lambda t: False):
    """log_e[t]: (k_t,) emission scores. log_t(t): (k_{t-1}, k_t) transition scores for t >= 1.
    broken(T): True if the transition matrix has no valid move.
    Returns (state index per step, segment id per step, list of break steps)."""
    n = len(log_e)
    states = np.zeros(n, dtype=np.int64)
    seg = np.zeros(n, dtype=np.int64)
    breaks: list[int] = []
    if n == 0:
        return states, seg, breaks

    back: list[np.ndarray | None] = [None] * n
    delta = np.asarray(log_e[0], dtype=np.float64)
    start = 0
    sid = 0

    def backtrack(end: int, last_delta: np.ndarray):
        j = int(np.argmax(last_delta))
        for t in range(end, start - 1, -1):
            states[t] = j
            seg[t] = sid
            if t > start:
                j = int(back[t][j])

    for t in range(1, n):
        T = None if restart(t) else log_t(t)
        if T is None or broken(T):
            backtrack(t - 1, delta)
            breaks.append(t)
            sid += 1
            start = t
            delta = np.asarray(log_e[t], dtype=np.float64)
            continue
        scores = delta[:, None] + T                     # (k_prev, k_cur)
        back[t] = np.argmax(scores, axis=0)
        delta = scores[back[t], np.arange(T.shape[1])] + log_e[t]
    backtrack(n - 1, delta)
    return states, seg, breaks
