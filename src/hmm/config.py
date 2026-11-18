"""HMM configuration. Saved/loaded as hmm_config.json (the file the API reads at demo time)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class HMMConfig:
    # ---- emission ----
    emission: str = "gaussian"      # "gaussian" | "pmatch" | "blend"
    sigma_m: float = 7.0            # GPS noise for the Gaussian emission (m)
    w_pmatch: float = 1.0           # blend weight: w*log(p_match) + (1-w)*log Gaussian(d)
    heading_kappa: float = 0.0      # optional heading term: kappa*(cos(bearing_diff) - 1); 0 = off
    heading_min_speed: float = 2.0  # only use heading when the point moved this fast (m/s)
    pmatch_floor: float = 1e-6      # p_match clipped to this before log
    # ---- transition (Newson–Krumm) ----
    beta_m: float = 5.0             # tolerance on |road - straight| (m) for dt = 1 s
    beta_dt_scale: bool = True      # beta_eff = beta * max(dt, 1)
    same_arc_back_tol_m: float = 15.0   # small backward moves on the same arc count as GPS noise
    search_base_m: float = 200.0    # shortest-path search cap = base + vmax * dt
    vmax_mps: float = 60.0
    search_max_m: float = 5000.0
    uturn_logp: float = -5.0        # extra log-penalty for switching to the reverse twin of the same road
    log_floor: float = -40.0       # worst log-transition for a reachable move
    log_unreachable: float = -60.0  # no path within the search cap
    # ---- gaps / segments ----
    gap_s: float = 30.0             # dt above this is a GPS gap
    split_on_gap: bool = False      # True: restart Viterbi after a gap; False: just loosen beta/search
    fill_missing: str = "prev"      # points with 0 candidates: "prev" arc, "next" arc, or "none" (-1)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        d = asdict(self)
        if extra:
            d["_meta"] = extra
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HMMConfig":
        d = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
