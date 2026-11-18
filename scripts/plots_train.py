"""Sensitivity plots for report §6 from results/tuning_train.csv (78 train trips).

results/train_sens_sigma_beta.png   stage 1: accuracy over sigma x beta
results/train_sens_uturn_heading.png stage 2: accuracy over U-turn penalty x heading weight
results/train_sens_uturns.png        U-turn count vs U-turn penalty (heading off / on)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def heat(piv, xlabel, ylabel, title, note, path, mark=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    v = piv.values * 100
    im = ax.imshow(v, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{i:g}" for i in piv.index])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    mid = np.nanmean(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=9, color="white" if v[i, j] > mid else INK)
    if mark is not None:
        ax.add_patch(plt.Rectangle((mark[1] - .5, mark[0] - .5), 1, 1, fill=False, ec=ORANGE, lw=2))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("HMM pointwise accuracy (%)", color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, note, color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path); plt.close(fig)


def main():
    d = pd.read_csv(ROOT / "results" / "hmm" / "tuning_train.csv")
    cfg = json.loads((ROOT / "hmm_config.json").read_text())
    note = f"78 train trips (00000033, 00000054 excluded); Gaussian emission. Selected: σ={cfg['sigma_m']:g}, " \
           f"β={cfg['beta_m']:g}, U-turn {cfg['uturn_logp']:g}, heading κ={cfg['heading_kappa']:g}"
    out = ROOT / "results" / "hmm"

    s1 = d[d.stage == "1_sigma_beta"]
    piv = s1.pivot(index="sigma_m", columns="beta_m", values="acc_hmm").sort_index(ascending=False)
    heat(piv, "β (transition tolerance, m)", "σ (GPS noise, m)",
         "Stage 1: accuracy vs σ and β (U-turn −3, heading κ 2)", note, out / "train_sens_sigma_beta.png")

    s2 = d[d.stage == "2_uturn_heading"]
    piv = s2.pivot(index="uturn_logp", columns="heading_kappa", values="acc_hmm").sort_index(ascending=False)
    heat(piv, "heading weight κ", "U-turn penalty (log)", "Stage 2: accuracy vs U-turn penalty and heading (σ 6, β 2)",
         note, out / "train_sens_uturn_heading.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    for k, col, lab in ((0, BLUE, "heading off (κ 0)"), (2, ORANGE, "heading on (κ 2)")):
        s = s2[s2.heading_kappa == k].sort_values("uturn_logp")
        ax.plot(-s.uturn_logp, s.uturns, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
        ax.text(-s.uturn_logp.iloc[0] + 0.2, s.uturns.iloc[0], lab, color=INK, fontsize=9, va="center")
    ax.set_yscale("log"); ax.set_xticks([0, 3, 6, 10], ["0", "−3", "−6", "−10"])
    ax.set_xlabel("U-turn penalty (log)"); ax.set_ylabel("U-turns in decoded routes (78 trips)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Spurious U-turns fall 20× with the penalty", loc="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, "σ 6 m, β 2 m; reference routes contain 0 U-turns", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "train_sens_uturns.png"); plt.close(fig)
    print("saved train_sens_*.png")


if __name__ == "__main__":
    main()
