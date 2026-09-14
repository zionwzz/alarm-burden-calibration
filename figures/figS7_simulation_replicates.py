#!/usr/bin/env python3
r"""Distribution of the replicate estimates for every estimator in every scenario.

Read from the saved replicate estimates (`simulation/results/*_REPLICATES.csv`) and the scenario
truths (`SIMULATION_LINKED_REWARD.json`); the replicate mean of each box is asserted against the
recorded bias before drawing. One panel per scenario, one row per estimator: the box spans the
interquartile range of estimate minus truth, the line inside it is the median, the whiskers reach
the 2.5th and 97.5th percentiles, and replicates beyond them are drawn individually. The panels
carry the manuscript's scenario letters (`alarmreplay/scenarios.py`).
"""
import csv
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402
from alarmreplay.scenarios import PAPER_LETTER as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "simulation", "results")
OUT = paths.FIGOUT

SCENARIOS = [("A", "Transport holds"), ("G", "Strong dependence"),
             ("D", "Cell heterogeneity"), ("B", "Composition shift"),
             ("C", f"As {L['B']}, 60 patients"), ("I", "Zeros by duration"),
             ("J", "Zeros by overshoot"), ("E", "Incomplete support"),
             ("F", "Transport fails")]
# rows from top to bottom; identity by position and label, hue as in the main figures
ESTIMATORS = [("replay", "Replay", st.REPLAY),
              ("lrc_d", "Duration-only", st.LRC_D),
              ("lrc_dz_pre", "Feature-augmented", st.LRC_DZ),
              ("lrc_dz", "In-sample map", "#E0A878"),
              ("baseline_oracle", "Baseline oracle", st.RECORDED),
              ("offset", "Fixed offset", st.LEVEL)]


def load():
    summary = json.load(open(os.path.join(RES, "SIMULATION_LINKED_REWARD.json")))
    reps = {}
    with open(os.path.join(RES, "SIMULATION_LINKED_REWARD_REPLICATES.csv")) as f:
        for r in csv.DictReader(f):
            reps.setdefault((r["scenario"], r["estimator"]), []).append(float(r["estimate"]))
    return summary, reps


def errors(summary, reps, sc, key):
    e = np.array(reps[(sc, key)]) - summary[sc]["_truth"]
    e = e[np.isfinite(e)]
    if abs(e.mean() - summary[sc][key]["bias"]) > 1e-4:
        raise SystemExit(f"{sc}/{key}: replicate mean differs from the recorded bias")
    return e


def box(ax, y, e, colour):
    q1, med, q3 = np.percentile(e, [25, 50, 75])
    lo, hi = np.percentile(e, [2.5, 97.5])
    h = 0.30
    ax.plot([lo, q1], [y, y], color=colour, lw=0.9, zorder=2)
    ax.plot([q3, hi], [y, y], color=colour, lw=0.9, zorder=2)
    ax.add_patch(Rectangle((q1, y - h), q3 - q1, 2 * h, facecolor="white", edgecolor=colour,
                           lw=0.9, zorder=3))
    ax.plot([med, med], [y - h, y + h], color=colour, lw=1.3, zorder=4)
    out = e[(e < lo) | (e > hi)]
    ax.plot(out, np.full(out.shape, y), marker="o", ls="none", ms=1.6, color=colour, alpha=0.6,
            zorder=2)


def panel(ax, summary, reps, sc, show_labels):
    ys = np.arange(len(ESTIMATORS))[::-1]
    ax.axvline(0.0, color=st.MUTED, lw=0.8, zorder=1)
    span = 0.0
    for y, (key, _, colour) in zip(ys, ESTIMATORS):
        e = errors(summary, reps, sc, key)
        box(ax, y, e, colour)
        span = max(span, np.abs(e).max())
    lim = np.ceil(span / 5.0) * 5.0
    ax.set_xlim(-lim, lim)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in ESTIMATORS] if show_labels else [], fontsize=7.0)
    ax.set_ylim(-1.5, len(ESTIMATORS) - 0.3)
    n = summary[sc]["_diagnostics"]["n_patients"]
    ax.text(0.98, 0.02, f"{n} patients", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color=st.MUTED)
    st.tidy(ax, grid_axis="x")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    summary, reps = load()
    fig = figpage.page_figure(plt, 5.6)
    gs = fig.add_gridspec(3, 3, left=0.205, right=0.985, top=0.955, bottom=0.060,
                          wspace=0.15, hspace=0.62)
    for k, (sc, title) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        panel(ax, summary, reps, sc, show_labels=(k % 3 == 0))
        ax.set_title(f"{L[sc]}   {title}", pad=4, fontsize=7.6)
        if k // 3 == 2:
            ax.set_xlabel("Estimate $-$ truth (points)", labelpad=1)
    figpage.save_fixed(fig, "FigureS7_simulation_replicates", OUT, plt)


if __name__ == "__main__":
    main()
