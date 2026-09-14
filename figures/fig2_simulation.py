#!/usr/bin/env python3
r"""Render the simulation summary from the saved simulation results.

Every mark is read from `simulation/results/`: the scenario summaries
(`SIMULATION_LINKED_REWARD.json`), the saved replicate estimates
(`SIMULATION_LINKED_REWARD_REPLICATES.csv`), the patient-count sweep
(`SIMULATION_SAMPLE_SIZE.json`) and the zero-inflated patient-count checks
(`ZERO_CHECK_*.json`). Nothing is simulated here; the replicate mean recomputed from the saved
estimates is asserted against the recorded bias before anything is drawn.

Panel A: estimate minus truth for the three primary estimators in every scenario, the mark at the
replicate mean (the bias) and the line spanning the central 95% of the replicate errors.
Panel B: coverage of the 95% patient-cluster bootstrap intervals in the same rows.
Panels C and D: the feature-augmented estimator as the patient count grows, in scenarios A and J:
coverage, and the empirical standard deviation beside the mean bootstrap standard error.

Scenario letters in this file are the simulation's keys; the figure prints the manuscript's
letters through `alarmreplay/scenarios.py`.
"""
import csv
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402
from alarmreplay.scenarios import PAPER_LETTER as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "simulation", "results")
OUT = paths.FIGOUT

# Rows from top to bottom, grouped by what the scenario stresses: the transport condition holds
# (A, G, D); the overshoot composition shifts (B, C); observed zeros at every duration (I, J); an
# identification condition deliberately fails (E, F). Labels are short enough for a 7 pt tick.
# simulation keys with the manuscript's letters (alarmreplay/scenarios.py) in front of the labels
ROWS = [(k, f"{L[k]}  {name}") for k, name in [
    ("A", "Transport holds"),
    ("G", "Stronger dependence"),
    ("D", "Pooled-cell heterogeneity"),
    ("B", "Composition shift"),
    ("C", f"As {L['B']}, 60 patients"),
    ("I", "Zeros depend on duration"),
    ("J", "Zeros depend on overshoot"),
    ("E", "Incomplete support"),
    ("F", "Reward transport fails"),
]]
GROUP_BREAKS = [3, 5, 7]          # a rule after these row indices (0-based, top to bottom)
VIOLATED = ("E", "F")             # identification deliberately violated: band and note

METHODS = [("replay", st.REPLAY, "o"), ("lrc_d", st.LRC_D, "s"), ("lrc_dz_pre", st.LRC_DZ, "D")]
ROW_OFFSET = {"replay": 0.0, "lrc_d": +0.24, "lrc_dz_pre": -0.24}
CI_STYLE = {"replay": ("-", 1.1), "lrc_d": ("-", 1.3), "lrc_dz_pre": ((0, (2.4, 1.3)), 1.3)}
NOMINAL = 0.95
N_COVERAGE = 400                  # replicates used for coverage everywhere in the study


def load():
    summary = json.load(open(os.path.join(RES, "SIMULATION_LINKED_REWARD.json")))
    reps = {}
    with open(os.path.join(RES, "SIMULATION_LINKED_REWARD_REPLICATES.csv")) as f:
        for r in csv.DictReader(f):
            reps.setdefault((r["scenario"], r["estimator"]), []).append(float(r["estimate"]))
    sweep = json.load(open(os.path.join(RES, "SIMULATION_SAMPLE_SIZE.json")))
    zero = {}
    for n in (600, 1200):
        p = os.path.join(RES, f"ZERO_CHECK_{n}.json")
        if os.path.exists(p):
            zero[n] = json.load(open(p))
    return summary, reps, sweep, zero


def errors(summary, reps, sc, key):
    """Replicate errors, checked against the recorded bias to the precision the tables print."""
    truth = summary[sc]["_truth"]
    e = np.array(reps[(sc, key)]) - truth
    e = e[np.isfinite(e)]
    recorded = summary[sc][key]["bias"]
    if abs(e.mean() - recorded) > 1e-4:
        raise SystemExit(f"{sc}/{key}: replicate mean {e.mean():.5f} differs from the recorded "
                         f"bias {recorded:.5f}")
    return e


def band(ax, n_reps, orientation="v"):
    """Nominal coverage with two binomial Monte Carlo standard errors either side."""
    half = 2.0 * np.sqrt(NOMINAL * (1.0 - NOMINAL) / n_reps)
    if orientation == "v":
        ax.axvspan(NOMINAL - half, NOMINAL + half, color=st.BAND, lw=0, zorder=0)
        ax.axvline(NOMINAL, color=st.MUTED, lw=0.8, zorder=1)
    else:
        ax.axhspan(NOMINAL - half, NOMINAL + half, color=st.BAND, lw=0, zorder=0)
        ax.axhline(NOMINAL, color=st.MUTED, lw=0.8, zorder=1)


def _rows_frame(ax, ys, show_labels):
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in ROWS] if show_labels else [], fontsize=7.0)
    if show_labels:
        for lab, (sc, _) in zip(ax.get_yticklabels(), ROWS):
            lab.set_color("#7A7F85" if sc in VIOLATED else st.INK)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    for k in GROUP_BREAKS:
        ax.axhline(ys[k - 1] - 0.5, color="#D5D8DB", lw=0.6, zorder=0)
    st.tidy(ax, grid_axis="x")


def panel_bias(ax, summary, reps):
    ys = np.arange(len(ROWS))[::-1]
    # the two rows whose identification conditions are deliberately broken
    lo = ys[[sc for sc, _ in ROWS].index("F")] - 0.5
    hi = ys[[sc for sc, _ in ROWS].index("E")] + 0.5
    ax.add_patch(Rectangle((-100, lo), 200, hi - lo, facecolor="#F3F4F5", edgecolor="none",
                           zorder=0))
    ax.axvline(0.0, color=st.MUTED, lw=0.8, zorder=1)
    for y, (sc, _) in zip(ys, ROWS):
        for key, colour, marker in METHODS:
            e = errors(summary, reps, sc, key)
            q = np.percentile(e, [2.5, 97.5])
            yy = y + ROW_OFFSET[key]
            ls, lw = CI_STYLE[key]
            ax.plot(q, [yy, yy], color=colour, lw=lw, ls=ls, solid_capstyle="butt",
                    dash_capstyle="butt", zorder=3)
            ax.plot([e.mean()], [yy], marker=marker, color=colour, ms=3.9,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    _rows_frame(ax, ys, show_labels=True)
    ax.set_xlim(-26, 22)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Estimate $-$ truth (percentage points)", labelpad=1)


def panel_coverage(ax, summary):
    ys = np.arange(len(ROWS))[::-1]
    band(ax, N_COVERAGE, "v")
    for y, (sc, _) in zip(ys, ROWS):
        for key, colour, marker in METHODS[1:]:
            c = summary[sc][key]["coverage"]
            ax.plot([c], [y + ROW_OFFSET[key]], marker=marker, color=colour, ms=3.9,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    _rows_frame(ax, ys, show_labels=False)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 0.95])
    ax.set_xticklabels(["0", "0.5", "0.95"])
    ax.set_xlabel("Coverage", labelpad=1)
    # the note for the two violated rows sits in the empty right half of their rows
    y_note = 0.5 * (ys[-1] + ys[-2])
    ax.text(0.86, y_note, "conditions\ndeliberately\nviolated", ha="right", va="center",
            fontsize=7.0, color=st.MUTED, style="italic", linespacing=1.05)


def sweep_series(summary, sweep, zero):
    """(patients, coverage, empirical SD, mean bootstrap SE) for the feature-augmented estimator."""
    a = [(r["n_patients"], r["lrc_dz_pre"]["coverage"], r["lrc_dz_pre"]["esd"],
          r["lrc_dz_pre"]["mean_se"]) for r in sweep["sizes"]]
    j300 = summary["J"]["lrc_dz_pre"]
    j = [(300, j300["coverage"], j300["esd"], j300["mean_se"])]
    for n, d in sorted(zero.items()):
        v = d["lrc_dz_pre"]
        j.append((n, v["coverage"], v["esd"], v["mean_se"]))
    return sorted(a), sorted(j)


def panel_sweep_coverage(ax, a, j):
    band(ax, N_COVERAGE, "h")
    for series, mfc, ls, name in [(a, st.LRC_DZ, "-", L["A"]), (j, "white", (0, (2.4, 1.3)), L["J"])]:
        n = [r[0] for r in series]
        c = [r[1] for r in series]
        ax.plot(n, c, color=st.LRC_DZ, lw=1.2, ls=ls, zorder=3)
        ax.plot(n, c, marker="D", ls="none", color=st.LRC_DZ, ms=3.9, markerfacecolor=mfc,
                markeredgewidth=0.9, zorder=4)
        ax.annotate(name, (n[-1], c[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.0, color=st.INK, va="center", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([300, 600, 1200, 2400])
    ax.set_xticklabels(["300", "600", "1200", "2400"])
    ax.minorticks_off()
    ax.set_xlim(240, 3300)
    ax.set_ylim(0.88, 1.0)
    ax.set_yticks([0.90, 0.95, 1.00])
    ax.set_xlabel("Patients", labelpad=1)
    ax.set_ylabel("Coverage", labelpad=2)
    st.tidy(ax, grid_axis="y")


def panel_sweep_sd(ax, a, j):
    n0, _, sd0, _ = a[0]
    grid = np.array([240.0, 3300.0])
    ax.plot(grid, sd0 * np.sqrt(n0 / grid), color="#C4C8CC", lw=0.9, ls=":", zorder=1)
    for series, mfc, ls, name in [(a, st.LRC_DZ, "-", L["A"]), (j, "white", (0, (2.4, 1.3)), L["J"])]:
        n = [r[0] for r in series]
        sd = [r[2] for r in series]
        se = [r[3] for r in series]
        ax.plot(n, sd, color=st.LRC_DZ, lw=1.2, ls=ls, zorder=3)
        ax.plot(n, sd, marker="D", ls="none", color=st.LRC_DZ, ms=3.9, markerfacecolor=mfc,
                markeredgewidth=0.9, zorder=4)
        ax.plot(n, se, marker="_", ls="none", color=st.RECORDED, ms=7.0, markeredgewidth=1.1,
                zorder=5)
        ax.annotate(name, (n[-1], sd[-1]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7.0, color=st.INK, va="center", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([300, 600, 1200, 2400])
    ax.set_xticklabels(["300", "600", "1200", "2400"])
    ax.minorticks_off()
    ax.set_xlim(240, 3300)
    ax.set_ylim(0.0, 1.7)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Patients", labelpad=1)
    ax.set_ylabel("Points", labelpad=2)
    st.tidy(ax, grid_axis="y")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    summary, reps, sweep, zero = load()
    a, j = sweep_series(summary, sweep, zero)

    fig = figpage.page_figure(plt, 4.85)
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], left=0.310, right=0.975,
                           top=0.955, bottom=0.500, wspace=0.10)
    ax_a = fig.add_subplot(top[0, 0])
    panel_bias(ax_a, summary, reps)
    st.panel(ax_a, "A", "Error of the estimated reduction")
    ax_b = fig.add_subplot(top[0, 1])
    panel_coverage(ax_b, summary)
    st.panel(ax_b, "B", "Coverage")

    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.310, right=0.975,
                              top=0.345, bottom=0.200, wspace=0.40)
    ax_c = fig.add_subplot(bottom[0, 0])
    panel_sweep_coverage(ax_c, a, j)
    st.panel(ax_c, "C", "Coverage by patient count")
    ax_d = fig.add_subplot(bottom[0, 1])
    panel_sweep_sd(ax_d, a, j)
    st.panel(ax_d, "D", "SD and bootstrap SE")

    handles = [
        Line2D([0], [0], color=st.REPLAY, marker="o", ms=3.9, ls="-", lw=1.1,
               markeredgecolor="white", markeredgewidth=0.5, label="Replay, uncorrected"),
        Line2D([0], [0], color=st.LRC_D, marker="s", ms=3.9, ls="-", lw=1.3,
               markeredgecolor="white", markeredgewidth=0.5, label="Calibrated, duration transport"),
        Line2D([0], [0], color=st.LRC_DZ, marker="D", ms=3.9, ls=CI_STYLE["lrc_dz_pre"][0], lw=1.3,
               markeredgecolor="white", markeredgewidth=0.5, label="Calibrated, feature-augmented"),
        Patch(facecolor=st.BAND, edgecolor="none", label="0.95 $\\pm$ 2 Monte Carlo SE"),
        Line2D([0], [0], color=st.RECORDED, marker="_", ms=7.0, ls="none", markeredgewidth=1.1,
               label="Mean bootstrap SE"),
        Line2D([0], [0], color="#C4C8CC", ls=":", lw=0.9, label="Proportional to 1/\u221an"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.60, 0.000),
               fontsize=7.0, handlelength=2.2, columnspacing=1.6, labelspacing=0.45)
    figpage.save_fixed(fig, "Figure2_simulation", OUT, plt)


if __name__ == "__main__":
    main()
