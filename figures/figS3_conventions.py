#!/usr/bin/env python3
r"""The measurement-convention scan, read from the scan record.

`sensitivity/S2_CONVENTION_SCAN.csv` in the working tree holds, for every channel and every
convention varied one at a time from the fixed defaults, the held-out burden ratio with its
patient-cluster bootstrap interval for the naive and the selected cell, and the episode recall
and onset agreement at the selected cell. The upper row draws the burden ratios under the
recorded-side conventions (part A of the scan), the lower row the recall and onset agreement
under the matching conventions (part B), which leave the burden ratio unchanged by construction.
"""
import csv
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402

SCAN = os.path.join(paths.WORK, "sensitivity", "S2_CONVENTION_SCAN.csv")
OUT = paths.FIGOUT

CHANNELS = [("ecgResp-numLimit-respRate-low", "RR low"),
            ("spO2-numLimit-satO2-low", "SpO₂ low"),
            ("ecg-numLimit-heartRate-high", "HR high"),
            ("ecg-numLimit-heartRate-low", "HR low")]
PART_A = [("default", "Fixed defaults"), ("d0_8", "Merge gap 8 s"), ("d0_16", "Merge gap 16 s"),
          ("d0_60", "Merge gap 60 s"), ("sil_include", "Silencing retained"),
          ("lim_modal", "Cohort modal limits")]
PART_B = [("default", "Fixed defaults"), ("tol_10", "Tolerance ±10 s"), ("tol_20", "Tolerance ±20 s"),
          ("tol_60", "Tolerance ±60 s"), ("onset_6", "Onset window 6 s"),
          ("onset_20", "Onset window 20 s")]
BAND = (0.80, 1.25)
RECALL_CRITERION, ONSET_CRITERION = 0.90, 0.80


def load():
    rows = {}
    for r in csv.DictReader(open(SCAN)):
        rows[(r["channel"], r["config"], r["cell"])] = r
    return rows


def panel_a(ax, rows, ch, show_labels):
    ys = np.arange(len(PART_A))[::-1]
    ax.axvspan(BAND[0], BAND[1], color=st.BAND, lw=0, zorder=0)
    ax.axvline(1.0, color=st.MUTED, lw=0.8, zorder=1)
    for y, (cfg, _) in zip(ys, PART_A):
        for cell, colour, marker, dy in [("naive", st.REPLAY, "o", +0.17),
                                         ("selected", st.RECORDED, "s", -0.17)]:
            r = rows[(ch, cfg, cell)]
            v, lo, hi = (float(r[k]) for k in ("burden_ratio", "ci_lo", "ci_hi"))
            ax.plot([lo, hi], [y + dy, y + dy], color=colour, lw=1.3, solid_capstyle="butt",
                    zorder=3)
            ax.plot([v], [y + dy], marker=marker, color=colour, ms=3.9, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 6.0)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.minorticks_off()
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in PART_A] if show_labels else [], fontsize=7.0)
    ax.set_ylim(-0.6, len(PART_A) - 0.4)
    st.tidy(ax, grid_axis="x")


def panel_b(ax, rows, ch, show_labels):
    ys = np.arange(len(PART_B))[::-1]
    ax.axvline(RECALL_CRITERION, color=st.RECORDED, lw=0.7, ls=(0, (1.2, 1.2)), zorder=1)
    ax.axvline(ONSET_CRITERION, color=st.MUTED, lw=0.7, ls=(0, (3, 1.5)), zorder=1)
    for y, (cfg, _) in zip(ys, PART_B):
        r = rows[(ch, cfg, "selected")]
        ax.plot([float(r["recall"])], [y + 0.17], marker="o", color=st.RECORDED, ms=3.9,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.plot([float(r["onset_agreement"])], [y - 0.17], marker="D", color=st.MUTED, ms=3.7,
                markerfacecolor="white", markeredgewidth=0.9, zorder=4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in PART_B] if show_labels else [], fontsize=7.0)
    ax.set_ylim(-0.6, len(PART_B) - 0.4)
    st.tidy(ax, grid_axis="x")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    rows = load()
    fig = figpage.page_figure(plt, 4.6)
    gs = fig.add_gridspec(2, 4, left=0.215, right=0.985, top=0.910, bottom=0.185,
                          wspace=0.14, hspace=0.70)
    for c, (ch, label) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, c])
        panel_a(ax, rows, ch, show_labels=(c == 0))
        ax.set_title(label, fontsize=7.6, pad=4)
        ax = fig.add_subplot(gs[1, c])
        panel_b(ax, rows, ch, show_labels=(c == 0))
        ax.set_title(label, fontsize=7.6, pad=4)
    xmid = 0.5 * (0.215 + 0.985)
    fig.text(xmid, 0.555, "Held-out burden ratio (reconstructed over recorded alarm seconds)",
             fontsize=7.6, ha="center", va="center")
    fig.text(xmid, 0.115, "Recall and onset agreement at the selected cell", fontsize=7.6,
             ha="center", va="center")
    fig.text(0.215, 0.965, "A   Recorded-side conventions", fontsize=8.0, fontweight="bold",
             ha="left", va="center")
    fig.text(0.215, 0.520, "B   Matching conventions", fontsize=8.0, fontweight="bold",
             ha="left", va="center")
    handles = [
        Line2D([0], [0], color=st.REPLAY, marker="o", ms=3.9, lw=1.3, markeredgecolor="white",
               markeredgewidth=0.5, label="Naive cell, burden ratio"),
        Line2D([0], [0], color=st.RECORDED, marker="s", ms=3.9, lw=1.3, markeredgecolor="white",
               markeredgewidth=0.5, label="Selected cell, burden ratio"),
        Line2D([0], [0], color=st.RECORDED, marker="o", ms=3.9, ls="none", markeredgecolor="white",
               markeredgewidth=0.5, label="Recall (criterion 0.90, dotted)"),
        Line2D([0], [0], color=st.MUTED, marker="D", ms=3.7, ls="none", markerfacecolor="white",
               markeredgewidth=0.9, label="Onset agreement (criterion 0.80, dashed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.60, 0.0),
               fontsize=7.0, handlelength=2.0, columnspacing=1.6, labelspacing=0.45)
    figpage.save_fixed(fig, "FigureS3_conventions", OUT, plt)


if __name__ == "__main__":
    main()
