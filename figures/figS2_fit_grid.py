#!/usr/bin/env python3
r"""The delay-model fitting grid against the three validation criteria, read from the fitting-grid record.

`delay_model/A2_FIT_GRID.csv` in the working tree holds, for every channel and every cell of the
persistence-by-extension grid, the pooled fitting-set recall, precision, F1, burden ratio and
onset agreement, and the cell the tuning-checked selection kept. One row of panels per channel,
one panel per criterion: the cell colour is the metric, the value is printed in the cell, cells
that meet the criterion are outlined, and the selected cell carries a heavy border. The criteria
were fixed before the held-out data were examined; the held-out adjudication itself is main
Figure 3.
"""
import csv
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402

GRID = os.path.join(paths.WORK, "delay_model", "A2_FIT_GRID.csv")
OUT = paths.FIGOUT

CHANNELS = [("ecgResp-numLimit-respRate-low", "Respiration rate, low"),
            ("spO2-numLimit-satO2-low", "SpO₂, low"),
            ("ecg-numLimit-heartRate-high", "Heart rate, high"),
            ("ecg-numLimit-heartRate-low", "Heart rate, low")]
# (column, title, pass test, colour scale range)
CRITERIA = [("FIT_recall", "Recall\n(criterion: at least 0.90)", lambda v: v >= 0.90, (0.5, 1.0)),
            ("FIT_burden_ratio", "Burden ratio\n(criterion: 0.80 to 1.25)",
             lambda v: 0.80 <= v <= 1.25, (1.0, 2.8)),
            ("FIT_onset_within10", "Onset within 10 s\n(criterion: at least 0.80)",
             lambda v: v >= 0.80, (0.0, 1.0))]
PASS = st.LRC_DZ
RAMP = LinearSegmentedColormap.from_list("grid", ["#F4F6F8", "#8FA3BF", "#1F3F6E"])


def load():
    rows = list(csv.DictReader(open(GRID)))
    taus = sorted({int(r["tau"]) for r in rows})
    gs = sorted({int(r["g"]) for r in rows})
    grid = {}
    for r in rows:
        grid[(r["channel"], int(r["tau"]), int(r["g"]))] = r
    return rows, taus, gs, grid


def panel(ax, grid, ch, taus, gs, column, test, vrange, show_x, show_y):
    lo, hi = vrange
    for j, g in enumerate(gs):
        for i, tau in enumerate(taus):
            r = grid[(ch, tau, g)]
            v = float(r[column])
            # burden ratio: darker means further above the band, so the ramp runs upward from 1
            frac = (v - lo) / (hi - lo)
            colour = RAMP(min(max(frac, 0.0), 1.0))
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=colour, edgecolor="white", lw=0.8,
                                   zorder=1))
            dark = frac > 0.55
            ax.text(i + 0.5, j + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7.0,
                    color="white" if dark else st.INK, zorder=3)
            if test(v):
                ax.add_patch(Rectangle((i + 0.06, j + 0.06), 0.88, 0.88, facecolor="none",
                                       edgecolor=PASS, lw=1.3, zorder=4))
            if r["selected"].strip():
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="none", edgecolor=st.INK, lw=1.6,
                                       zorder=5))
    ax.set_xlim(0, len(taus))
    ax.set_ylim(0, len(gs))
    ax.set_xticks(np.arange(len(taus)) + 0.5)
    ax.set_yticks(np.arange(len(gs)) + 0.5)
    ax.set_xticklabels([str(t) for t in taus] if show_x else [])
    ax.set_yticklabels([str(g) for g in gs] if show_y else [])
    ax.tick_params(length=0, pad=2)
    if show_x:
        ax.set_xlabel("Persistence τ (s)", labelpad=1)
    if show_y:
        ax.set_ylabel("Gap g (s)", labelpad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def check_statements(rows):
    """The two statements the supplement makes about this grid, checked before drawing."""
    passes = lambda r: [test(float(r[col])) for col, _, test, _ in CRITERIA]
    if any(all(passes(r)) for r in rows):
        raise SystemExit("a fitting-grid cell meets all three criteria; the supplement says none does")
    if any(passes(r)[1] and float(r["FIT_recall"]) >= 0.90 for r in rows):
        raise SystemExit("a cell meets the burden criterion with recall at least 0.90; the supplement "
                         "says recall falls below 0.90 wherever the burden criterion is met")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    rows, taus, gs, grid = load()
    check_statements(rows)
    fig = figpage.page_figure(plt, 4.4)
    gsx = fig.add_gridspec(len(CHANNELS), len(CRITERIA), left=0.125, right=0.985, top=0.905,
                           bottom=0.135, wspace=0.10, hspace=0.42)
    for r, (ch, label) in enumerate(CHANNELS):
        for c, (column, title, test, vrange) in enumerate(CRITERIA):
            ax = fig.add_subplot(gsx[r, c])
            panel(ax, grid, ch, taus, gs, column, test, vrange,
                  show_x=(r == len(CHANNELS) - 1), show_y=(c == 0))
            if r == 0:
                ax.set_title(title, fontsize=7.4, pad=5, linespacing=1.1)
            if c == 0:
                ax.text(-0.42, 0.5, label, transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=7.4, color=st.INK)
    # key: pass outline and selected cell
    kx, ky = 0.50, 0.030
    fig.patches.append(Rectangle((kx - 0.13, ky - 0.012), 0.014, 0.020, transform=fig.transFigure,
                                 facecolor="none", edgecolor=PASS, lw=1.3))
    fig.text(kx - 0.11, ky, "meets the criterion", fontsize=7.0, va="center", color=st.INK)
    fig.patches.append(Rectangle((kx + 0.10, ky - 0.012), 0.014, 0.020, transform=fig.transFigure,
                                 facecolor="none", edgecolor=st.INK, lw=1.6))
    fig.text(kx + 0.12, ky, "selected cell", fontsize=7.0, va="center", color=st.INK)
    figpage.save_fixed(fig, "FigureS2_fit_grid", OUT, plt)


if __name__ == "__main__":
    main()
