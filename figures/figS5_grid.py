#!/usr/bin/env python3
r"""Replay and linked-reward calibration over the full limit-shift grid, from the link analysis.

`policy/LINK_ANALYSIS.json` in the working tree carries, for every channel and every candidate
shift, the replayed reduction, the duration-transport factor and corrected reduction with their
patient-cluster intervals, and the feature-augmented factor and corrected reduction under the
independent tuning map. The upper row draws the reductions, the lower row the correction factors
against one. Main Figure 5 is the one-step column of the upper row.
"""
import json
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

LINK = os.path.join(paths.WORK, "policy", "LINK_ANALYSIS.json")
OUT = paths.FIGOUT

CHANNELS = [("ecgResp-numLimit-respRate-low", "RR low"),
            ("spO2-numLimit-satO2-low", "SpO₂ low"),
            ("ecg-numLimit-heartRate-high", "HR high"),
            ("ecg-numLimit-heartRate-low", "HR low")]
SHIFTS = ["-1", "1", "2", "3"]
TICK = {"-1": "−1", "1": "+1", "2": "+2", "3": "+3"}
DX = {"replay": -0.18, "lrc_d": 0.0, "lrc_dz": 0.18}
CI_STYLE = {"lrc_d": ("-", 1.3), "lrc_dz": ((0, (2.4, 1.3)), 1.3)}
YMIN, YMAX = -100, 100        # reductions shown; anything beyond is marked at the edge


def load():
    return json.load(open(LINK))


def series(P, ch, s):
    k = P[ch]["kappa"][s]
    z = P[ch]["transport_feature"]["by_offset"][s]
    return {
        "replay": (k["replayed_reduction_pct"], None, None),
        "lrc_d": (k["corrected_reduction_pct"], *k["corrected_CI"]),
        "lrc_dz": (z["corrected_reduction_z_pct"], *z["corrected_z_CI"]),
        "kappa_d": (k["kappa"], *k["CI"]),
        "kappa_z": (z["kappa_z"], *z["CI"]),
    }


def _mark(ax, x, v, lo, hi, colour, marker, key):
    if lo is not None:
        ls, lw = CI_STYLE[key]
        ax.plot([x, x], [lo, hi], color=colour, lw=lw, ls=ls, solid_capstyle="butt",
                dash_capstyle="butt", zorder=3)
    ax.plot([x], [v], marker=marker, color=colour, ms=3.9, markeredgecolor="white",
            markeredgewidth=0.5, zorder=4)


def panel_reduction(ax, P, ch, show_y):
    ax.axhline(0.0, color=st.MUTED, lw=0.8, zorder=1)
    for i, s in enumerate(SHIFTS):
        d = series(P, ch, s)
        for key, colour, marker in [("replay", st.REPLAY, "o"), ("lrc_d", st.LRC_D, "s"),
                                    ("lrc_dz", st.LRC_DZ, "D")]:
            v, lo, hi = d[key]
            x = i + DX[key]
            if v < YMIN:
                # off the scale: an edge mark and the value
                ax.plot([x], [YMIN + 4], marker="v", color=colour, ms=3.6, zorder=4)
                if key == "lrc_dz":
                    ax.text(i - 0.45, YMIN + 14,
                            f"off scale:\n{d['replay'][0]:.0f}% to {d['lrc_dz'][0]:.0f}%".replace("-", "−"),
                            ha="left", va="bottom", fontsize=7.0, color=st.MUTED, linespacing=1.1)
                continue
            if lo is not None:
                lo, hi = max(lo, YMIN), min(hi, YMAX)
            _mark(ax, x, v, lo, hi, colour, marker, key)
    ax.set_xlim(-0.6, len(SHIFTS) - 0.4)
    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels([])
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([-100, -50, 0, 50, 100])
    if show_y:
        ax.set_ylabel("Reduction in linked burden (%)", labelpad=2)
    else:
        ax.set_yticklabels([])
    st.tidy(ax, grid_axis="y")


def panel_kappa(ax, P, ch, show_y):
    ax.axhline(1.0, color=st.MUTED, lw=0.8, zorder=1)
    for i, s in enumerate(SHIFTS):
        d = series(P, ch, s)
        for key, src, colour, marker in [("lrc_d", "kappa_d", st.LRC_D, "s"),
                                         ("lrc_dz", "kappa_z", st.LRC_DZ, "D")]:
            v, lo, hi = d[src]
            _mark(ax, i + DX[key], v, lo, hi, colour, marker, key)
    ax.set_xlim(-0.6, len(SHIFTS) - 0.4)
    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels([TICK[s] for s in SHIFTS])
    ax.set_ylim(0.65, 1.85)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_xlabel("Limit shift (steps)", labelpad=1)
    if show_y:
        ax.set_ylabel("Correction factor", labelpad=2)
    else:
        ax.set_yticklabels([])
    st.tidy(ax, grid_axis="y")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    P = load()
    fig = figpage.page_figure(plt, 4.3)
    gs = fig.add_gridspec(2, 4, left=0.115, right=0.985, top=0.930, bottom=0.200,
                          wspace=0.14, hspace=0.22)
    for c, (ch, label) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, c])
        panel_reduction(ax, P, ch, show_y=(c == 0))
        ax.set_title(label, fontsize=7.6, pad=4)
        ax = fig.add_subplot(gs[1, c])
        panel_kappa(ax, P, ch, show_y=(c == 0))
    handles = [
        Line2D([0], [0], color=st.REPLAY, marker="o", ms=3.9, ls="none", markeredgecolor="white",
               markeredgewidth=0.5, label="Replay, uncorrected"),
        Line2D([0], [0], color=st.LRC_D, marker="s", ms=3.9, ls="-", lw=1.3,
               markeredgecolor="white", markeredgewidth=0.5, label="Calibrated, duration transport"),
        Line2D([0], [0], color=st.LRC_DZ, marker="D", ms=3.9, ls=CI_STYLE["lrc_dz"][0], lw=1.3,
               markeredgecolor="white", markeredgewidth=0.5, label="Calibrated, feature-augmented"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0),
               fontsize=7.0, handlelength=2.2, columnspacing=1.6, labelspacing=0.45)
    figpage.save_fixed(fig, "FigureS5_grid", OUT, plt)


if __name__ == "__main__":
    main()
