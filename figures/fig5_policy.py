#!/usr/bin/env python3
r"""Render the one-step comparison and its calibration shift from supplied aggregates."""
import csv
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

PLOT_DATA = paths.PLOTDATA
OUT = paths.FIGOUT

# The channel names, with the chemical subscript typed as its own character rather than set as
# mathematics. Mathtext would shrink it below the production floor at these label sizes, and the
# supplementary figures already write the same name this way.
CHANNEL_LABEL = figpage.typed_subscripts(st.CHANNEL_LABEL)

SHIFT = "1"

# One sub-row per estimate, so identity is carried by vertical position, marker shape and interval
# line style as well as by hue: the three interval styles stay separable in a greyscale print, where
# the palette's blue, orange and grey all fall within twenty luminance steps of one another.
ROW = {"replay": 0.00, "lrc_d": +0.17, "lrc_dz": -0.17}
CI_STYLE = {"lrc_d": ("-", 1.4), "lrc_dz": ((0, (2.4, 1.3)), 1.4)}
CAP = 0.085                      # half-height of the marker-anchored interval caps, in row units


def _interval(ax, lo, hi, y, colour, alpha, key):
    """A confidence interval drawn so that it reads without colour: own row, own dash, own caps."""
    ls, lw = CI_STYLE[key]
    ax.plot([lo, hi], [y, y], color=colour, lw=lw, ls=ls, alpha=alpha,
            solid_capstyle="butt", dash_capstyle="butt", zorder=3)
    for x in (lo, hi):
        ax.plot([x, x], [y - CAP, y + CAP], color=colour, lw=1.0, alpha=alpha, zorder=3)


def load():
    rows = {}
    with open(os.path.join(PLOT_DATA, "policy_change.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["channel"], r["shift_steps"])] = r
    return rows


def panel(ax, rows, shift, show_y=True):
    chans = [c for c in st.CHANNEL_ORDER if (c, shift) in rows]
    ys = np.arange(len(chans))[::-1]
    ax.axvline(0.0, color="#C4C8CC", lw=0.8, zorder=1)

    for y, ch in zip(ys, chans):
        r = rows[(ch, shift)]
        faint = ch == st.LEAST_STABLE
        # the column is a pooled-tail crowding flag, whatever its name says: see the docstring
        tail_sensitive = r["outside_support"] == "yes"
        alpha = 0.40 if (faint or tail_sensitive) else 1.0

        rep = float(r["replayed_change_pct"])
        d = float(r["corrected_duration_pct"])
        dz = float(r["corrected_overshoot_pct"]) if r["corrected_overshoot_pct"] else np.nan

        # the scenario range, as a band across the two calibrated sub-rows rather than a bar on one
        # line: it spans the two point estimates and nothing else, and it is not an interval.
        if np.isfinite(dz):
            ax.add_patch(Rectangle((min(d, dz), y + ROW["lrc_dz"] - 0.055),
                                   abs(d - dz), ROW["lrc_d"] - ROW["lrc_dz"] + 0.11,
                                   facecolor="#DCDFE2", edgecolor="none", alpha=alpha, zorder=1))

        for key, val, lo, hi, colour, marker in [
            ("replay", rep, None, None, st.REPLAY, "o"),
            ("lrc_d", d, float(r["corrected_duration_ci_lo"]),
             float(r["corrected_duration_ci_hi"]), st.LRC_D, "s"),
            ("lrc_dz", dz,
             float(r["corrected_overshoot_ci_lo"]) if r["corrected_overshoot_ci_lo"] else None,
             float(r["corrected_overshoot_ci_hi"]) if r["corrected_overshoot_ci_hi"] else None,
             st.LRC_DZ, "D"),
        ]:
            if not np.isfinite(val):
                continue
            yy = y + ROW[key]
            if lo is not None and hi is not None:
                _interval(ax, lo, hi, yy, colour, alpha * 0.9, key)
            ax.plot([val], [yy], marker=marker, color=colour, ms=4.2, alpha=alpha,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)

        if tail_sensitive:
            ax.text(0.0, y, "  pooled-tail sensitive", va="center", fontsize=7.0,
                    color=st.MUTED, style="italic")

    ax.set_yticks(ys)
    if show_y:
        ax.set_yticklabels([CHANNEL_LABEL[c] for c in chans], fontsize=7.0)
        for lab, ch in zip(ax.get_yticklabels(), chans):
            lab.set_color("#9AA0A6" if ch == st.LEAST_STABLE else st.INK)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(chans) - 0.4)
    st.tidy(ax, grid_axis="x")


def shift_panel(ax, rows, shift):
    """Calibration shift: calibrated minus replayed reduction, in percentage points."""
    chans = [c for c in st.CHANNEL_ORDER if (c, shift) in rows]
    ys = np.arange(len(chans))[::-1]
    ax.axvline(0.0, color=st.REPLAY, lw=0.9, zorder=2)
    for y, ch in zip(ys, chans):
        r = rows[(ch, shift)]
        faint = ch == st.LEAST_STABLE
        alpha = 0.40 if faint else 1.0
        rep = float(r["replayed_change_pct"])
        for key, slot, colour, marker in [("corrected_duration_pct", "lrc_d", st.LRC_D, "s"),
                                          ("corrected_overshoot_pct", "lrc_dz", st.LRC_DZ, "D")]:
            if not r[key]:
                continue
            # same sub-rows as panel A, so a reader tracks one estimate across the two panels
            ax.plot([float(r[key]) - rep], [y + ROW[slot]], marker=marker, color=colour, ms=4.2,
                    alpha=alpha, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(chans) - 0.4)
    st.tidy(ax, grid_axis="x")


def main():
    st.use_style()
    # Drawn at the width it is placed at, with the bounding box switched off, so that a size in
    # points here is that many points on the page (figures/figpage.py).
    figpage.use_page_type(mpl)
    rows = load()
    fig = figpage.page_figure(plt, 2.45)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.60], left=0.228, right=0.976,
                          top=0.860, bottom=0.430, wspace=0.11)

    ax_a = fig.add_subplot(gs[0, 0])
    panel(ax_a, rows, SHIFT)
    st.panel(ax_a, "A", "Reduction in linked alarm burden")
    ax_a.title.set_fontsize(7.8)          # so the title clears the B panel letter
    ax_a.set_xlim(18, 96)
    ax_a.set_xticks([20, 40, 60, 80])
    # "Reduction, one-step ..." rather than "Reduction from a one-step ...": at 7 pt the longer
    # phrase is wider than the panel and runs into the second panel's own axis label.
    ax_a.set_xlabel("Reduction, one-step less stringent limit (%)", labelpad=1, fontsize=7.4)

    ax_b = fig.add_subplot(gs[0, 1])
    shift_panel(ax_b, rows, SHIFT)
    st.panel(ax_b, "B", "Calibration shift")
    ax_b.set_xlim(-3.2, 9.5)
    ax_b.set_xticks([-2, 0, 2, 4, 6, 8])
    ax_b.set_xlabel("Calibrated $-$ replayed (points)", labelpad=1, fontsize=7.4)

    # proxy handles, so the key shows the interval's dash pattern and caps and not only its hue
    handles = [
        Line2D([0], [0], color=st.REPLAY, marker="o", ms=4.2, ls="none",
               markeredgecolor="white", markeredgewidth=0.5, label="Replay, uncorrected"),
        Line2D([0], [0], color=st.LRC_D, marker="s", ms=4.2, ls=CI_STYLE["lrc_d"][0], lw=1.4,
               markeredgecolor="white", markeredgewidth=0.5,
               label="Calibrated, duration transport"),
        Line2D([0], [0], color=st.LRC_DZ, marker="D", ms=4.2, ls=CI_STYLE["lrc_dz"][0], lw=1.4,
               markeredgecolor="white", markeredgewidth=0.5,
               label="Calibrated, feature-augmented"),
        Patch(facecolor="#DCDFE2", edgecolor="none", label="Scenario range"),
    ]
    # The key names the marks and nothing else. What the intervals are, and what the scenario range
    # is not, is a statement about the estimates and belongs in the caption.
    # two rows rather than four columns: a one-row key is wider than the axes, and the tight
    # bounding box then widens the whole figure, which costs type size on placement.
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.060),
               fontsize=7.0, handlelength=2.2, columnspacing=1.8, labelspacing=0.55)
    figpage.save_fixed(fig, "Figure5_policy", OUT, plt)


if __name__ == "__main__":
    main()
