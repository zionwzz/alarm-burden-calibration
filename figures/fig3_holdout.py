#!/usr/bin/env python3
"""Render held-out burden ratios and prespecified validation criteria from the exported plot data."""
import csv
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

PLOT_DATA = paths.PLOTDATA
OUT = paths.FIGOUT

# The channel names, with the chemical subscript typed as its own character rather than set as
# mathematics. Mathtext would shrink it below the production floor at these label sizes, and the
# supplementary figures already write the same name this way.
CHANNEL_LABEL = figpage.typed_subscripts(st.CHANNEL_LABEL)

BAND = (0.80, 1.25)
CRITERIA = [("Burden", "burden"), ("Recall", "recall"), ("Onset", "onset")]


def load():
    with open(os.path.join(PLOT_DATA, "holdout_validation.csv")) as f:
        rows = {r["channel"]: r for r in csv.DictReader(f)}
    return [rows[c] for c in st.CHANNEL_ORDER if c in rows]


def forest(ax, rows):
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(BAND[0], BAND[1], color=st.BAND, lw=0, zorder=0)
    ax.axvline(1.0, color=st.MUTED, lw=0.8, zorder=1)

    for y, r in zip(ys, rows):
        faint = r["channel"] == st.LEAST_STABLE
        for key, colour, marker, dy, lab in [
            ("naive", st.REPLAY, "o", +0.16, "Replay, uncorrected"),
            ("fitted", st.RECORDED, "s", -0.16, "Prespecified delay model"),
        ]:
            v = float(r[f"{key}_ratio"])
            lo, hi = float(r[f"{key}_ci_lo"]), float(r[f"{key}_ci_hi"])
            alpha = 0.45 if faint else 1.0
            ax.plot([lo, hi], [y + dy, y + dy], color=colour, lw=1.5, alpha=alpha,
                    solid_capstyle="butt", zorder=3)
            ax.plot([v], [y + dy], marker=marker, color=colour, ms=4.6, alpha=alpha,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                    label=lab if y == ys[0] else None)

    ax.set_yticks(ys)
    ax.set_yticklabels([CHANNEL_LABEL[r["channel"]] for r in rows], fontsize=7.0)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color("#9AA0A6" if r["channel"] == st.LEAST_STABLE else st.INK)
    ax.set_xscale("log")
    ax.set_xlim(0.72, 3.4)
    ax.set_xticks([0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticklabels(["0.8", "1", "1.25", "1.5", "2", "2.5", "3"])
    ax.set_xlabel("Burden ratio: reconstructed ÷ recorded alarm-seconds", labelpad=1,
                  fontsize=7.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    # a direct label on the shaded swatch, not a note: it names the band and nothing else
    ax.text(np.sqrt(BAND[0] * BAND[1]), len(rows) - 0.52, "prespecified\nvalidation range",
            ha="center", va="top", fontsize=7.0, color=st.MUTED, linespacing=1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    st.tidy(ax, grid_axis="x")


def criteria_matrix(ax, rows):
    ys = np.arange(len(rows))[::-1]
    for j, (lab, key) in enumerate(CRITERIA):
        for y, r in zip(ys, rows):
            if key == "burden":
                passed = BAND[0] <= float(r["fitted_ratio"]) <= BAND[1]
            elif key == "recall":
                passed = float(r["episode_recall"]) >= 0.90
            else:
                passed = float(r["onset_within_10s"]) >= 0.80
            # the least stable channel is de-emphasised in this panel exactly as it is in panel A:
            # same alpha on the filled mark, a lighter rule on the empty one.
            faint = r["channel"] == st.LEAST_STABLE
            alpha = 0.45 if faint else 1.0
            face = st.RECORDED if passed else "white"
            edge = st.RECORDED if passed else ("#DCDFE2" if faint else "#C4C8CC")
            ax.add_patch(Rectangle((j - 0.34, y - 0.30), 0.68, 0.60, facecolor=face,
                                   edgecolor=edge, lw=0.8,
                                   alpha=alpha if passed else 1.0))
            if not passed:
                cross = "#DCDFE2" if faint else "#C4C8CC"
                ax.plot([j - 0.16, j + 0.16], [y - 0.14, y + 0.14], color=cross, lw=0.9)
                ax.plot([j - 0.16, j + 0.16], [y + 0.14, y - 0.14], color=cross, lw=0.9)
    ax.set_xticks(range(len(CRITERIA)))
    ax.set_xticklabels([g[0] for g in CRITERIA], fontsize=7.0)
    ax.set_xlim(-0.62, len(CRITERIA) - 0.38)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.text(0.5, -0.19, "filled = criterion met", transform=ax.transAxes, ha="center", va="top",
            fontsize=7.0, color=st.MUTED)


def main():
    st.use_style()
    rows = load()
    # Drawn at the width it is placed at, with the bounding box switched off, so that a size in
    # points here is that many points on the page (figures/figpage.py).
    figpage.use_page_type(mpl)
    fig = figpage.page_figure(plt, 2.62)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55], left=0.238, right=0.988,
                          top=0.855, bottom=0.375, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    forest(ax_a, rows)
    st.panel(ax_a, "A", "Replay against the recorded stream")
    ax_a.title.set_fontsize(7.8)          # so the title clears the B panel letter
    criteria_matrix(ax_b, rows)
    st.panel(ax_b, "B", "Validation criteria")

    figpage.save_fixed(fig, "Figure3_holdout", OUT, plt)


if __name__ == "__main__":
    main()
