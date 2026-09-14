#!/usr/bin/env python3
r"""Load aggregate reward data and render duration panels.

The submitted figure is assembled by fig4_combined.py. Nominal short-duration
bins use unconnected estimates; pooled tails occupy four equal-width slots.
Open markers indicate fewer than 20 excursions. Heat-map omissions reflect
the supplied aggregate export, including its 30-excursion reporting minimum.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

MIN_PRECISE = 20          # analysis/link_analysis.py: support_diagnostics(min_for_precision=20)
REPORT_FLOOR = 30         # the exported aggregate's reporting floor for a duration-by-overshoot cell

EXACT_MAX = 120.0                 # the exact cells end here
BREAK = 12.0                      # the visual break between exact cells and pooled tail bins
SLOT = 15.0                       # plotting width of one pooled tail bin
TAIL0 = EXACT_MAX + BREAK
Y0, Y1 = -0.105, 1.20                    # room for the rug below and for every interval above

# the two blank-cell fills of the overshoot figure, named once and used by both the cells and the key
# Both carry a visible outline, so that a blank cell reads as a marked-off blank rather than as a
# near-zero value: the pale end of the colour ramp is close in tone to a flat light grey, and in a
# greyscale print the outline is what separates them.
BLANK_UNDER_FLOOR = dict(facecolor="white", edgecolor="#AEB3B8", lw=0.5, hatch="////")
BLANK_NO_CELL = dict(facecolor="#E7E9EA", edgecolor="#AEB3B8", lw=0.5)

DURATION_GROUPS = ["<10s", "10-20s", "20-40s", "40-120s", ">=120s"]
DURATION_TICK = {"<10s": "<10", "10-20s": "10–20", "20-40s": "20–40",
                 "40-120s": "40–120", ">=120s": "≥120"}


def tail_slot(i):
    """Centre of the i-th pooled tail bin, in plotting units."""
    return TAIL0 + SLOT * (i + 0.5)


def load_cells():
    """Every duration cell the estimator uses: the exact cells and the pooled tail cells alike."""
    out = defaultdict(lambda: {"exact": [], "pooled": []})
    with open(os.path.join(PLOT_DATA, "reward_by_cell.csv")) as f:
        for r in csv.DictReader(f):
            if not r["annunciation_ratio"]:
                continue
            out[r["channel"]][r["cell_kind"]].append(
                (float(r["mean_duration_s"]), float(r["annunciation_ratio"]),
                 float(r["ratio_ci_lo"]), float(r["ratio_ci_hi"]), int(r["excursions"])))
    for k in out:
        for kind in out[k]:
            out[k][kind].sort()
    return out


def load_overshoot():
    out = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(PLOT_DATA, "reward_by_overshoot.csv")) as f:
        for r in csv.DictReader(f):
            out[r["channel"]][r["duration_group"]].append(
                (int(r["overshoot_bin"]), float(r["annunciation_ratio"]), int(r["excursions"])))
    for ch in out:
        for g in out[ch]:
            out[ch][g].sort()
    return out


def ratio_panel(ax, cells, faint, show_y):
    exact, pooled = cells["exact"], cells["pooled"]
    d = np.array([r[0] for r in exact])
    h = np.array([r[1] for r in exact])
    lo = np.array([r[2] for r in exact])
    hi = np.array([r[3] for r in exact])
    n = np.array([r[4] for r in exact])
    precise = n >= MIN_PRECISE
    col = st.UNSUPPORTED if faint else st.LRC_D
    thin = "#C4C8CC" if faint else "#9FB4D4"          # the same hue, lightened: still the channel

    # the pooled tail region, marked as a region so it is never read as more exact cells
    ax.axvspan(TAIL0, TAIL0 + 4 * SLOT, color="#F1F2F3", lw=0, zorder=0)
    for k in (1, 2, 3):
        ax.plot([TAIL0 + k * SLOT] * 2, [Y0, Y1], color="#E3E5E8", lw=0.5, zorder=0)

    # Independent per-cell estimates: drawn as points with their pointwise intervals and never
    # joined. The estimator fits no shape, so a connecting line would assert a smoothness and a
    # monotonicity that are not in the estimate; 28-45% of consecutive exact cells in fact step
    # downward (SpO2 low 28%, respiration rate low 38%, both heart-rate channels 45%).
    # Every cell gets its interval. Cells below the precision floor are the ones whose interval is
    # widest and most worth seeing, so they are marked open rather than left bare.
    ax.vlines(d[precise], lo[precise], hi[precise], color=col, lw=0.7, alpha=0.55, zorder=1)
    ax.vlines(d[~precise], lo[~precise], hi[~precise], color=thin, lw=0.6, alpha=0.85, zorder=1)
    ax.plot(d[precise], h[precise], ls="none", marker="o", ms=2.2, color=col, zorder=3)
    if (~precise).any():
        ax.plot(d[~precise], h[~precise], ls="none", marker="o", ms=2.4, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=0.6, zorder=3)

    # the pooled tail cells: a bar over the bin's slot, not a point, because the cell is a bin
    for i, (dp, hp, lop, hip, npx) in enumerate(pooled):
        x = tail_slot(i)
        pc = col if npx >= MIN_PRECISE else thin
        ax.plot([x - 0.36 * SLOT, x + 0.36 * SLOT], [hp, hp], color=pc, lw=1.7,
                solid_capstyle="butt", zorder=3)
        ax.vlines(x, lop, hip, color=pc, lw=0.8, alpha=0.75, zorder=2)
        for yc in (lop, hip):
            ax.plot([x - 0.09 * SLOT, x + 0.09 * SLOT], [yc, yc], color=pc, lw=0.7, zorder=2)

    # occupancy rug: where the replayed burden sits across the exact cells
    w = n * d
    w = w / w.max() if w.max() else w
    for di, wi in zip(d, w):
        ax.plot([di, di], [Y0 + 0.012, Y0 + 0.012 + 0.048 * wi], color="#B0B5BA", lw=0.9,
                solid_capstyle="butt", zorder=1)
    # the tail's share of burden-weighted occupancy: a measured value annotating the shaded region,
    # not a note about it
    share = sum(r[0] * r[4] for r in pooled) / sum(r[0] * r[4] for r in exact + pooled)
    ax.text(TAIL0 + 2 * SLOT, Y1 - 0.01, f"{100 * share:.0f}%\nof burden", ha="center", va="top",
            fontsize=7.0, color=st.MUTED, linespacing=1.25)

    ax.set_xlim(-4, TAIL0 + 4 * SLOT + 2)
    ax.set_ylim(Y0, Y1)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if show_y:
        # 8 pt, not the sheet default: this label carries a subscript, and mathtext sets a
        # subscript at a fraction of its base (figures/figpage.py, MATHTEXT_BASE_PT).
        ax.set_ylabel("Annunciation ratio\n" + r"$\bar a_0(d)/d$", labelpad=2, linespacing=1.4,
                      fontsize=8.0)
    else:
        ax.set_yticklabels([])
    st.tidy(ax, grid_axis="y")

    # the axis break, drawn after tidy() so it sits on top of the spine
    xb = EXACT_MAX + BREAK / 2
    ax.plot([xb - 3.2, xb + 3.2], [Y0, Y0], color="white", lw=2.2, clip_on=False, zorder=5)
    for dx in (-1.6, 1.4):
        ax.plot([xb + dx - 1.1, xb + dx + 1.1], [Y0 - 0.028, Y0 + 0.028], color="#9AA0A6",
                lw=0.7, clip_on=False, zorder=6)
    # a direct label on the shaded region, so the break needs no sentence to explain it. It sits
    # under the tick row, not beside it, so it can never crowd the 120 tick.
    ax.annotate("pooled\n$\\geq$120 s", xy=(TAIL0 + 2 * SLOT, Y0), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                fontsize=7.0, color=st.MUTED, linespacing=1.2, annotation_clip=False)


def overshoot_panel(ax, groups, show_y, vmin, vmax, cmap):
    """Duration by overshoot, as a heat map.

    The estimates are independent per-cell means, so they are not joined: a line across overshoot
    bins would assert a monotone trend, and on heart rate high the relationship is neither monotone
    nor strong. Two kinds of blank cell occur and they are drawn differently, because they mean
    different things: an overshoot bin in which the aggregate reports no cell at all on this channel,
    and a single cell it does not report because that cell holds fewer than 30 excursions.
    """
    labels = [g for g in DURATION_GROUPS if g in groups]
    grid = np.full((len(labels), 6), np.nan)
    for i, g in enumerate(labels):
        for b, h, _n in groups[g]:
            grid[i, b] = h
    reported = np.isfinite(grid).any(axis=0)          # bins carrying at least one reported cell

    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    for j in range(grid.shape[1]):
        for i in range(grid.shape[0]):
            if np.isfinite(grid[i, j]):
                continue
            style = BLANK_UNDER_FLOOR if reported[j] else BLANK_NO_CELL
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, **style))
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16+"], fontsize=7.0)
    ax.set_yticks(range(len(labels)))
    if show_y:
        ax.set_yticklabels([DURATION_TICK[g] for g in labels], fontsize=7.0)
        ax.set_ylabel("Excursion duration (s)", labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)


def duration_figure(cells, chans):
    """Figure 3: the duration-marginal ratio, one panel per channel.

    Two rows of two rather than one row of four. At the width this figure is placed at, a row of
    four leaves each panel about an inch wide, which is narrower than the longest channel name: the
    titles would have had to be abbreviated or set below the floor. Two rows of two keep every panel
    title and every tick at full size.
    """
    fig = figpage.page_figure(plt, 4.20)
    gs = fig.add_gridspec(2, 2, left=0.140, right=0.990, top=0.935, bottom=0.225,
                          hspace=0.50, wspace=0.15)
    for j, ch in enumerate(chans):
        ax = fig.add_subplot(gs[divmod(j, 2)])
        ratio_panel(ax, cells[ch], ch == st.LEAST_STABLE, show_y=(j % 2 == 0))
        ax.set_title(CHANNEL_LABEL[ch], pad=3, fontsize=7.6,
                     color="#9AA0A6" if ch == st.LEAST_STABLE else st.INK, fontweight="normal")
    # one shared axis label under the block: four copies of it were four copies of the same words
    fig.supxlabel("Duration $d$ (s)", y=0.085, fontsize=8.0, color=st.INK)

    # a marker key, not a note: it says what an open marker is and nothing more
    handles = [
        Line2D([0], [0], color=st.LRC_D, marker="o", ms=3.4, ls="none",
               label="cell with $\\geq$%d excursions" % MIN_PRECISE),
        Line2D([0], [0], color=st.LRC_D, marker="o", ms=3.6, ls="none", markerfacecolor="white",
               markeredgewidth=0.7, label="fewer than %d" % MIN_PRECISE),
        Line2D([0], [0], color="#B0B5BA", lw=1.6, label="occupancy rug"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.006), ncol=3,
               fontsize=7.0, handlelength=1.4, columnspacing=1.4)
    figpage.save_fixed(fig, "Figure4_reward_duration", OUT, plt)


def overshoot_figure(over, chans):
    """Figure 3b: the same ratio by duration group and overshoot bin, as heat maps."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ratio", ["#F4F1EC", "#E3B778", st.LRC_DZ, "#6E3405"])
    allv = [h for ch in chans for g in over[ch] for _b, h, _n in over[ch][g]]
    vmin, vmax = 0.0, max(allv) if allv else 1.0

    # Two rows of two, for the same reason as the duration figure: a row of four at this width
    # leaves no panel wide enough for its own channel name at full size.
    fig = figpage.page_figure(plt, 4.05)
    gs = fig.add_gridspec(2, 2, left=0.128, right=0.990, top=0.945, bottom=0.334,
                          hspace=0.40, wspace=0.13)
    im = None
    for j, ch in enumerate(chans):
        ax = fig.add_subplot(gs[divmod(j, 2)])
        overshoot_panel(ax, over[ch], show_y=(j % 2 == 0), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(CHANNEL_LABEL[ch], pad=3, fontsize=7.6,
                     color="#9AA0A6" if ch == st.LEAST_STABLE else st.INK, fontweight="normal")
        if j == 0:
            im = ax.images[0]
    fig.supxlabel("Overshoot beyond the limit (channel steps)", y=0.258, fontsize=8.0,
                  color=st.INK)

    cax = fig.add_axes([0.128, 0.151, 0.32, 0.024])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    # The scale runs past one because the ratio is not a fraction. Matplotlib's automatic ticks stop
    # at 1.0 and leave the top of the bar unlabelled, so the maximum is ticked explicitly.
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    # the top of the scale, named above the bar so it does not crowd the 1.0 tick below it
    cb.ax.annotate(f"max {vmax:.2f}", xy=(vmax, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(0, 2.5), textcoords="offset points", ha="right", va="bottom",
                   fontsize=7.0, color=st.MUTED)
    cb.set_label(r"Annunciation ratio $a_0(d,z)/d$", fontsize=8.0, labelpad=2)
    cb.ax.tick_params(labelsize=7.0, length=2)
    cb.outline.set_visible(False)

    # the two blank-cell fills, as swatches. The exported aggregate carries no count for a cell it
    # does not report, so a bin with no reported cell cannot be told from one with only sub-floor
    # cells: the key says "no cell reported", which is what the file supports.
    # "in this bin" rather than "in this overshoot bin": the axis beneath the key already names
    # the bins as overshoot bins, and the full phrase does not fit the column at 7 pt.
    handles = [Patch(label="fewer than %d excursions: no ratio reported" % REPORT_FLOOR,
                     **BLANK_UNDER_FLOOR),
               Patch(label="no cell reported in this bin on this channel",
                     **BLANK_NO_CELL)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.470, 0.212), ncol=1,
               fontsize=7.0, handlelength=1.5, handleheight=1.1, labelspacing=0.5,
               borderaxespad=0.0)
    figpage.save_fixed(fig, "Figure4b_reward_overshoot", OUT, plt)


def main():
    st.use_style()
    # Drawn at the width they are placed at, with the bounding box switched off, so that a size in
    # points here is that many points on the page (figures/figpage.py).
    figpage.use_page_type(mpl)
    cells = load_cells()
    over = load_overshoot()
    chans = [c for c in st.CHANNEL_ORDER if c in cells]
    duration_figure(cells, chans)
    overshoot_figure(over, [c for c in st.CHANNEL_ORDER if c in over])


if __name__ == "__main__":
    main()
