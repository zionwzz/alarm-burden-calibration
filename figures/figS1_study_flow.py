#!/usr/bin/env python3
r"""Study flow: from the release to the analysis sets, read from the exported design records.

Counts come from `design.json` in the plot-data directory (the cohort block the plot-data exporter
writes from the alarm-epidemiology summary and the split record) and from
`diagnostics/COHORT_COMPLETENESS.json` in the working tree (the intended cohort of every scan job
and the admissions no scanner could read). Nothing is typed in by hand.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402

DESIGN = os.path.join(paths.PLOTDATA, "design.json")
COMPLETENESS = os.path.join(paths.WORK, "diagnostics", "COHORT_COMPLETENESS.json")
OUT = paths.FIGOUT

W, H = figpage.PLACEMENT_WIDTH_IN, 4.35     # inches; the axes are drawn in these units
EDGE = "#9AA0A6"
FILL = "white"
SPLIT_FILL = "#EEF2F8"
COL_W, COL_GAP = 1.62, 0.17                 # three columns fill the placement width exactly


def n(x):
    return f"{int(round(x)):,}"


def load():
    d = json.load(open(DESIGN))["cohort"]
    c = json.load(open(COMPLETENESS))
    jobs = {j["job"]: j for j in c["jobs"]}
    unreadable = jobs["a2_FIT"]["missing_file"] + jobs["a2_FIT"]["read_error"]
    return d, jobs, unreadable


def box(ax, x, y, w, h, text, fill=FILL, size=7.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=fill, edgecolor=EDGE, linewidth=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=st.INK,
            linespacing=1.2, zorder=3, fontweight=weight)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=st.MUTED, lw=0.7, shrinkA=0, shrinkB=0,
                                mutation_scale=7), zorder=1)


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    d, jobs, unreadable = load()
    sp = d["splits"]
    fig = figpage.page_figure(plt, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    xc = W / 2

    # row 1: the release
    y1 = H - 0.40
    box(ax, xc - 1.60, y1, 3.2, 0.34,
        f"ALOTT release: {n(d['admissions'])} admissions, {n(d['patients'])} patients",
        size=7.6, weight="bold")
    # row 2: admissions with alarm rows, and those that contribute to no denominator
    y2 = H - 1.02
    box(ax, 0.55, y2, 2.95, 0.46,
        f"{n(d['admissions_with_alarm_rows'])} admissions with recorded alarm rows\n"
        f"{n(d['admission_days'])} admission-days, {n(d['annunciation_runs'])} runs")
    box(ax, 3.68, y2, 1.42, 0.46,
        f"{n(d['alarm_file_without_rows'])} without alarm rows,\n{n(d['no_alarm_file'])} without an alarm file\n"
        "(descriptive tables only)", size=7.0)
    arrow(ax, xc, y1, xc, y2 + 0.46)
    arrow(ax, 3.50, y2 + 0.23, 3.68, y2 + 0.23)

    # the split: a stem from row 2 to a bus, then one arrow down into each set
    y_bus = H - 1.34
    ax.plot([xc, xc], [y2, y_bus], color=st.MUTED, lw=0.7, zorder=1)
    ax.text(xc + 0.06, (y2 + y_bus) / 2, "patient-level split, 50 / 25 / 25,\nfixed before any model was fitted",
            ha="left", va="center", fontsize=7.0, color=st.MUTED, style="italic", linespacing=1.1)
    xs = [COL_GAP / 2 + i * (COL_W + COL_GAP) for i in range(3)]
    centres = [x + COL_W / 2 for x in xs]
    ax.plot([centres[0], centres[-1]], [y_bus, y_bus], color=st.MUTED, lw=0.7, zorder=1)
    y3 = H - 1.96
    for x, c, (name, key) in zip(xs, centres, [("Fitting set", "FIT"), ("Tuning set", "TUNE"),
                                              ("Held-out set", "TEST")]):
        box(ax, x, y3, COL_W, 0.50,
            f"{name}\n{n(sp[key]['admissions'])} admissions\n{n(sp[key]['patients'])} patients",
            fill=SPLIT_FILL, size=7.2, weight="bold")
        arrow(ax, c, y_bus, c, y3 + 0.50)

    # what each set is used for, with the intended cohort of the corresponding scan job
    uses = [
        [f"Delay-model fitting grid and\ncrossing-to-annunciation\nscan: all {n(jobs['a2_FIT']['intended'])}",
         f"Quarter A, every fourth\nadmission: {n(jobs['lk']['intended'])}\nlinkage, reward, replay",
         f"Quarter B: {n(jobs['lk2_FIT4p2']['intended'])}\nreproducibility of the reward"],
        [f"Delay-model check on the\nfitting grid: all {n(jobs['a2_TUNE']['intended'])}",
         f"Feature-map selection,\nevery second admission:\n{n(jobs['lk2_TUNE2p0']['intended'])}"],
        [f"Delay-model adjudication,\nread once: all {n(jobs['a2_TEST']['intended'])}",
         f"Measurement-convention\nscan: all {n(jobs['s2']['intended'])}"],
    ]
    bh, gap = 0.50, 0.12
    for x, c, col in zip(xs, centres, uses):
        y = y3
        for u in col:
            y -= gap + bh
            box(ax, x, y, COL_W, bh, u, size=7.0)
            arrow(ax, c, y + bh + gap, c, y + bh)
    ax.text(COL_GAP / 2, 0.08,
            f"{unreadable} fitting-set admissions could not be read by any scanner ({jobs['a2_FIT']['missing_file']} without "
            f"telemetry files,\n{jobs['a2_FIT']['read_error']} with corrupt measurement archives); every job otherwise "
            "covered its whole intended set.",
            ha="left", va="bottom", fontsize=7.0, color=st.MUTED, linespacing=1.2)
    figpage.save_fixed(fig, "FigureS1_study_flow", OUT, plt)


if __name__ == "__main__":
    main()
