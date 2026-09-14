#!/usr/bin/env python3
r"""Draw the simulation's data-generating laws, read from the simulation code itself.

The duration law is shown as the share of excursions per two-second bin in a large draw from
`simulation/simulate_linked_reward.py`'s own routines (fixed seed; a display of the law, not a
result), and the reward and non-annunciation laws are evaluated from the same module's functions,
so the picture cannot drift from the generator. Nothing here feeds any table.

A: excursion durations at the limit in force and under the candidate limit for the reference law,
   and for scenario E, whose baseline is capped at 100 seconds and whose candidate lengthens
   excursions.
B: mean positive reward by duration for the laws that do not depend on overshoot.
C: mean positive reward by overshoot class in scenarios B and C, with the composition shift.
D: probability of no annunciation by duration in scenarios I and J.

Scenario letters in this file are the simulation's keys; the figure prints the manuscript's
letters through `alarmreplay/scenarios.py`.
"""
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "simulation"))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402
import simulate_linked_reward as sim  # noqa: E402
from alarmreplay.scenarios import PAPER_LETTER as L  # noqa: E402

OUT = paths.FIGOUT

SEED = 20260905
N_PATIENTS = 20000
BIN = 2.0
DMAX = 160.0
# overshoot classes are ordinal: one hue, light to dark
Z_COLOURS = ["#E7B07A", "#D68A3F", "#B65E12", "#7A3B05"]
BASELINE = st.RECORDED
CANDIDATE = st.REPLAY


def duration_shares(rng, scenario, u):
    """Share of excursions in each two-second bin, from the generator's own duration routine."""
    k = 4.0
    frail = rng.gamma(k, 1.0, N_PATIENTS) / k
    n_exc = np.maximum(rng.poisson(25.0 * frail), 1)
    scale = sim.duration_scale(scenario, u)
    d = np.concatenate([sim.draw_durations(rng, int(n), scale, f, scenario, u)
                        for n, f in zip(n_exc, frail)])
    edges = np.arange(0.0, 402.0 + BIN, BIN)
    counts, _ = np.histogram(d, bins=edges)
    return edges[:-1] + BIN, counts / counts.sum()      # bin labelled by the grid value it holds


def panel_durations(ax, rng):
    ax.axvspan(60, 120, color=st.BAND, lw=0, zorder=0)
    ax.axvspan(120, DMAX, color="#DCDFE2", lw=0, zorder=0)
    # E: the limit in force is capped at 100 s, so beyond it the candidate's excursions fall in
    # cells that hold no baseline excursion at all
    ax.axvspan(100, DMAX, facecolor="none", edgecolor="#9AA0A6", hatch="////", lw=0, zorder=1)
    ax.axvline(sim.THETA, color="#C4C8CC", lw=0.8, ls=":", zorder=1)
    series = [("A", 0, BASELINE, "-", f"Limit in force ({L['E']}: capped at 100 s)"),
              ("A", 1, CANDIDATE, "-", "Candidate, reference law"),
              ("E", 1, CANDIDATE, (0, (2.4, 1.3)), f"Candidate, scenario {L['E']}")]
    handles = []
    for sc, u, colour, ls, label in series:
        x, share = duration_shares(rng, sc, u)
        keep = x <= DMAX
        ax.plot(x[keep], share[keep], color=colour, lw=1.2, ls=ls, zorder=3)
        handles.append(Line2D([0], [0], color=colour, lw=1.2, ls=ls, label=label))
    ax.set_xlim(0, DMAX)
    ax.set_ylim(0, 0.13)
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Duration (s)", labelpad=1)
    ax.set_ylabel("Share of excursions per 2-s bin", labelpad=2)
    ax.text(90, 0.030, "pooled\n[60, 120)", fontsize=7.0, color=st.MUTED, ha="center",
            va="bottom", linespacing=1.05)
    ax.text(140, 0.030, "pooled\n[120, ∞)", fontsize=7.0, color=st.MUTED, ha="center",
            va="bottom", linespacing=1.05)
    ax.text(130, 0.050, f"{L['E']}: no baseline\nexcursions\nbeyond 100 s", fontsize=7.0, color=st.INK,
            ha="center", va="bottom", linespacing=1.05)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=7.0,
              handlelength=2.2)
    st.tidy(ax, grid_axis="y")


def reward_mean(d, z, scenario, limit):
    ann, _ = sim._reward_components(d, np.full(d.shape, z, dtype=int), scenario, limit)
    return ann


def panel_reward_common(ax):
    d = np.arange(2.0, DMAX + 1, 2.0)
    ax.axvline(sim.THETA, color="#C4C8CC", lw=0.8, ls=":", zorder=1)
    common = ", ".join(sorted(L[k] for k in ("A", "E", "G", "I", "J")))
    for sc, limit, colour, ls, label in [("A", 0, BASELINE, "-", f"{common}: 0.85"),
                                         ("D", 0, "#4A5058", (0, (4, 1.5, 1, 1.5)),
                                          f"{L['D']}: rising to 0.95"),
                                         ("F", 1, CANDIDATE, (0, (2.4, 1.3)), f"{L['F']}, candidate: 0.68")]:
        ax.plot(d, reward_mean(d, 0, sc, limit), color=colour, lw=1.2, ls=ls, zorder=3,
                label=label)
    ax.set_xlim(0, DMAX)
    ax.set_ylim(0, 150)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Duration $d$ (s)", labelpad=1)
    ax.set_ylabel("Mean positive reward (s)", labelpad=2)
    ax.legend(loc="upper left", fontsize=7.0, handlelength=2.2, title="Multiplier of (d − 8)\u208a",
              title_fontsize=7.0, alignment="left")
    st.tidy(ax, grid_axis="y")


def panel_reward_by_class(ax):
    d = np.arange(2.0, DMAX + 1, 2.0)
    ax.axvline(sim.THETA, color="#C4C8CC", lw=0.8, ls=":", zorder=1)
    for z in range(4):
        ax.plot(d, reward_mean(d, z, "B", 0), color=Z_COLOURS[z], lw=1.2, zorder=3,
                label=f"$z={z}$:  {sim.PZ_BASE[z]:.2f} → {sim.PZ_SHIFT[z]:.2f}")
    ax.set_xlim(0, DMAX)
    ax.set_ylim(0, 150)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Duration $d$ (s)", labelpad=1)
    ax.set_ylabel("Mean positive reward (s)", labelpad=2)
    ax.legend(loc="upper left", fontsize=7.0, handlelength=2.2,
              title="Class share, in force → candidate", title_fontsize=7.0, alignment="left")
    st.tidy(ax, grid_axis="y")


def panel_miss(ax):
    d = np.arange(2.0, DMAX + 1, 2.0)
    ax.plot(d, sim._miss_probability(d, np.zeros(d.shape, dtype=int), "I"), color=BASELINE,
            lw=1.2, zorder=3, label=f"{L['I']}, every class")
    for z in range(4):
        ax.plot(d, sim._miss_probability(d, np.full(d.shape, z, dtype=int), "J"),
                color=Z_COLOURS[z], lw=1.2, ls=(0, (2.4, 1.3)), zorder=3, label=f"{L['J']}, $z={z}$")
    ax.set_xlim(0, DMAX)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_xlabel("Duration $d$ (s)", labelpad=1)
    ax.set_ylabel("Probability of no annunciation", labelpad=2)
    ax.legend(loc="upper right", fontsize=7.0, handlelength=2.2, ncol=1)
    st.tidy(ax, grid_axis="y")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    rng = np.random.default_rng(SEED)
    fig = figpage.page_figure(plt, 4.6)
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.985, top=0.945, bottom=0.085,
                          wspace=0.32, hspace=0.50)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_durations(ax_a, rng)
    st.panel(ax_a, "A", "Excursion durations")
    ax_b = fig.add_subplot(gs[0, 1])
    panel_reward_common(ax_b)
    st.panel(ax_b, "B", "Reward laws, all classes")
    ax_c = fig.add_subplot(gs[1, 0])
    panel_reward_by_class(ax_c)
    st.panel(ax_c, "C", f"Reward by overshoot class ({L['B']}, {L['C']})")
    ax_d = fig.add_subplot(gs[1, 1])
    panel_miss(ax_d)
    st.panel(ax_d, "D", f"Non-annunciation ({L['I']}, {L['J']})")
    figpage.save_fixed(fig, "FigureS6_simulation_laws", OUT, plt)


if __name__ == "__main__":
    main()
