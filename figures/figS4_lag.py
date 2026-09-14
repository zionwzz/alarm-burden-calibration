#!/usr/bin/env python3
r"""Time from the numeric crossing to the recorded annunciation, from the onset-lead histograms.

`comparison/c1_onset_hist_*.json` in the working tree are the onset-lead histograms the
crossing scan wrote on the whole fitting set at the naive setting, one file per stripe, at
two-second resolution from -120 to 120 seconds with the tails beyond counted at the ends. The
stripes are summed here exactly as `tables/table_crossing.py` sums them for the table, and the
median is read off the histogram the same way (the lower edge of the bin that reaches it). The
persistence of the selected delay model is read from `delay_model/A2_SELECTED_MODEL.json`.
"""
import glob
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figpage  # noqa: E402
from alarmreplay import figstyle_lrc as st  # noqa: E402
from alarmreplay import paths  # noqa: E402

AGG = paths.WORK
OUT = paths.FIGOUT

CHANNELS = [("ecgResp-numLimit-respRate-low", "Respiration rate, low"),
            ("spO2-numLimit-satO2-low", "SpO₂, low"),
            ("ecg-numLimit-heartRate-high", "Heart rate, high"),
            ("ecg-numLimit-heartRate-low", "Heart rate, low")]
XMIN, XMAX = -10, 40          # seconds shown; the shares beyond are annotated


def load():
    H, lo, hi, n, meta = {}, {}, {}, {}, None
    for fp in sorted(glob.glob(os.path.join(AGG, "comparison", "c1_onset_hist_*.json"))):
        d = json.load(open(fp))
        if "hist" not in d:
            continue
        if meta is not None:
            for k in ("lo", "hi", "bin_width_s"):
                if d[k] != meta[k]:
                    raise SystemExit(f"{fp}: histogram geometry differs between stripes")
        if d["admissions_done"] < d["admissions_total"]:
            raise SystemExit(f"{fp} is incomplete")
        meta = d
        for ch, h in d["hist"].items():
            H[ch] = np.array(h, dtype=float) + (H[ch] if ch in H else 0.0)
            lo[ch] = lo.get(ch, 0) + d["lo_tail"][ch]
            hi[ch] = hi.get(ch, 0) + d["hi_tail"][ch]
            n[ch] = n.get(ch, 0) + d["n_matched"][ch]
    sel = json.load(open(os.path.join(AGG, "delay_model", "A2_SELECTED_MODEL.json")))
    return H, lo, hi, n, meta, sel


def median(h, lo_tail, hi_tail, base, bw):
    """The lower edge of the bin that reaches the median, as the table reads it."""
    tot = h.sum() + lo_tail + hi_tail
    acc = lo_tail
    if acc >= 0.5 * tot:
        return base
    for i, c in enumerate(h):
        acc += c
        if acc >= 0.5 * tot:
            return base + bw * i
    return base + bw * len(h)


def panel(ax, h, lo_tail, hi_tail, base, bw, tau, label, show_x, show_y):
    edges = base + bw * np.arange(len(h) + 1)
    tot = h.sum() + lo_tail + hi_tail
    share = h / tot
    keep = (edges[:-1] >= XMIN) & (edges[:-1] < XMAX)
    ax.bar(edges[:-1][keep], share[keep], width=bw, align="edge", color="#8FA3BF",
           edgecolor="white", linewidth=0.4, zorder=2)
    med = median(h, lo_tail, hi_tail, base, bw)
    ax.axvline(med + bw / 2, color=st.RECORDED, lw=1.0, zorder=3)
    ax.axvline(tau, color=st.LRC_DZ, lw=1.0, ls=(0, (2.4, 1.3)), zorder=3)
    beyond = (share[edges[:-1] >= XMAX].sum() + hi_tail / tot)
    before = (share[edges[:-1] < XMIN].sum() + lo_tail / tot)
    ax.text(0.98, 0.95, f"median {med:g} s\nbeyond {XMAX} s: {100*beyond:.1f}%\nbefore −10 s: {100*before:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=st.INK,
            linespacing=1.15)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xticks([-10, 0, 10, 20, 30, 40])
    ax.set_xticklabels(["−10", "0", "10", "20", "30", "40"] if show_x else [])
    if not show_y:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("Crossing to annunciation (s)", labelpad=1)
    if show_y:
        ax.set_ylabel("Share of matched episodes", labelpad=2)
    ax.set_title(label, pad=4)
    st.tidy(ax, grid_axis="y")


def main():
    st.use_style()
    figpage.use_page_type(mpl)
    H, lo, hi, n, meta, sel = load()
    base, bw = meta["lo"], meta["bin_width_s"]
    fig = figpage.page_figure(plt, 3.7)
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.985, top=0.930, bottom=0.165,
                          wspace=0.12, hspace=0.45)
    for k, (ch, label) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        panel(ax, H[ch], lo[ch], hi[ch], base, bw, sel[ch]["tau"], label,
              show_x=(k // 2 == 1), show_y=(k % 2 == 0))
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=st.RECORDED, lw=1.0, label="Median interval"),
               Line2D([0], [0], color=st.LRC_DZ, lw=1.0, ls=(0, (2.4, 1.3)),
                      label="Persistence τ of the selected delay model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0),
               fontsize=7.0, handlelength=2.2, columnspacing=1.8)
    figpage.save_fixed(fig, "FigureS4_lag", OUT, plt)


if __name__ == "__main__":
    main()
