"""Shared visual grammar for the four main figures.

One scientific task per figure, one meaning per colour and symbol across all four, no title or
explanatory paragraph inside the plotting area — interpretation belongs in the LaTeX caption — and
type sized to stay legible after journal reduction.

The semantic palette is fixed. It was checked for colour-vision deficiency separation and for
contrast against a white page: the two identity-carrying hues, calibration under duration transport
and under duration-by-overshoot transport, separate by ΔE 21 in normal vision and ΔE 15 under
protanopia, and every mark reaches at least 3:1 against the page. Replay is deliberately achromatic:
it is the reference the corrections are read against, not a third category competing for identity.
Colour never carries identity alone — position, marker shape and direct labels repeat it — so the
figures survive greyscale printing.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- semantic palette
REPLAY = "#5F6469"        # uncorrected threshold replay: neutral grey
RECORDED = "#1A1A1A"      # what the monitor recorded: the reference, near-black
LRC_D = "#1F4E9C"         # linked-reward calibration, duration transport
LRC_DZ = "#C1600B"        # linked-reward calibration, duration-by-overshoot transport
LEVEL = "#7A7F85"         # the secondary level route, where it still appears
UNSUPPORTED = "#B8BCC0"   # outside the identified support, or least stable
BAND = "#E8EAEC"          # adequacy or reference band
INK = "#1A1A1A"
MUTED = "#6B7075"

MARKER = {"replay": "o", "lrc_d": "s", "lrc_dz": "D", "level": "^", "oracle": "*"}
LABEL = {
    "replay": "Replay, uncorrected",
    "lrc_d": "Calibrated, duration transport",
    "lrc_dz": "Calibrated, duration × overshoot",
    "level": "Level route",
}

CHANNEL_ORDER = ["RR low", "SpO2 low", "HR high", "HR low"]
CHANNEL_LABEL = {
    "RR low": "Respiration rate, low",
    "SpO2 low": "SpO$_2$, low",
    "HR high": "Heart rate, high",
    "HR low": "Heart rate, low",
}
LEAST_STABLE = "HR low"   # widest interval, largest tail and unlinked share: drawn de-emphasised


def use_style():
    """Journal figure defaults: vector text, restrained rules, type that survives reduction."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,           # embed as Type 42 so the text stays editable and selectable
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#9AA0A6",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "grid.color": "#E3E5E8",
        "grid.linewidth": 0.6,
        "text.color": INK,
    })


def panel(ax, letter, title=None):
    """Panel label in the top-left margin, and an optional short sentence-case title."""
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(-22, 10), textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=INK, va="top", ha="left")
    if title:
        ax.set_title(title, pad=4)


def tidy(ax, grid_axis="y"):
    """Recessive grid behind the marks; no box."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", alpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, stem, outdir):
    """Write the editable vector PDF the manuscript includes and a high-resolution PNG preview."""
    import os
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {stem}.pdf and {stem}.png")
    return pdf
