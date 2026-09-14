#!/usr/bin/env python3
"""The page geometry every figure in this paper is built to.

One constant governs the whole figure set: the width, in inches, at which a figure is placed in the
journal's text column. Type in a figure is set in points, but those are points of the *saved* file;
a figure saved wider than the column it is placed in has every glyph in it reduced by the ratio of
the two widths. So the rule this module exists to enforce is:

    the saved width of a figure equals the width it is placed at,

after which a size set in a script is the size on the printed page, and the production floor can be
checked by reading the numbers in the script rather than by measuring the file.

The rule has one practical consequence. A tight bounding box makes the saved width a function of
how far the labels happen to stick out, which is exactly the coupling this removes, so every figure
here is saved with the bounding box switched off: the figure size is fixed, the margins are set
inside it, and nothing is drawn outside the figure rectangle. `save_fixed` below does that, and
verifies it.

This lives under `figures/` rather than in the shared style modules because it is a property of the
page the figures are placed on, not of their visual grammar: the two style modules describe how the
figures look, this one describes how large they are printed.
"""
import os

# ----------------------------------------------------------------- the two numbers that govern all
PLACEMENT_WIDTH_IN = 5.2   # the journal text column each figure is included at
MIN_TYPE_PT = 7.0          # the production floor: no glyph on the page may be smaller

# Mathtext shrinks a sub- or superscript to a fraction of its base size, and shrinks a nested one
# again. At Matplotlib's default of 0.7 a 7 pt label carries a 4.9 pt subscript: below the floor while the label
# containing it sits on it. Raised to the value below, a subscript on an 8 pt label prints at 7.0 pt,
# so any label carrying mathtext is set at 8 pt or more. One level of sub- or superscript is all this
# figure set uses.
MATHTEXT_SHRINK = 0.88
MATHTEXT_BASE_PT = MIN_TYPE_PT / MATHTEXT_SHRINK    # 7.96: the smallest label that may carry math

# A chemical subscript is not mathematics and does not need mathtext: set as its own character it
# keeps the label's own size, and it matches how the supplementary figures already write the same
# channel name. `typed_subscripts` rewrites a label mapping in place of `$_n$`.
_TYPED_SUBSCRIPT = {f"$_{d}$": chr(0x2080 + d) for d in range(10)}

# Every figure PDF the manuscript includes, in the order the paper presents them.
FIGURE_PDFS = [
    "Figure1_framework",
    "Figure2_simulation",
    "Figure3_holdout",
    "Figure4_reward",
    "Figure5_policy",
    "FigureS1_study_flow",
    "FigureS2_fit_grid",
    "FigureS3_conventions",
    "FigureS4_lag",
    "FigureS5_grid",
    "FigureS6_simulation_laws",
    "FigureS7_simulation_replicates",
]

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def figure_dir():
    """Where the figure scripts write, honouring the same environment variable they do."""
    from alarmreplay.paths import FIGOUT
    return FIGOUT


def page_figure(plt, height_in, width_in=PLACEMENT_WIDTH_IN):
    """A figure exactly as wide as the column it will be placed in.

    Only the height is a free parameter. Give the figure its margins with `add_axes`,
    `add_gridspec(left=..., right=...)` or `subplots_adjust`, not with a tight bounding box.
    """
    return plt.figure(figsize=(width_in, height_in))


def use_page_type(mpl, base=None):
    """Type defaults that hold at the placement width, applied over a style module's own.

    Call after the style module's own defaults. It switches the tight bounding box off, so the
    saved width is the figure width; sets the floor as the smallest default in the sheet; and
    raises the mathtext sub- and superscript fraction, which is otherwise the one size in a figure
    that no rc parameter controls.
    """
    floor = MIN_TYPE_PT if base is None else base
    mpl.rcParams.update({
        "savefig.bbox": None,          # the saved width is the figure width, and nothing else
        "savefig.pad_inches": 0.0,
        "font.size": floor + 0.6,
        "axes.labelsize": floor + 0.6,
        "axes.titlesize": floor + 1.0,
        "xtick.labelsize": floor,
        "ytick.labelsize": floor,
        "legend.fontsize": floor,
        "figure.titlesize": floor + 1.4,
    })
    _raise_mathtext_shrink()


def _raise_mathtext_shrink():
    """Make a mathtext subscript large enough to survive printing.

    Mathtext sizes a sub- or superscript at a fixed fraction of its base, held in a module constant
    rather than in an rc parameter, and the default fraction puts a subscript on a 7 pt label at
    4.9 pt. The constant is read at parse time by the node classes, so setting it here applies to
    every figure drawn afterwards. Guarded, so a release that renames it costs a slightly small
    subscript and not a failed build.
    """
    try:
        from matplotlib import _mathtext
    except Exception:
        return
    if not hasattr(_mathtext, "SHRINK_FACTOR"):
        return
    _mathtext.SHRINK_FACTOR = MATHTEXT_SHRINK
    if hasattr(_mathtext, "GROW_FACTOR"):
        _mathtext.GROW_FACTOR = 1.0 / MATHTEXT_SHRINK


def typed_subscripts(mapping):
    """A copy of a label mapping with `$_n$` written as the subscript character itself.

    The channel names are the only labels in the set that carry a chemical subscript, and they are
    also among the smallest type in the figures, so shrinking them as mathematics is what puts a
    figure below the floor. Written as a character the subscript takes the label's own size.
    """
    out = {}
    for key, label in mapping.items():
        for math, typed in _TYPED_SUBSCRIPT.items():
            label = label.replace(math, typed)
        out[key] = label
    return out


def save_fixed(fig, stem, outdir, plt, png=True, png_dpi=400):
    """Write the figure at exactly its own size, and check that it came out that way.

    Returns the path of the PDF. Raises if the saved width differs from the figure width by more
    than a rounding error, which is the failure this module exists to prevent and which is silent
    everywhere else.
    """
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.0)
    if png:
        fig.savefig(os.path.join(outdir, stem + ".png"), bbox_inches=None, pad_inches=0.0,
                    dpi=png_dpi)
    plt.close(fig)

    want = fig.get_figwidth()
    got = _pdf_width_in(pdf)
    if got is not None and abs(got - want) > 0.01:
        raise RuntimeError(f"{stem}.pdf saved {got:.3f} in wide, not the {want:.3f} in it was "
                           f"drawn at; a bounding box is still being applied")
    print(f"  wrote {stem}.pdf ({want:.2f} in wide)")
    return pdf


def _pdf_width_in(path):
    """The media-box width of a one-page PDF, in inches; None if it cannot be read."""
    import re
    try:
        with open(path, "rb") as f:
            raw = f.read()
        m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", raw)
        return (float(m.group(3)) - float(m.group(1))) / 72.0 if m else None
    except Exception:
        return None
