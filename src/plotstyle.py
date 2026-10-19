"""One look for every figure in the paper.

    import plotstyle as ps
    ps.use()
    ax.plot(x, y, **ps.line("moge2-l"))
    ps.save(fig, "02_depth_and_focal_vs_zoom")      # PNG for the notebook, PDF for the paper

Each model family has one hue (colour-blind safe, Okabe-Ito); within a family, small models
are lighter and large models darker, so the model ladder reads at a glance.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
PAPER_FIG = ROOT / "paper" / "fig"

# Okabe-Ito
FAMILY_COLOUR = {
    "dav2": "#E69F00",       # orange
    "zoedepth": "#CC79A7",   # pink
    "depthpro": "#D55E00",   # vermillion
    "moge2": "#0072B2",      # blue
    "unidepth2": "#009E73",  # green
    "unik3d": "#56B4E9",     # sky blue
    "metric3d": "#000000",   # black
    "da3": "#8E44AD",        # purple
    "vggt": "#7F7F7F",       # grey
}
SIZE_SHADE = {"s": 0.55, "b": 0.8, "l": 1.0}

LABEL = {
    "dav2-s": "DA-V2-S", "dav2-b": "DA-V2-B", "dav2-l": "DA-V2-L", "zoedepth": "ZoeDepth",
    "depthpro": "Depth Pro", "moge2-s": "MoGe-2-S", "moge2-b": "MoGe-2-B", "moge2-l": "MoGe-2-L",
    "unidepth2-s": "UniDepthV2-S", "unidepth2-b": "UniDepthV2-B", "unidepth2-l": "UniDepthV2-L",
    "unik3d-l": "UniK3D-L", "metric3d-s": "Metric3D v2-S", "metric3d-l": "Metric3D v2-L",
    "metric3d-s-staleK": "Metric3D v2-S (stale K)", "da3-nested": "DA3-Nested",
    "da3-metric-l": "DA3-Metric-L", "da3-l": "DA3-L", "vggt": "VGGT-1B",
    "dav2-s-fixed": "DA-V2-S (frozen)", "dav2-s-anchor": "anchor-only FT", "dav2-s-eft": "EFT",
    "dav2-s-eftb": "EFT+B (ours)",
}


def family(model):
    return model.split("-")[0]


def colour(model):
    if model.startswith("dav2-s-"):  # fine-tuned variants of DA-V2-S
        return {"dav2-s-fixed": "#999999", "dav2-s-anchor": "#56B4E9", "dav2-s-eft": "#E69F00",
                "dav2-s-eftb": "#D55E00"}[model]
    base = to_rgb(FAMILY_COLOUR.get(family(model), "#333333"))
    size = model.split("-")[1][:1] if "-" in model else "l"
    shade = SIZE_SHADE.get(size, 1.0)
    # blend towards white for smaller models
    return tuple(1 - shade * (1 - c) for c in base)


def line(model, **kw):
    style = dict(color=colour(model), label=LABEL.get(model, model), lw=1.8, marker="o", ms=4)
    if model.endswith("staleK"):
        style.update(ls="--", marker="x")
    style.update(kw)
    return style


def use():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # editable text in the PDF
        "font.family": "DejaVu Sans",
    })


def save(fig, name, paper=True):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    if paper:
        PAPER_FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show()
