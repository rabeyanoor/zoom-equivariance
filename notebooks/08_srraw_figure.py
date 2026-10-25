# %% [markdown]
# # 08 - Real optical zoom (SR-RAW): one static scene shot at several focal lengths
#
# Maps come from `scripts/srraw_maps.py` (kernel "tpu figs"). The sequence is not hand-picked: it is the one whose
# DA-V2-S alpha under optical zoom is the median over all SR-RAW test sequences. The camera does not move, so the
# true depth of every point is the same in every column. The number in each panel is the median depth over the
# pixels inside the longest shot's field of view, i.e. the same physical points in every column.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import analysis as A
import plotstyle as ps

ps.use()
RES = ROOT / "outputs/results"
MODELS = [("dav2-s", "DA-V2-S"), ("dav2-s-eftb", "EFT+B (ours)"), ("depthpro", "Depth Pro"),
          ("depthpro-eftf", "EFT-F (ours)"), ("moge2-s", "MoGe-2-S")]

# %% the sequence with DA-V2-S's median optical alpha (the rule the kernel was built with)
sr = pd.read_csv(RES / "dav2-s__srraw.csv", dtype={"seq": str}).assign(model="dav2-s")  # keep leading zeros
a = A.srraw_alpha(sr)
a = a[a.view == "optical"].dropna(subset=["alpha"]).sort_values("alpha").reset_index(drop=True)
pick = a.iloc[(len(a) - 1) // 2]
print(f"median-alpha sequence: {pick.seq} (alpha {pick.alpha:.2f}, median over {len(a)} sequences "
      f"{a.alpha.median():.2f})")
SEQ = pick.seq


# %%
def core(shape, s_from, s_to):
    """Pixels of the s_to field of view, in a shot zoomed by s_from relative to the widest one."""
    h, w = shape
    k = s_to / s_from
    ch, cw = int(round(h / k)), int(round(w / k))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return slice(y0, y0 + ch), slice(x0, x0 + cw)


def median_core(d, s, s_last):
    v = d[core(d.shape, s, s_last)].astype(np.float32)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else np.nan


maps = np.load(RES / f"srraw_maps_{SEQ}.npz")
focals = list(maps["focals"])
s_of = {f: f / focals[0] for f in focals}
s_last = s_of[focals[-1]]
models = [(m, lab) for m, lab in MODELS if f"{m}|{focals[0]:g}" in maps]
for m, lab in models:
    print(f"{lab:16s}", "  ".join(f"{f:g} mm: {median_core(maps[f'{m}|{f:g}'], s_of[f], s_last):6.1f} m" for f in focals))

# %%
img0 = maps[f"img|{focals[0]:g}"]
h, w = img0.shape[:2]
n_rows, n_cols = 1 + len(models), len(focals)
width = 7.0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, width / n_cols * h / w * n_rows * 1.12),
                         gridspec_kw={"hspace": 0.08, "wspace": 0.02})
vals = np.concatenate([maps[f"{m}|{f:g}"].astype(np.float32).ravel() for m in ("dav2-s", "depthpro") for f in focals])
vmin, vmax = np.nanpercentile(vals[vals > 0], [2, 98])
for k, f in enumerate(focals):
    axes[0, k].imshow(maps[f"img|{f:g}"])
    axes[0, k].set_title(f"{f:g} mm ({s_of[f]:.2f}×)", fontsize=8, pad=2)
    for j, (m, lab) in enumerate(models):
        d = maps[f"{m}|{f:g}"].astype(np.float32)
        ax = axes[1 + j, k]
        ax.imshow(d, cmap="turbo_r", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.text(0.015, 0.8, f"{median_core(d, s_of[f], s_last):.1f} m", transform=ax.transAxes, color="w",
                fontsize=6.5, bbox=dict(facecolor="k", alpha=0.55, pad=1, lw=0))
        if k == 0:
            ax.set_ylabel(lab, fontsize=7)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
ps.save(fig, "figS_srraw")
