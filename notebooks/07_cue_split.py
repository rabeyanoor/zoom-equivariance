# %% [markdown]
# # 07 - Which part of a zoom moves the depth: lost context or magnification?
#
# A digital zoom removes the context around the crop *and* makes the crop s times larger.
# `audit.cue_rows` separates the two with images built from each photo:
#
# - **paste**: the 2x crop at its own place and size, the rest filled with the mean colour
#   (camera unchanged, so K stays; only context is gone);
# - **shrink**: the whole image 2x smaller on a plain canvas (all context kept, smaller).
#
# Comparisons on the same pixels: context = paste vs original, magnify = zoom vs paste,
# zoom = zoom vs original, shrink = shrink vs original. An apparent-size reader gives
# 1, 0.5, 0.5 and 2; a camera-aware model gives 1 everywhere.
# Runs: Kaggle CPU kernels `zoom-equivariance-audit-cues{,2,3}` (GPU quota was spent).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import analysis as A
import audit
import datasets
import plotstyle as ps
import zoom

ps.use()
TAB = ROOT / "outputs/tables"
MODELS = ["dav2-s", "zoedepth", "moge2-s", "unidepth2-s", "depthpro", "da3-metric-l"]
KINDS = ["context", "magnify", "zoom", "shrink"]

# %% [markdown]
# ## 1. What the model sees

# %%
sample = datasets.Loader("kitti")(datasets.sample_ids("kitti", 100)[3])
img = sample["img"]
h, w = img.shape[:2]
box = zoom.zoom_box(h, w, 2.0)
small, _ = audit.shrink_image(img, 2.0)
views = [("original", img), ("zoom 2$\\times$", zoom.zoom_image(img, box)),
         ("paste: context removed", audit.paste_crop(img, box)), ("shrink: 2$\\times$ smaller", small)]
fig, axes = plt.subplots(2, 2, figsize=(3.4, 1.35))
for ax, (title, im) in zip(axes.ravel(), views):
    ax.imshow(im)
    ax.set_title(title, fontsize=6, pad=1.5)
    ax.axis("off")
fig.subplots_adjust(wspace=0.03, hspace=0.25, left=0, right=1, top=0.9, bottom=0)
ps.save(fig, "fig7_cue_views")

# %% [markdown]
# ## 2. Depth ratio per intervention

# %%
files = sorted(A.RESULTS.glob("*__cues.csv"))
cues = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(cues.groupby("model").name.nunique())
ratio = cues.groupby(["model", "kind"]).log_ratio_self.median().unstack()[KINDS].pipe(np.exp)
ratio = ratio.reindex([m for m in MODELS if m in ratio.index])
by_ds = cues.groupby(["model", "dataset", "kind"]).log_ratio_self.median().unstack()[KINDS].pipe(np.exp)
ratio.round(3).to_csv(TAB / "cue_split.csv")
by_ds.round(3).to_csv(TAB / "cue_split_by_dataset.csv")
ratio.round(2)

# %%
# per image: how much of the zoom's log shrink is magnification (1 = all of it)
w_ = cues.pivot_table(index=["model", "dataset", "name"], columns="kind", values="log_ratio_self")
share = (w_.magnify / w_.zoom).where(w_.zoom.abs() > np.log(1.1))
share.groupby(level="model").median().round(2)

# %%
fig, ax = plt.subplots(figsize=(3.4, 2.0))
x = np.arange(len(KINDS))
width = 0.8 / len(ratio)
for k, (m, r) in enumerate(ratio.iterrows()):
    ax.bar(x + (k - (len(ratio) - 1) / 2) * width, r.values, width, color=ps.line(m)["color"], label=ps.LABEL[m])
ax.scatter(x, [1, 0.5, 0.5, 2.0], marker="_", s=300, color="k", zorder=5, label="apparent size")
ax.axhline(1, color="k", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_yticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
ax.minorticks_off()
ax.set_xticks(x, ["context", "magnify", "zoom", "shrink"])
ax.set_ylabel("depth ratio (same pixels)")
ax.legend(fontsize=5, ncol=2, frameon=False)
fig.tight_layout()
ps.save(fig, "fig8_cue_split")

# %% [markdown]
# ## Reading
# Filled in from the tables above once all three cue kernels are in (see the paper, Sec. 5).
