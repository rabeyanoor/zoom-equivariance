# %% [markdown]
# # 04 - Off-centre zooms and the "stale K" case
#
# Two practical cases a reviewer will ask about:
#
# 1. **Off-centre crops.** A phone crop or a detection crop is rarely centred. The law is
#    the same (focal x s, principal point shifts), but the crop now contains e.g. mostly road
#    or mostly ceiling. Does alpha depend on where we zoom? (50 images per set, KITTI and
#    NYUv2, centre + four quadrants.)
# 2. **Stale intrinsics.** Metric3D v2 takes K as input and is equivariant by design when K
#    is right. If the zoom happens after capture and the metadata is not updated (common for
#    screenshots, crops, social media), the model receives the old K. How bad is that?

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import analysis as A

FIG, TAB = ROOT / "outputs/figures", ROOT / "outputs/tables"

off = A.load("*__offc.csv")
pi = A.per_image(off)
pi["where"] = pi.apply(lambda r: "centre" if (r.cx, r.cy) == (0.5, 0.5) else
                       ("top" if r.cy < 0.5 else "bottom") + "-" + ("left" if r.cx < 0.5 else "right"), axis=1)
tab = pi.groupby(["model", "dataset", "where"]).alpha.median().unstack("where")
tab.round(3).to_csv(TAB / "offcentre_alpha.csv")
tab.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order = ["centre", "top-left", "top-right", "bottom-left", "bottom-right"]
for ax, ds in zip(axes, ["kitti", "nyu"]):
    t = tab.xs(ds, level="dataset")[order]
    for m, r in t.iterrows():
        ax.plot(order, r.values, "o-", label=A.PRETTY[m])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=A.PRETTY_DS[ds], ylabel="median alpha")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "04_offcentre_alpha.png", dpi=110)
plt.show()

# %% [markdown]
# ## Stale K for Metric3D v2

# %%
main = A.load()
m3 = main[main.model.isin(["metric3d-s", "metric3d-s-staleK"])]
s = A.summary(A.per_image(m3))
stale = s.pivot(index="dataset", columns="model", values=["alpha", "absrel1", "absrel2", "absrel3"])
stale.round(3).to_csv(TAB / "stale_K.csv")
stale.round(3)
