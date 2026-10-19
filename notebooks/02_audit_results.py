# %% [markdown]
# # 02 - The zoom audit: do metric depth models respect a digital zoom?
#
# Results come from `scripts/run_audit.py`, run on a Kaggle GPU (kernel
# `zoom-equivariance-audit`), 200 images per set, centre zooms s = 1, 1.25, 1.5, 2, 3.
#
# Per image we fit two exponents (see `src/analysis.py`):
# - **alpha**: depth of the same pixels scales like s^-alpha. Ideal 0. A model that only
#   reads apparent size gives 1 (twice as big = twice as close).
# - **beta**: predicted focal scales like s^beta. Ideal 1.
#
# Control: the *blur* row degrades the image like a 2x zoom without changing the field
# of view. If depth moved as much under blur, interpolation would be the explanation.

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
TAB.mkdir(parents=True, exist_ok=True)
MAIN = ["dav2-s", "dav2-b", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s", "metric3d-s-staleK"]

df = A.load()
df = df[df.model.isin(MAIN)]
MAIN = [m for m in MAIN if m in set(df.model)]  # models whose runs are in
print(df.groupby(["model", "dataset"]).name.nunique().unstack())

# %%
pi = A.per_image(df)
summ = A.summary(pi).merge(A.blur_control(df).reset_index(), on=["model", "dataset"], how="left")
summ.to_csv(TAB / "audit_by_model_dataset.csv", index=False)
cols = ["model", "dataset", "n", "alpha", "alpha_lo", "alpha_hi", "beta", "die2", "blur2", "absrel1", "absrel2", "absrel3"]
summ[cols].round(3)

# %% [markdown]
# ## Table 1: pooled over the four sets

# %%
pooled = A.summary(pi, by=("model",)).set_index("model").loc[MAIN]
blur = A.blur_control(df).groupby("model").mean()
pooled["blur2"] = blur
pooled["absrel_ratio_3"] = pooled.absrel3 / pooled.absrel1
t1 = pooled[["alpha", "alpha_lo", "alpha_hi", "beta", "die2", "blur2", "absrel1", "absrel2", "absrel3", "absrel_ratio_3"]]
t1.index = [A.PRETTY[m] for m in t1.index]
t1.round(3).to_csv(TAB / "table1_pooled.csv")
t1.round(3)

# %% [markdown]
# ## Figure 2: how depth and focal move with zoom

# %%
z = df[df.kind == "zoom"].copy()
z["log_fx"] = np.log(z.fx / z.fx_base)
curve = z.groupby(["model", "zoom"]).agg(depth=("log_ratio_self", "median"), focal=("log_fx", "median"),
                                         s=("s_true", "median")).reset_index()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
ss = np.linspace(1, 3, 50)
a1.plot(ss, np.ones_like(ss), "k--", lw=1, label="ideal (depth fixed)")
a1.plot(ss, 1 / ss, "k:", lw=1, label="apparent size only (1/s)")
a2.plot(ss, ss, "k--", lw=1, label="ideal (f x s)")
a2.plot(ss, np.ones_like(ss), "k:", lw=1, label="ignores zoom")
for m in MAIN:
    c = curve[curve.model == m].sort_values("zoom")
    a1.plot(c.s, np.exp(c.depth), "o-", ms=4, label=A.PRETTY[m])
    if c.focal.notna().all():
        a2.plot(c.s, np.exp(c.focal), "o-", ms=4, label=A.PRETTY[m])
a1.set(xlabel="zoom factor s", ylabel="depth(zoomed) / depth(original), same pixels", title="Depth under digital zoom")
a2.set(xlabel="zoom factor s", ylabel="predicted f(zoomed) / f(original)", title="Predicted focal under digital zoom")
for a in (a1, a2):
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "02_depth_and_focal_vs_zoom.png", dpi=110)
plt.show()

# %% [markdown]
# ## Per-dataset alpha (with 95% bootstrap CI)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
width = 0.8 / len(MAIN)
for k, m in enumerate(MAIN):
    s = summ[summ.model == m].set_index("dataset").reindex(A.DATASETS)
    x = np.arange(len(A.DATASETS)) + k * width
    ax.bar(x, s.alpha, width, yerr=[s.alpha - s.alpha_lo, s.alpha_hi - s.alpha], label=A.PRETTY[m], capsize=2)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xticks(np.arange(len(A.DATASETS)) + 0.4 - width / 2, [A.PRETTY_DS[d] for d in A.DATASETS])
ax.set_ylabel("alpha (0 = ideal, 1 = apparent size)")
ax.legend(fontsize=7, ncol=4)
plt.tight_layout()
plt.savefig(FIG / "02_alpha_per_dataset.png", dpi=110)
plt.show()

# %% [markdown]
# ## What does a 2x zoom cost in accuracy?

# %%
acc = summ.pivot(index="model", columns="dataset", values=["absrel1", "absrel2", "absrel3"]).loc[MAIN]
acc.round(3)
