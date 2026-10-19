# %% [markdown]
# # 05 - Equivariance Fine-Tuning (EFT): does the fix work?
#
# Trained on a Kaggle GPU (kernel `zoom-equivariance-eft`) from **unlabeled** images:
# - outdoor head: frames of 7 KITTI raw drives, none of them a drive of the KITTI val
#   selection (checked in the kernel);
# - indoor head: RGB of the 795 official NYUv2 train images.
#
# Frozen and EFT models are evaluated with identical preprocessing (input height 392),
# so every difference comes from the weights. DIODE was never seen in training: it shows
# whether the model learned "how zoom looks" or just memorised KITTI / NYU.

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
EFT = ROOT / "outputs" / "eft"

# %% [markdown]
# ## Training curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, dom in zip(axes, ("outdoor", "indoor")):
    log = pd.read_csv(EFT / f"train_log_{dom}.csv")
    ax.plot(log.step, log.l_zoom.rolling(50).mean(), label="zoom loss")
    ax.plot(log.step, log.l_anchor.rolling(50).mean(), label="anchor loss")
    ax.set(title=f"EFT {dom}", xlabel="step", ylabel="L1 in log depth")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "05_eft_training.png", dpi=110)
plt.show()

# %% [markdown]
# ## Table 2: frozen vs EFT

# %%
df = A.load()
df = df[df.model.isin(["dav2-s-fixed", "dav2-s-eft"])]
s = A.summary(A.per_image(df))
t2 = s.pivot(index="dataset", columns="model", values=["alpha", "absrel1", "absrel2", "absrel3", "d1_3"]).reindex(A.DATASETS)
t2.round(3).to_csv(TAB / "table2_eft.csv")
t2.round(3)

# %%
z = df[df.kind == "zoom"]
curve = z.groupby(["model", "dataset", "zoom"]).agg(depth=("log_ratio_self", "median"), absrel=("abs_rel", "mean"),
                                                    s=("s_true", "median")).reset_index()
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for k, ds in enumerate(A.DATASETS):
    for m, style in (("dav2-s-fixed", "o--"), ("dav2-s-eft", "o-")):
        c = curve[(curve.model == m) & (curve.dataset == ds)].sort_values("zoom")
        axes[0, k].plot(c.s, np.exp(c.depth), style, label=A.PRETTY[m])
        axes[1, k].plot(c.s, c.absrel, style, label=A.PRETTY[m])
    axes[0, k].axhline(1, color="k", lw=0.8)
    axes[0, k].set(title=A.PRETTY_DS[ds] + (" (unseen)" if ds.startswith("diode") else ""), ylabel="depth ratio vs 1x")
    axes[1, k].set(xlabel="zoom s", ylabel="AbsRel")
    for a in axes[:, k]:
        a.grid(alpha=0.3)
axes[0, 0].legend()
plt.tight_layout()
plt.savefig(FIG / "05_eft_vs_frozen.png", dpi=110)
plt.show()
