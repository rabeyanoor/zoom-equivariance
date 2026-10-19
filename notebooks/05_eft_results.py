# %% [markdown]
# # 05 - Fine-tuning on unlabeled images: does the fix work?
#
# Trained on Kaggle GPUs from **unlabeled** images only:
# - outdoor head: frames of 7 KITTI raw drives, none of them a drive of the KITTI val
#   selection (checked in the kernel);
# - indoor head: RGB of the 795 official NYUv2 train images.
#
# Four variants of DA-V2-S, all scored with identical preprocessing (input height 392):
# - **frozen** - the released model;
# - **blur-only** - same data, steps and flips, zoom loss off: what fine-tuning alone does
#   (an anchor-only run would be the frozen model: student == teacher gives zero gradient);
# - **EFT** - zoom self-consistency (in the spirit of Zhong et al., ICCV 2023);
# - **EFT+B** - EFT plus blur invariance, against the "blurry = zoomed" shortcut;
# - **EFT+B, lambda=3** - with a stronger anchor, to probe the cost on unzoomed images.
#
# DIODE is never used in training. Every comparison is paired (same images), tested with a
# Wilcoxon signed-rank test and Holm-corrected over all comparisons.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import analysis as A
import plotstyle as ps

ps.use()
TAB = ROOT / "outputs/tables"
EFT = ROOT / "outputs" / "eft"
# anchor-only is omitted: it is the frozen model (zero loss and gradient at the start)
FT = ["dav2-s-fixed", "dav2-s-blur", "dav2-s-eft", "dav2-s-eftb", "dav2-s-eftb3"]

df = A.load()
df = df[df.model.isin(FT)]
FT = [m for m in FT if m in set(df.model)]
pi = A.per_image(df)
print(pi.groupby(["model", "dataset"]).size().unstack())

# %% [markdown]
# ## Training curves

# %%
def train_log(tag, dom):
    # the first EFT run wrote train_log_<domain>.csv, later runs train_log_<tag>_<domain>.csv
    for f in (EFT / f"train_log_{tag}_{dom}.csv", EFT / f"train_log_{dom}.csv"):
        if f.exists():
            return pd.read_csv(f)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, dom in zip(axes, ("outdoor", "indoor")):
    for tag, m in (("eft", "dav2-s-eft"), ("eftb", "dav2-s-eftb"), ("eftb3", "dav2-s-eftb3"), ("blur", "dav2-s-blur")):
        log = train_log(tag, dom)
        if log is None:
            continue
        if tag != "blur":
            ax.plot(log.step, log.l_zoom.rolling(100).mean(), color=ps.colour(m), lw=1.4, label=f"{ps.LABEL[m]}: zoom loss")
        ax.plot(log.step, log.l_anchor.rolling(100).mean(), color=ps.colour(m), lw=1, ls="--", label=f"{ps.LABEL[m]}: anchor loss")
    ax.set(title=f"{dom} head", xlabel="step", ylabel="L1 in log depth")
axes[1].legend(fontsize=6)
fig.tight_layout()
ps.save(fig, "figS_eft_training")

# %% [markdown]
# ## Table 2: frozen vs anchor-only vs EFT vs EFT+B

# %%
s = A.summary(pi)
t2 = s.pivot(index="dataset", columns="model", values=["alpha", "absrel1", "absrel3"]).reindex(A.DATASETS)
blur = df[df.kind == "blur"].groupby(["model", "dataset"]).log_ratio_self.apply(
    lambda v: float(np.exp(np.median(np.abs(v))) - 1)).unstack("model")
t2.round(3).to_csv(TAB / "table2_eft.csv")
blur.round(3).to_csv(TAB / "eft_blur_control.csv")
display(t2.round(3))
display((100 * blur).round(1).rename(columns=ps.LABEL))

# %% [markdown]
# ## Paired tests (Wilcoxon signed-rank, Holm-corrected)
# For each dataset and each fine-tuned variant against the frozen model: AbsRel at 1x (does
# tuning hurt normal images?), AbsRel at 3x (does it help zoomed ones?), and |alpha|.

# %%
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


tests = []
wide = pi.assign(abs_alpha=pi.alpha.abs()).pivot_table(index=["dataset", "name"], columns="model",
                                                       values=["absrel1", "absrel3", "abs_alpha"])
for ds in A.DATASETS:
    for m in FT[1:]:
        for metric in ("absrel1", "absrel3", "abs_alpha"):
            pair = wide.loc[ds][[(metric, "dav2-s-fixed"), (metric, m)]].dropna()
            if len(pair) < 10:
                continue
            a, b = pair.iloc[:, 0], pair.iloc[:, 1]
            tests.append(dict(dataset=ds, model=m, metric=metric, n=len(pair), frozen=a.mean(), tuned=b.mean(),
                              change=b.mean() - a.mean(), p=wilcoxon(a, b).pvalue))
tests = pd.DataFrame(tests)
tests["p_holm"] = holm(tests.p)
tests["significant"] = tests.p_holm < 0.05
tests.round(4).to_csv(TAB / "eft_paired_tests.csv", index=False)
tests.round(4)

# %% [markdown]
# ## Figure 5: depth ratio and accuracy vs zoom

# %%
z = df[df.kind == "zoom"]
curve = z.groupby(["model", "dataset", "zoom"]).agg(depth=("log_ratio_self", "median"), absrel=("abs_rel", "mean"),
                                                    s=("s_true", "median")).reset_index()
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
for k, ds in enumerate(A.DATASETS):
    for m in FT:
        c = curve[(curve.model == m) & (curve.dataset == ds)].sort_values("zoom")
        axes[0, k].plot(c.s, np.exp(c.depth), color=ps.colour(m), lw=1.4, marker="o", ms=3, label=ps.LABEL[m])
        axes[1, k].plot(c.s, c.absrel, color=ps.colour(m), lw=1.4, marker="o", ms=3)
    axes[0, k].axhline(1, color="k", lw=0.7)
    axes[0, k].set_title(A.PRETTY_DS[ds] + (" (unseen)" if ds.startswith("diode") else ""), fontsize=8)
    axes[1, k].set_xlabel("zoom $s$")
axes[0, 0].set_ylabel(r"$\hat D_s/\hat D_1$")
axes[1, 0].set_ylabel("AbsRel")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
ps.save(fig, "fig5_finetuning")
