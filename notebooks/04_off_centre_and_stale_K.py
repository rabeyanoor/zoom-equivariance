# %% [markdown]
# # 04 - Off-centre zooms, the principal point, stale intrinsics, and a training-free fix
#
# 1. **Off-centre crops.** A phone crop or a detection crop is rarely centred. Does alpha
#    depend on where we zoom? (50 images per set, KITTI and NYUv2, centre + four quadrants.)
# 2. **Principal point.** For an off-centre crop the law also moves the principal point.
#    Do models that predict intrinsics follow it, or assume it sits in the image centre?
# 3. **Stale intrinsics.** Metric3D v2 with the unzoomed K: alpha_stale = alpha_true + 1 by
#    construction (paper, Sec. 3); here we check the data.
# 4. **ZoomFix.** When the zoom factor is known (EXIF DigitalZoomRatio, an app's own crop),
#    multiplying depth by s**e undoes the shrink. Two ways to pick e: leave-one-dataset-out on
#    labelled data, or label-free from the same domain (alpha measured on half the images,
#    scored on the other half). The second needs the --fine-fix run (exponents up to 1.5).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display  # a Jupyter builtin; imported so the file also runs as a script
import numpy as np

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import analysis as A
import plotstyle as ps

ps.use()
TAB = ROOT / "outputs/tables"

# %% [markdown]
# ## 1. Alpha by crop position

# %%
off = A.load("*__offc.csv")
pio = A.per_image(off)
pio["where"] = np.where((pio.cx == 0.5) & (pio.cy == 0.5), "centre",
                        np.where(pio.cy < 0.5, "top", "bottom") + "-" + np.where(pio.cx < 0.5, "left", "right"))
order = ["centre", "top-left", "top-right", "bottom-left", "bottom-right"]
tab = pio.groupby(["model", "dataset", "where"]).alpha.median().unstack("where")[order]
tab.round(3).to_csv(TAB / "offcentre_alpha.csv")
display(tab.round(2))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, ds in zip(axes, ["kitti", "nyu"]):
    t = tab.xs(ds, level="dataset")
    for m, r in t.iterrows():
        ax.plot(order, r.values, **ps.line(m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=A.PRETTY_DS[ds], ylabel=r"median $\alpha$")
    ax.tick_params(axis="x", rotation=25)
axes[0].legend()
fig.tight_layout()
ps.save(fig, "figS_offcentre_alpha")

# %% [markdown]
# ## 2. Principal point under off-centre zooms
# Error in units of image width. "centred guess" is the error of always answering the
# image centre; a model that tracks the law should be well below it.

# %%
pp = A.principal_point(off)
if len(pp):
    pp.round(4).to_csv(TAB / "principal_point.csv", index=False)
    q = pp[pp.offset == "quadrant"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, ds in zip(axes, ["kitti", "nyu"]):
        for m in q.model.unique():
            g = q[(q.model == m) & (q.dataset == ds)].sort_values("zoom")
            ax.plot(g.zoom, g.model_error, **ps.line(m))
        g = q[q.dataset == ds].groupby("zoom").centred_guess_error.median()
        ax.plot(g.index, g.values, color="0.4", ls=":", lw=1.2, label="always the centre")
        ax.set(title=A.PRETTY_DS[ds], xlabel="zoom $s$", ylabel="|principal point error| / width")
    axes[0].legend()
    fig.tight_layout()
    ps.save(fig, "figS_principal_point")
    display(q.groupby(["model", "dataset"])[["model_error", "centred_guess_error"]].median().round(3))
else:
    print("no principal-point columns yet (they come with the audit-a2 run)")

# %% [markdown]
# ## 3. Stale K for Metric3D v2

# %%
main = A.load()
m3 = A.per_image(main[main.model.isin(["metric3d-s", "metric3d-s-staleK"])])
w = m3.pivot_table(index=["dataset", "name"], columns="model", values="alpha")
diff = (w["metric3d-s-staleK"] - w["metric3d-s"]).dropna()
print(f"alpha(stale) - alpha(true): mean {diff.mean():.4f}, median {diff.median():.4f}, n = {len(diff)}")
stale = A.summary(m3).pivot(index="dataset", columns="model", values=["alpha", "absrel1", "absrel3"])
stale.round(3).to_csv(TAB / "stale_K.csv")
stale.round(3)

# %% [markdown]
# ## 4. ZoomFix: a training-free correction when the zoom factor is known

# %%
# deployed metric models only: not the fine-tuned variants (their own notebook), not the runs given the true
# camera ("-k": they already know the zoom), not DA3-L and VGGT, which normalise depth per scene, so a metric
# correction means nothing for them, and not GeoCalib, which predicts no depth
deployed = [m for m in main.model.unique()
            if not m.startswith("dav2-s-") and not m.endswith("-k")
            and m not in ("depthpro-eftf", "da3-l", "vggt", "geocalib")]
pi = A.per_image(main[main.model.isin(deployed)])
fix = A.zoomfix(pi)
if len(fix):
    fix.round(4).to_csv(TAB / "zoomfix.csv", index=False)
    pooled = fix.groupby("model")[["exponent", "absrel_zoomed", "absrel_fixed", "absrel_unzoomed_same_pixels"]].mean()
    pooled.index = [ps.LABEL.get(m, m) for m in pooled.index]
    display(pooled.round(3))

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    models = [m for m in fix.model.unique()]
    x = np.arange(len(models))
    wd = 0.27
    g = fix.groupby("model")[["absrel_zoomed", "absrel_fixed", "absrel_unzoomed_same_pixels"]].mean().loc[models]
    ax.bar(x - wd, g.absrel_zoomed, wd, color="0.75", label="3x zoom, as is")
    ax.bar(x, g.absrel_fixed, wd, color=[ps.colour(m) for m in models], label="3x zoom + ZoomFix")
    ax.bar(x + wd, g.absrel_unzoomed_same_pixels, wd, color="none", edgecolor="k", lw=0.8,
           label="unzoomed, same pixels (target)")
    ax.set_xticks(x, [ps.LABEL.get(m, m) for m in models], rotation=20)
    ax.set_ylabel("AbsRel")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    ps.save(fig, "fig6_zoomfix")
else:
    print("no ZoomFix columns yet (they come with the audit-a2 run)")

# %%
fine = list(A.RESULTS.glob("*__fix.csv"))
if fine:
    cf = A.zoomfix_crossfit(A.load("*__fix.csv"))
    cf.round(4).to_csv(TAB / "zoomfix_crossfit.csv", index=False)
    display(cf.round(3))
else:
    print("cross-fitted ZoomFix needs the --fine-fix run (audit-d)")
