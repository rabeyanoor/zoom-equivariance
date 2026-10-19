# %% [markdown]
# # 03 - Can the zoom test rank models without labels?
#
# The zoom test needs no ground truth: it only compares a model with itself. If its
# score tracks real error, a practitioner can check a model on their own unlabeled
# photos. Two questions:
#
# 1. **Across model x dataset pairs**: does |alpha| (or DIE at 2x) rank-correlate with the
#    AbsRel on *zoomed* images (where the model has to cope with an unusual camera)?
# 2. **Within a pair, per image**: are the images with a large self-inconsistency the ones
#    with a large error? (risk-coverage: drop the most inconsistent images first)
#
# Spearman CIs by bootstrap over the pairs.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import analysis as A

FIG, TAB = ROOT / "outputs/figures", ROOT / "outputs/tables"
MAIN = ["dav2-s", "dav2-b", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s", "metric3d-s-staleK"]

df = A.load()
df = df[df.model.isin(MAIN)]
pi = A.per_image(df)
summ = A.summary(pi)
summ["abs_alpha"] = summ.alpha.abs()


def spearman_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) > 2:
            boots.append(spearmanr(x[i], y[i]).statistic)
    return rho, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


# %% [markdown]
# ## 1. Across pairs

# %%
rows = []
for score in ("abs_alpha", "die2"):
    for target in ("absrel2", "absrel3", "absrel1"):
        rho, lo, hi = spearman_ci(summ[score], summ[target])
        rows.append(dict(score=score, target=target, spearman=rho, lo=lo, hi=hi, pairs=len(summ)))
proxy = pd.DataFrame(rows)
proxy.round(3).to_csv(TAB / "proxy_across_pairs.csv", index=False)
proxy.round(3)

# %% [markdown]
# Within one dataset the comparison is fairer (same scenes, same depth range):

# %%
within = []
for ds, g in summ.groupby("dataset"):
    for target in ("absrel2", "absrel3"):
        within.append(dict(dataset=ds, target=target, spearman=spearmanr(g.abs_alpha, g[target]).statistic, models=len(g)))
within = pd.DataFrame(within)
within.round(3).to_csv(TAB / "proxy_within_dataset.csv", index=False)
within.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
marker = {"kitti": "o", "nyu": "s", "diode-in": "^", "diode-out": "v"}
for ax, target, title in ((axes[0], "absrel2", "AbsRel at 2x zoom"), (axes[1], "absrel3", "AbsRel at 3x zoom")):
    for m in MAIN:
        g = summ[summ.model == m]
        for _, r in g.iterrows():
            ax.scatter(r.abs_alpha, r[target], marker=marker[r.dataset], color=f"C{MAIN.index(m)}",
                       label=A.PRETTY[m] if r.dataset == "kitti" else None, s=45)
    rho = proxy[(proxy.score == "abs_alpha") & (proxy.target == target)].iloc[0]
    ax.set(xlabel="|alpha|  (label-free)", ylabel=title,
           title=f"Spearman {rho.spearman:.2f} [{rho.lo:.2f}, {rho.hi:.2f}]")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "03_proxy_scatter.png", dpi=110)
plt.show()

# %% [markdown]
# ## 2. Per image: risk-coverage
# Sort images by their own |log D_2 / D_1| and keep the most consistent fraction. If the
# test is informative, the AbsRel of the kept images should fall as coverage shrinks,
# faster than with a random order.

# %%
def risk_coverage(g, score, target, steps=10):
    g = g.dropna(subset=[score, target]).sort_values(score)
    cov = np.linspace(0.1, 1.0, steps)
    return cov, np.array([g[target].iloc[: max(1, int(c * len(g)))].mean() for c in cov])


fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=False)
aurc = []
for ax, ds in zip(axes, A.DATASETS):
    for m in MAIN:
        g = pi[(pi.model == m) & (pi.dataset == ds)]
        if len(g) < 10:
            continue
        cov, risk = risk_coverage(g, "die2", "absrel2")
        ax.plot(cov, risk / risk[-1], "-", label=A.PRETTY[m])
        aurc.append(dict(model=m, dataset=ds, rel_aurc=float(np.mean(risk / risk[-1]))))
    ax.axhline(1, color="k", lw=0.8)
    ax.set(title=A.PRETTY_DS[ds], xlabel="coverage (most self-consistent first)", ylabel="AbsRel / AbsRel(all)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=6)
plt.tight_layout()
plt.savefig(FIG / "03_risk_coverage.png", dpi=110)
plt.show()
aurc = pd.DataFrame(aurc)
aurc.round(3).to_csv(TAB / "proxy_risk_coverage.csv", index=False)
print("mean relative AURC (1.0 = no better than random):", aurc.rel_aurc.mean().round(3))
