"""Write every number the paper quotes, straight from the audit CSVs.

    python scripts/make_paper_numbers.py

Outputs paper/gen/numbers.tex (macros such as \\alphaDAVS) and paper/gen/table*.tex.
Nothing in the paper text is typed by hand; rerun this after any new result.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402

GEN = ROOT / "paper" / "gen"
GEN.mkdir(parents=True, exist_ok=True)
MAIN = ["dav2-s", "dav2-b", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s", "metric3d-s-staleK"]
FT = ["dav2-s-fixed", "dav2-s-anchor", "dav2-s-eft"]

macros = {}


def key(*parts):
    """'dav2-s', 'alpha' -> 'DavsAlpha'-style macro name (letters only, as LaTeX needs)."""
    word = "".join(p.title() for p in re.split(r"[^A-Za-z0-9]+", "-".join(parts)) if p)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}
    return "".join(digits.get(c, c) for c in word)


def put(name, value, fmt="{:.2f}"):
    macros[name] = fmt.format(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else "--"


df = A.load()
have = [m for m in MAIN if m in set(df.model)]
pi = A.per_image(df[df.model.isin(have)])
blur = A.blur_control(df)

# ---- pooled numbers per model
pooled = A.summary(pi, by=("model",)).set_index("model")
for m in have:
    r = pooled.loc[m]
    put(key(m, "alpha"), r.alpha)
    put(key(m, "alphaLo"), r.alpha_lo)
    put(key(m, "alphaHi"), r.alpha_hi)
    put(key(m, "beta"), r.beta)
    put(key(m, "die"), r.die2)
    put(key(m, "absrelOne"), r.absrel1, "{:.3f}")
    put(key(m, "absrelThree"), r.absrel3, "{:.3f}")
    put(key(m, "absrelGrowth"), 100 * (r.absrel3 / r.absrel1 - 1), "{:.0f}")
    put(key(m, "ratioThree"), float(np.exp(-r.alpha * np.log(3))))
    if np.isfinite(r.beta):
        put(key(m, "focalThree"), float(np.exp(r.beta * np.log(3))))
        put(key(m, "alphaPlusBeta"), r.alpha + r.beta)
    put(key(m, "blur"), 100 * (np.exp(blur.xs(m, level="model").mean()) - 1), "{:.1f}")
put("nImages", int(pi.groupby(["model"]).name.nunique().min()), "{:d}")
put("nModels", len(have), "{:d}")

# ---- Table 1
rows = []
for m in have:
    r = pooled.loc[m]
    beta = f"{r.beta:.2f}" if np.isfinite(r.beta) else "--"
    rows.append(f"{A.PRETTY[m]} & {A.GROUP[m]} & {r.alpha:.2f} [{r.alpha_lo:.2f}, {r.alpha_hi:.2f}] & {beta} & "
                f"{r.die2:.3f} & {r.absrel1:.3f} & {r.absrel2:.3f} & {r.absrel3:.3f} \\\\")
(GEN / "table1.tex").write_text("\n".join([
    r"\begin{table*}[t]\centering\small",
    r"\caption{Zoom audit, pooled over KITTI, NYUv2, DIODE-indoor and DIODE-outdoor (" + macros["nImages"]
    + r" images per model). $\alpha$: depth exponent (ideal 0, apparent size only 1), median over images with "
      r"95\% bootstrap interval. $\beta$: focal exponent (ideal 1). DIE$_2$: median $|\log \hat D_2/\hat D_1|$. "
      r"AbsRel on the pixels of the crop at zoom $1\times$, $2\times$, $3\times$.}",
    r"\label{tab:audit}",
    r"\begin{tabular}{llcccccc}\toprule",
    r"Model & Group & $\alpha$ & $\beta$ & DIE$_2$ & AbsRel $1\times$ & $2\times$ & $3\times$ \\ \midrule",
    *rows,
    r"\bottomrule\end{tabular}\end{table*}"]))

# ---- per-dataset alpha table (supplement / main if space)
summ = A.summary(pi)
lines = []
for m in have:
    vals = []
    for ds in A.DATASETS:
        r = summ[(summ.model == m) & (summ.dataset == ds)]
        vals.append(f"{r.alpha.iloc[0]:.2f}" if len(r) else "--")
    lines.append(f"{A.PRETTY[m]} & " + " & ".join(vals) + r" \\")
(GEN / "table_alpha_by_dataset.tex").write_text("\n".join([
    r"\begin{table}[t]\centering\small",
    r"\caption{Depth exponent $\alpha$ per dataset (median over images).}\label{tab:alpha_ds}",
    r"\begin{tabular}{lcccc}\toprule",
    "Model & " + " & ".join(A.PRETTY_DS[d] for d in A.DATASETS) + r" \\ \midrule",
    *lines,
    r"\bottomrule\end{tabular}\end{table}"]))

# ---- zoom-squash prediction: alpha vs depth spread
stats_f = A.RESULTS / "scene_stats.csv"
if stats_f.exists():
    from scipy.stats import spearmanr

    st = pd.read_csv(stats_f)
    j = pi[(pi.cx == 0.5) & (pi.cy == 0.5)].merge(st[["name", "depth_spread"]], on="name")
    for m in have:
        g = j[j.model == m].dropna(subset=["alpha", "depth_spread"])
        if len(g) > 20:
            put(key(m, "spreadRho"), spearmanr(g.depth_spread, g.alpha).statistic)

# ---- label-free proxy across model x dataset pairs
from scipy.stats import spearmanr  # noqa: E402

pairs = summ.assign(abs_alpha=summ.alpha.abs())
if len(pairs) > 4:
    put("proxyRhoTwo", spearmanr(pairs.abs_alpha, pairs.absrel2).statistic)
    put("proxyRhoThree", spearmanr(pairs.abs_alpha, pairs.absrel3).statistic)
    put("proxyPairs", len(pairs), "{:d}")

# ---- fine-tuning
ft = df[df.model.isin(FT)]
if len(ft):
    sft = A.summary(A.per_image(ft))
    rows = []
    for ds in A.DATASETS:
        cells = []
        for m in FT:
            r = sft[(sft.model == m) & (sft.dataset == ds)]
            cells += [f"{r.alpha.iloc[0]:.2f}", f"{r.absrel1.iloc[0]:.3f}", f"{r.absrel3.iloc[0]:.3f}"] if len(r) else ["--"] * 3
            if len(r):
                put(key(m, ds, "alpha"), r.alpha.iloc[0])
                put(key(m, ds, "absrelOne"), r.absrel1.iloc[0], "{:.3f}")
                put(key(m, ds, "absrelThree"), r.absrel3.iloc[0], "{:.3f}")
        rows.append(A.PRETTY_DS[ds] + (" (unseen)" if ds.startswith("diode") else "") + " & " + " & ".join(cells) + r" \\")
    (GEN / "table2.tex").write_text("\n".join([
        r"\begin{table*}[t]\centering\small",
        r"\caption{Fine-tuning DA-V2-S on unlabeled images. Same preprocessing for all three. "
        r"Anchor-only: same data and steps, no zoom loss. DIODE is not used in training.}\label{tab:eft}",
        r"\begin{tabular}{l ccc ccc ccc}\toprule",
        r"& \multicolumn{3}{c}{Frozen} & \multicolumn{3}{c}{Anchor-only FT} & \multicolumn{3}{c}{EFT} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Dataset & $\alpha$ & AbsRel $1\times$ & $3\times$ & $\alpha$ & AbsRel $1\times$ & $3\times$ & $\alpha$ & AbsRel $1\times$ & $3\times$ \\ \midrule",
        *rows,
        r"\bottomrule\end{tabular}\end{table*}"]))

(GEN / "numbers.tex").write_text("% generated by scripts/make_paper_numbers.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(len(macros), "macros ->", GEN / "numbers.tex")
