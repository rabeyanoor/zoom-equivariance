"""Write every number the paper quotes, straight from the audit CSVs.

    python scripts/make_paper_numbers.py

Outputs paper/gen/numbers.tex (one \\newcommand per number) and paper/gen/table*.tex.
Nothing in the paper text is typed by hand; rerun this after any new result. Every macro the
paper uses is always defined: a number whose run is not in yet prints as "--", so the paper
compiles at any stage and the missing results are visible.

Macro names: key("dav2-s", "alpha") -> \\DavTwoSAlpha (LaTeX names may only hold letters).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402

GEN = ROOT / "paper" / "gen"
GEN.mkdir(parents=True, exist_ok=True)
MAIN = ["dav2-s", "dav2-b", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s", "metric3d-s-staleK"]
LADDER = ["dav2-l", "moge2-b", "moge2-l", "unidepth2-b", "unidepth2-l", "unik3d-l", "da3-nested", "da3-metric-l",
          "metric3d-l", "da3-l", "vggt"]
RELATIVE = {"da3-l", "vggt"}  # depth normalised per scene: only beta is meaningful
FT = ["dav2-s-fixed", "dav2-s-zhong", "dav2-s-blur", "dav2-s-eft", "dav2-s-eftb", "dav2-s-eftb3"]
CONTROLS = ["optical", "digital_lr", "bilinear", "jpeg"]
macros = {}


def key(*parts):
    word = "".join(p[:1].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", "-".join(parts)) if p)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six"}
    return "".join(digits.get(c, c) for c in word)


def put(name, value, fmt="{:.2f}"):
    ok = isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)
    macros[name] = fmt.format(value) if ok else "--"


def fmt(x, d=2):
    return f"{x:.{d}f}" if x is not None and np.isfinite(x) else "--"


def write_table(name, lines):
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")


df = A.load()
present = set(df.model)
pi = A.per_image(df[df.model.isin(MAIN + LADDER)])
pi.loc[pi.model.isin(RELATIVE), "alpha"] = np.nan
blur = A.blur_control(df)
pooled = A.summary(pi, by=("model",), scene_ci=True).set_index("model")
summ = A.summary(pi)
ab = pi.assign(ab=pi.alpha + pi.beta).groupby("model").ab.median()

# ------------------------------------------------------------------ per-model numbers (always defined)
for m in MAIN + LADDER:
    r = pooled.loc[m] if m in pooled.index else pd.Series(dtype=float)
    g = r.get
    put(key(m, "alpha"), g("alpha", np.nan))
    put(key(m, "alphaLo"), g("alpha_lo_scene", np.nan))
    put(key(m, "alphaHi"), g("alpha_hi_scene", np.nan))
    put(key(m, "beta"), g("beta", np.nan))
    put(key(m, "alphaPlusBeta"), ab.get(m, np.nan))
    put(key(m, "die"), g("die2", np.nan), "{:.3f}")
    put(key(m, "absrelOne"), g("absrel1", np.nan), "{:.3f}")
    put(key(m, "absrelThree"), g("absrel3", np.nan), "{:.3f}")
    put(key(m, "baseThree"), g("base3", np.nan), "{:.3f}")
    a, b = g("alpha", np.nan), g("beta", np.nan)
    put(key(m, "ratioThree"), np.exp(-a * np.log(3)) if np.isfinite(a) else np.nan)
    put(key(m, "focalThree"), np.exp(b * np.log(3)) if np.isfinite(b) else np.nan)
    bl = blur.xs(m, level="model").mean() if m in blur.index.get_level_values("model") else np.nan
    put(key(m, "blur"), 100 * (np.exp(bl) - 1) if np.isfinite(bl) else np.nan, "{:.1f}")
n_main = pi[pi.model.isin(MAIN)].groupby("model").name.nunique()
put("nImages", int(n_main.min()) if len(n_main) else np.nan, "{:d}")
put("nScenes", int(pooled.n_scenes.max()) if "n_scenes" in pooled else np.nan, "{:d}")

# stale K: alpha(stale) - alpha(true), per image
w = pi.pivot_table(index=["dataset", "name"], columns="model", values="alpha")
if {"metric3d-s", "metric3d-s-staleK"} <= set(w.columns):
    put("staleDiff", (w["metric3d-s-staleK"] - w["metric3d-s"]).median(), "{:.3f}")
else:
    put("staleDiff", np.nan)

# ------------------------------------------------------------------ Table 1 (main audit)
rows = []
for m in MAIN:
    if m not in pooled.index:
        continue
    r = pooled.loc[m]
    rows.append(f"{A.PRETTY[m]} & {A.GROUP[m]} & {fmt(r.alpha)} [{fmt(r.alpha_lo_scene)}, {fmt(r.alpha_hi_scene)}] & "
                f"{fmt(r.beta)} & {fmt(ab.get(m, np.nan))} & {fmt(100 * (np.exp(blur.xs(m, level='model').mean()) - 1), 1)}\\% & "
                f"{fmt(r.absrel1, 3)} & {fmt(r.base3, 3)} & {fmt(r.absrel3, 3)} \\\\")
write_table("table1", [
    r"\begin{table*}[t]\centering\small",
    r"\caption{Zoom audit, pooled over KITTI, NYUv2, DIODE-indoor and DIODE-outdoor (\nImages{} images per model, "
    r"\nScenes{} scenes). $\alpha$: depth exponent (ideal 0; apparent size only: 1), median over images with a 95\% "
    r"interval from a scene-level bootstrap. $\beta$: focal exponent (ideal 1). $\alpha+\beta$: median per image "
    r"(1 = lateral sizes kept, \cref{sec:law}). Blur: depth change under the blur control. AbsRel at $1\times$ on the "
    r"full image, and at $3\times$ on the crop for the unzoomed prediction (``same px'') and the zoomed one.}",
    r"\label{tab:audit}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llccccccc}\toprule",
    r"& & & & & & \multicolumn{3}{c}{AbsRel} \\ \cmidrule(lr){7-9}",
    r"Model & Group & $\alpha$ & $\beta$ & $\alpha+\beta$ & Blur & $1\times$ & $3\times$ same px & $3\times$ zoomed \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ Table 3 (model ladder)
# the ladder runs use a smaller sample, so the fair comparison is on the images every model saw
ladder_models = [x for x in MAIN + LADDER if x in pooled.index and not x.endswith("staleK")]
seen = pi[pi.model.isin(ladder_models)].groupby(["dataset", "name"]).model.nunique()
shared = seen[seen == len(ladder_models)].index
sh = pi.set_index(["dataset", "name"])
sh = sh[sh.index.isin(shared)].groupby("model")[["alpha", "beta"]].median()
put("nLadderShared", len(shared), "{:d}")
for m in ladder_models:
    put(key(m, "alphaShared"), sh.alpha.get(m, np.nan))
    put(key(m, "betaShared"), sh.beta.get(m, np.nan))
rows = []
for m in ladder_models:
    r = pooled.loc[m]
    group = A.GROUP.get(m, "").replace("pixels only", "pixels").replace("predicts camera", "pred. K") \
        .replace("takes camera (true K)", "takes K")
    rows.append(f"{A.PRETTY.get(m, m)} & {group} & {fmt(r.alpha)} & {fmt(r.beta)} & {int(r.n)} & "
                f"{fmt(sh.alpha.get(m, np.nan))} & {fmt(sh.beta.get(m, np.nan))} \\\\")
write_table("table_ladder", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Model ladder: larger variants and newer families. Left: each model on its own sample. Right: all "
    r"models on the \nLadderShared{} images that every one of them saw (paired). DA3-L and VGGT normalise depth per "
    r"scene, so only $\beta$ is defined for them. pred.\ K: predicts the camera; takes K: given the true "
    r"camera.}\label{tab:ladder}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llccccc}\toprule",
    r"& & \multicolumn{3}{c}{own sample} & \multicolumn{2}{c}{shared} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-7}",
    r"Model & Group & $\alpha$ & $\beta$ & images & $\alpha$ & $\beta$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table}"])

# ------------------------------------------------------------------ per-dataset alpha
lines = []
for m in [x for x in MAIN if x in present]:
    cells = []
    for ds in A.DATASETS:
        r = summ[(summ.model == m) & (summ.dataset == ds)]
        cells.append(fmt(r.alpha.iloc[0]) if len(r) else "--")
    lines.append(f"{A.PRETTY[m]} & " + " & ".join(cells) + r" \\")
write_table("table_alpha_by_dataset", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Depth exponent $\alpha$ per dataset (median over images).}\label{tab:alpha_ds}",
    r"\begin{tabular}{lcccc}\toprule",
    "Model & " + " & ".join(A.PRETTY_DS[d] for d in A.DATASETS) + r" \\ \midrule",
    *lines, r"\bottomrule\end{tabular}\end{table}"])

# ------------------------------------------------------------------ zoom-squash: alpha vs depth spread
stats_f = A.RESULTS / "scene_stats.csv"
for m in ("depthpro", "moge2-s", "unidepth2-s", "dav2-s"):
    rho = np.nan
    if stats_f.exists():
        st = pd.read_csv(stats_f)
        g = pi[(pi.model == m) & (pi.cx == 0.5)].merge(st[["name", "depth_spread"]], on="name").dropna(
            subset=["alpha", "depth_spread"])
        if len(g) > 20:
            g = g.assign(rs=g.groupby("dataset").depth_spread.rank(pct=True), ra=g.groupby("dataset").alpha.rank(pct=True))
            rho = spearmanr(g.rs, g.ra).statistic
    put(key(m, "spreadRho"), rho)

# ------------------------------------------------------------------ label-free proxy
pairs = summ[summ.model.isin(MAIN)].assign(abs_alpha=lambda d: d.alpha.abs())
for tgt, tag in (("absrel1", "One"), ("absrel3", "Three")):
    put(f"proxyRho{tag}", spearmanr(pairs.abs_alpha, pairs[tgt]).statistic if len(pairs) > 4 else np.nan)
put("proxyPairs", len(pairs), "{:d}")
proxy_f = ROOT / "outputs" / "tables" / "proxy_across_pairs.csv"
if proxy_f.exists():
    pr = pd.read_csv(proxy_f).set_index(["score", "target"]).loc[("abs_alpha", "absrel3")]
    put("proxyRhoThreeLo", pr.lo)
    put("proxyRhoThreeHi", pr.hi)
else:
    put("proxyRhoThreeLo", np.nan)
    put("proxyRhoThreeHi", np.nan)
for ds in A.DATASETS:
    g = pairs[pairs.dataset == ds]
    put(key("proxy", ds), spearmanr(g.abs_alpha, g.absrel3).statistic if len(g) > 3 else np.nan)

# ------------------------------------------------------------------ controls: optical vs digital, interpolation, JPEG
ctrl_files = list(A.RESULTS.glob("*__ctrl.csv"))
ctrl = A.load("*__ctrl.csv") if ctrl_files else pd.DataFrame(columns=["model", "kind", "log_ratio_self"])
rows = []
for m in ("dav2-s", "depthpro", "moge2-s", "metric3d-s"):
    cells = []
    for c in CONTROLS:
        v = ctrl[(ctrl.model == m) & (ctrl.kind == c)].log_ratio_self
        ratio = float(np.exp(v.median())) if len(v) else np.nan
        put(key(m, c), ratio)
        cells.append(fmt(ratio))
    rows.append(f"{A.PRETTY[m]} & " + " & ".join(cells) + r" \\")
write_table("table_controls", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Confounder controls: median depth ratio on the same pixels (1 = unchanged). Optical: real-detail "
    r"$2\times$ zoom of a half-resolution camera; digital: the same zoom by crop and up-sampling; bilinear: the "
    r"$2\times$ digital zoom with bilinear resampling; JPEG: quality 75, no zoom.}\label{tab:controls}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lcccc}\toprule",
    r"Model & optical $2\times$ & digital $2\times$ & bilinear $2\times$ & JPEG \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table}"])

# ------------------------------------------------------------------ ZoomFix (cross-fitted, label-free exponent)
fix_files = list(A.RESULTS.glob("*__fix.csv"))
fix = A.zoomfix_crossfit(A.load("*__fix.csv")) if fix_files else pd.DataFrame()
for m in MAIN:
    g = fix[fix.model == m] if len(fix) else pd.DataFrame()
    put(key(m, "fixE"), g.exponent.mean() if len(g) else np.nan)
    put(key(m, "fixAbsrel"), g.absrel_fixed.mean() if len(g) else np.nan, "{:.3f}")
    put(key(m, "fixZoomed"), g.absrel_zoomed.mean() if len(g) else np.nan, "{:.3f}")
    put(key(m, "fixTarget"), g.absrel_unzoomed_same_pixels.mean() if len(g) else np.nan, "{:.3f}")
    # per dataset: a pooled mean hides that the fix can hurt where the zoom changes depth unevenly
    for ds in ["kitti", "nyu", "diode-in", "diode-out"]:
        r = g[g.dataset == ds] if len(g) else pd.DataFrame()
        for col, name in [("absrel_zoomed", "Zoomed"), ("absrel_fixed", "Absrel"), ("absrel_unzoomed_same_pixels", "Target")]:
            put(key(m, "fix", ds, name), r[col].iloc[0] if len(r) else np.nan, "{:.3f}")
        # the same-pixel unzoomed error as a median, to show where a heavy tail drives the mean
        f = A.RESULTS / f"{m}__{ds}__fix.csv"
        z = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["kind", "zoom", "abs_rel_base"])
        z = z[(z.kind == "zoom") & (z.zoom == 3.0)]
        put(key(m, "fix", ds, "targetMedian"), z.abs_rel_base.median() if len(z) else np.nan, "{:.3f}")

# ------------------------------------------------------------------ principal point under off-centre zooms
off = A.load("*__offc.csv") if list(A.RESULTS.glob("*__offc.csv")) else pd.DataFrame()
ppt = A.principal_point(off) if len(off) else pd.DataFrame()
for m in ["moge2-s", "unidepth2-s", "depthpro"]:
    g = ppt[(ppt.model == m) & (ppt.offset == "quadrant") & (ppt.zoom == 2.0)] if len(ppt) else pd.DataFrame()
    put(key(m, "ppQuadTwo"), g.model_error.median() if len(g) else np.nan, "{:.3f}")
    put(key(m, "ppGuessTwo"), g.centred_guess_error.median() if len(g) else np.nan, "{:.3f}")

# ------------------------------------------------------------------ fine-tuning (Table 2)
ft = df[df.model.isin(FT)]
sft = A.summary(A.per_image(ft)) if len(ft) else pd.DataFrame(columns=["model", "dataset"])
rows = []
for m in FT:
    for ds in A.DATASETS:
        r = sft[(sft.model == m) & (sft.dataset == ds)]
        get = (lambda c: r[c].iloc[0]) if len(r) else (lambda c: np.nan)
        put(key(m, ds, "alpha"), get("alpha"))
        put(key(m, ds, "absrelOne"), get("absrel1"), "{:.3f}")
        put(key(m, ds, "absrelThree"), get("absrel3"), "{:.3f}")
        bl = blur.get((m, ds), np.nan)
        put(key(m, ds, "blur"), 100 * (np.exp(bl) - 1) if np.isfinite(bl) else np.nan, "{:.1f}")
for ds in A.DATASETS:
    cells = []
    for m in FT:
        cells += [macros[key(m, ds, "alpha")], macros[key(m, ds, "absrelOne")], macros[key(m, ds, "blur")] +
                  (r"\%" if macros[key(m, ds, "blur")] != "--" else "")]
    rows.append(A.PRETTY_DS[ds] + (r"$^\dagger$" if ds.startswith("diode") else "") + " & " + " & ".join(cells) + r" \\")
names = {"dav2-s-fixed": "Frozen", "dav2-s-zhong": "Zhong et al.-style", "dav2-s-blur": "Blur-only",
         "dav2-s-eft": "EFT", "dav2-s-eftb": "EFT+B (ours)", "dav2-s-eftb3": r"EFT+B, $\lambda{=}3$"}
write_table("table2", [
    r"\begin{table*}[t]\centering\scriptsize",
    r"\caption{Fine-tuning DA-V2-S on unlabeled images, identical preprocessing for all variants. For each: "
    r"$\alpha$, AbsRel on unzoomed images, and the depth change under the blur control (the resolution shortcut). "
    r"$^\dagger$never used in training.}\label{tab:eft}",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\begin{tabular}{l" + " ccc" * len(FT) + r"}\toprule",
    "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{names[m]}}}" for m in FT) + r" \\",
    "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(FT))),
    "Dataset" + r" & $\alpha$ & Abs$_{1\times}$ & blur" * len(FT) + r" \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

(GEN / "numbers.tex").write_text("% generated by scripts/make_paper_numbers.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
missing = sum(v == "--" for v in macros.values())
print(f"{len(macros)} macros ({missing} still '--') -> {GEN / 'numbers.tex'}")
