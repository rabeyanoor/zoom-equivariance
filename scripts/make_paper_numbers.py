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
from scipy.stats import spearmanr, wilcoxon

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
SCALE_FT = ["dav2-s-scale", "dav2-s-scalenb"]  # EFT-S: frozen model + learned per-image log-scale
SEEDS = ["dav2-s-eftb", "dav2-s-eftb-s1", "dav2-s-eftb-s2"]  # EFT+B with training seeds 0, 1, 2
CONTROLS = ["optical", "digital_lr", "bilinear", "jpeg"]
macros = {}


def key(*parts):
    word = "".join(p[:1].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", "-".join(parts)) if p)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six"}
    return "".join(digits.get(c, c) for c in word)


def minus(s):
    """A negative number as a real minus sign in text and in math mode ("-0.04" in text mode is a hyphen, which
    LaTeX may also break across lines)."""
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def put(name, value, fmt="{:.2f}"):
    ok = isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)
    macros[name] = minus(fmt.format(value)) if ok else "--"


def fmt(x, d=2):
    return minus(f"{x:.{d}f}") if x is not None and np.isfinite(x) else "--"


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
    d = (w["metric3d-s-staleK"] - w["metric3d-s"]).dropna()
    put("staleDiff", d.median(), "{:.3f}")
    put("staleDiffMean", d.mean(), "{:.3f}")
    put("staleImages", len(d), "{:d}")
else:
    put("staleDiff", np.nan)
    put("staleDiffMean", np.nan)
    put("staleImages", np.nan)

# ------------------------------------------------------------------ Table 1 (main audit)
# UniDepthV2-S's pooled AbsRel is far above the others; per dataset (mean, as in Table 1) it comes from DIODE-outdoor
for ds in A.DATASETS:
    g = pi[(pi.model == "unidepth2-s") & (pi.dataset == ds) & np.isclose(pi.cx, 0.5) & np.isclose(pi.cy, 0.5)]
    put("ud" + key(ds, "absrelOne"), g.absrel1.mean() if len(g) else np.nan, "{:.2f}")
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
    r"full image, and at $3\times$ on the crop for the unzoomed prediction (``same px'') and the zoomed one. The high AbsRel "
    r"of UniDepthV2-S at $1\times$ is DIODE-outdoor alone (\udDiodeOutAbsrelOne; other sets "
    r"\udNyuAbsrelOne--\udDiodeInAbsrelOne), a scale over-estimate (\cref{sec:results}).}",
    r"\label{tab:audit}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llccccccc}\toprule",
    r"& & & & & & \multicolumn{3}{c}{AbsRel} \\ \cmidrule(lr){7-9}",
    r"Model & Group & $\alpha$ & $\beta$ & $\alpha+\beta$ & Blur & $1\times$ & $3\times$ same px & $3\times$ zoomed \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ Table 3 (model ladder)
# the ladder runs use a smaller sample, so the fair comparison is on the images every model saw;
# the 50-image runs (DA3, VGGT, Metric3D-L) would shrink that set to a few dozen images, so the paired
# column covers the models with at least 100 images per set
ladder_models = [x for x in MAIN + LADDER if x in pooled.index and not x.endswith("staleK")]
n_img = pi.groupby("model").name.size()
paired = [m for m in ladder_models if n_img.get(m, 0) >= 400]
seen = pi[pi.model.isin(paired)].groupby(["dataset", "name"]).model.nunique()
shared = seen[seen == len(paired)].index
sh = pi[pi.model.isin(paired)].set_index(["dataset", "name"])
sh = sh[sh.index.isin(shared)].groupby("model")[["alpha", "beta"]].median()
put("nLadderShared", len(shared), "{:d}")
for m in ladder_models:
    put(key(m, "alphaShared"), sh.alpha.get(m, np.nan))
    put(key(m, "betaShared"), sh.beta.get(m, np.nan))
# the 50-image runs, paired with the main models on the images they share
small = [m for m in ladder_models if m not in paired]
main_here = [m for m in MAIN if m in ladder_models]
seen = pi[pi.model.isin(small + main_here)].groupby(["dataset", "name"]).model.nunique()
shared_small = seen[seen == len(small + main_here)].index
sm = pi[pi.model.isin(small + main_here)].set_index(["dataset", "name"])
sm = sm[sm.index.isin(shared_small)].groupby("model")[["alpha", "beta"]].median()
put("nSmallShared", len(shared_small), "{:d}")
for m in small + main_here:
    put(key(m, "alphaSmallShared"), sm.alpha.get(m, np.nan))
    put(key(m, "betaSmallShared"), sm.beta.get(m, np.nan))
params_file = ROOT / "outputs" / "results" / "params.json"
params = pd.read_json(params_file, typ="series") if params_file.exists() else pd.Series(dtype=float)
for m in ladder_models:
    put(key(m, "params"), params.get(m, np.nan) / 1e6, "{:,.0f}")
rows = []
for m in ladder_models:
    r = pooled.loc[m]
    group = A.GROUP.get(m, "").replace("pixels only", "pixels").replace("predicts camera", "pred. K") \
        .replace("takes camera (true K)", "takes K")
    size = f"{params[m] / 1e6:,.0f}M" if m in params else "--"
    rows.append(f"{A.PRETTY.get(m, m)} & {group} & {size} & {fmt(r.alpha)} & {fmt(r.beta)} & "
                f"{fmt(sh.alpha.get(m, np.nan))} & {fmt(sh.beta.get(m, np.nan))} \\\\")
write_table("table_ladder", [
    r"\begin{table*}[t]\centering\small",  # supplement only: full width
    r"\caption{Model ladder: larger variants and newer families. Left: each model on its own sample (200 images "
    r"per set for the main models, 100 for the larger variants and UniK3D, 50 for DA3, VGGT and Metric3D v2-L). "
    r"Right: the models with at least 100 images per set on the \nLadderShared{} images that all of them saw "
    r"(paired). DA3-L and VGGT normalise depth per scene, so only $\beta$ is defined for them. pred.\ K: predicts "
    r"the camera; takes K: given the true camera.}\label{tab:ladder}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llrcccc}\toprule",
    r"& & & \multicolumn{2}{c}{own sample} & \multicolumn{2}{c}{shared} \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Model & Group & Params & $\alpha$ & $\beta$ & $\alpha$ & $\beta$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ per-dataset alpha
lines = []
for m in [x for x in MAIN if x in present]:
    cells = []
    for ds in A.DATASETS:
        r = summ[(summ.model == m) & (summ.dataset == ds)]
        cells.append(fmt(r.alpha.iloc[0]) if len(r) else "--")
        put(key(m, "ds", ds, "alpha"), r.alpha.iloc[0] if len(r) else np.nan)  # e.g. \DepthproDsDiodeOutAlpha
    lines.append(f"{A.PRETTY[m]} & " + " & ".join(cells) + r" \\")
write_table("table_alpha_by_dataset", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Depth exponent $\alpha$ per dataset (median over images).}\label{tab:alpha_ds}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule",
    "Model & " + " & ".join(A.PRETTY_DS[d] for d in A.DATASETS) + r" \\ \midrule",
    *lines, r"\bottomrule\end{tabular}}\end{table}"])

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
# pairs share models and datasets: an interval that resamples both, and the zoom-only error change
# (AbsRel at 3x minus the unzoomed prediction on the same pixels), which removes the error the model has anyway
pairs = pairs.assign(dabsrel3=pairs.absrel3 - pairs.base3)
for tgt, tag in (("absrel3", "Three"), ("dabsrel3", "DeltaThree")):
    if len(pairs) > 4 and pairs[tgt].notna().all():
        r, lo, hi = A.twoway_boot_spearman(pairs, "abs_alpha", tgt)
    else:
        r = lo = hi = np.nan
    put(f"proxyRho{tag}TwoWay", r)
    put(f"proxyRho{tag}TwoWayLo", lo)
    put(f"proxyRho{tag}TwoWayHi", hi)
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

# ------------------------------------------------------------------ selective prediction (notebook 03)
rc_f = ROOT / "outputs" / "tables" / "proxy_risk_coverage.csv"
rc = pd.read_csv(rc_f) if rc_f.exists() else pd.DataFrame(columns=["model", "dataset", "rel_aurc"])
rc = rc[rc.model.isin(MAIN) & (rc.model != "metric3d-s-staleK")]
put("riskPairs", len(rc), "{:d}")
put("riskBelowOne", int((rc.rel_aurc < 1).sum()), "{:d}")
put("riskAurc", rc.rel_aurc.mean())
for ds in A.DATASETS:
    put(key("risk", ds), rc[rc.dataset == ds].rel_aurc.mean())

# ------------------------------------------------------------------ cue split: context loss vs magnification
CUE_MODELS = ["dav2-s", "zoedepth", "moge2-s", "unidepth2-s", "depthpro", "da3-metric-l"]
CUE_FT = ["dav2-s-fixed", "dav2-s-eftb", "dav2-s-scale"]  # second block: does fine-tuning change the cue?
CUE_KINDS = ["context", "magnify", "zoom", "shrink"]
cue_files = list(A.RESULTS.glob("*__cues.csv"))
cues = pd.concat([pd.read_csv(f) for f in cue_files], ignore_index=True) if cue_files else \
    pd.DataFrame(columns=["model", "kind", "name", "log_ratio_self", "fx", "fx_base"])
rows = []
for m in CUE_MODELS + CUE_FT:
    if m == CUE_FT[0] and cues.model.isin(CUE_FT).any():
        rows.append(r"\midrule")
    g = cues[cues.model == m]
    put(key(m, "cue", "images"), g.name.nunique(), "{:d}")
    cells = []
    for c in CUE_KINDS:
        v = g[g.kind == c].log_ratio_self
        ratio = float(np.exp(v.median())) if len(v) else np.nan
        put(key(m, "cue", c), ratio)
        cells.append(fmt(ratio))
    lz = g[g.kind == "zoom"].log_ratio_self.median() if len(g) else np.nan
    lm = g[g.kind == "magnify"].log_ratio_self.median() if len(g) else np.nan
    put(key(m, "cue", "share"), 100 * lm / lz if len(g) and lz != 0 else np.nan, "{:.0f}")
    sh = g[g.kind == "shrink"]
    focal = float(np.exp(np.median(np.log(sh.fx / sh.fx_base)))) if len(sh) and sh.fx.notna().any() else np.nan
    put(key(m, "cue", "shrink", "focal"), focal)
    cells.append(fmt(focal) if np.isfinite(focal) else "--")
    if len(g):
        # short names in the lower block so the table fits one column
        name = A.PRETTY.get(m, m).replace("DA-V2-S + ", "").replace(" (ours)", "").replace("DA-V2-S (frozen)", "frozen")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
# per dataset: is magnification the consistent cause? It is for the pixels-only models; the camera-predicting ones
# differ by model (MoGe-2 magnification, UniDepthV2 context, Depth Pro by dataset), so they are reported per model
CUE_PIX = ["dav2-s", "zoedepth"]
per_ds = cues[cues.model.isin(CUE_PIX)].groupby(["model", "dataset", "kind"]).log_ratio_self.median().pipe(np.exp)
mag, ctx = per_ds.xs("magnify", level="kind"), per_ds.xs("context", level="kind")
put("cuePixPairs", len(mag), "{:d}")
put("cueMagnifyBelowOne", int((mag < 1).sum()), "{:d}")
put("cueMagnifyMin", mag.min() if len(mag) else np.nan)
put("cueMagnifyMax", mag.max() if len(mag) else np.nan)
put("cueContextMin", ctx.min() if len(ctx) else np.nan)
put("cueContextMax", ctx.max() if len(ctx) else np.nan)
for m in ["moge2-s", "unidepth2-s", "depthpro"]:
    g = cues[cues.model == m].groupby(["dataset", "kind"]).log_ratio_self.median().pipe(np.exp)
    put(key(m, "cue", "magnifyBelowOne"), int((g.xs("magnify", level="kind") < 1).sum()) if len(g) else np.nan, "{:d}")
    put(key(m, "cue", "contextBelowOne"), int((g.xs("context", level="kind") < 1).sum()) if len(g) else np.nan, "{:d}")
# the zoom error of the frozen DA-V2-S at 3x is almost one number per image: global log shift against shape error
z3 = df[(df.model == "dav2-s-fixed") & (df.kind == "zoom") & np.isclose(df.zoom, 3.0) & np.isclose(df.cx, 0.5)]
put("fixedShiftThree", np.median(np.abs(z3.log_ratio_self)) if len(z3) else np.nan)
put("fixedShapeThree", np.nanmedian(z3.shape_err) if "shape_err" in z3 and len(z3) else np.nan, "{:.3f}")

write_table("table_cues", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Splitting a $2\times$ zoom into its two parts, median depth ratio on the same pixels (1 = unchanged). "
    r"\emph{Context}: the crop pasted at its own place and size on a plain canvas. \emph{Magnify}: the zoomed image "
    r"against that pasted crop. \emph{Zoom}: both together. \emph{Shrink}: the whole image made $2\times$ smaller on a "
    r"canvas. A model that reads apparent size gives 1, 0.5, 0.5 and 2. $^{*}$Predicted focal on the shrunk "
    r"image (correct: 0.5). Lower block: DA-V2-S at the fine-tuning input size, frozen and tuned with EFT+B and EFT-S "
    r"(\cref{sec:eft}).}\label{tab:cues}",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{lccccc}\toprule",
    r"Model & context & magnify & zoom & shrink & focal$^{*}$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table}"])

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

# JPEG together with a 2x zoom (*__jzfull.csv, complete test sets): median depth ratio on the same pixels; the two
# JPEG orders are paired against the plain zoom of the same image
_jzpat = "*__jzfull.csv" if list(A.RESULTS.glob("*__jzfull.csv")) else "*__jz.csv"
if list(A.RESULTS.glob(_jzpat)):
    jz = A.load(_jzpat)
    rows = []
    for m in ("dav2-s", "depthpro", "moge2-s", "metric3d-s", "unidepth2-s"):
        t = jz[jz.model == m].pivot_table(index=["dataset", "name"], columns="kind", values="log_ratio_self")
        if t.empty:
            continue
        cells = {"JzZoom": t["zoom"], "JzJpegOne": t["jpeg"], "JpegAfterZoom": t["jpeg_after_zoom"] - t["zoom"],
                 "ZoomOfJpeg": t["zoom_of_jpeg"] - t["zoom"]}
        vals = {k: float(np.exp(v.median())) for k, v in cells.items()}
        for k, v in vals.items():
            put(key(m, k), v)
        kit = t.xs("kitti", level="dataset")
        put(key(m, "ZoomOfJpegKitti"), float(np.exp((kit["zoom_of_jpeg"] - kit["zoom"]).median())))
        put(key(m, "JzImages"), len(t), "{:d}")
        rows.append(f"{A.PRETTY[m]} & {len(t)} & " + " & ".join(fmt(vals[k]) for k in cells) + r" \\")
    write_table("table_jpegzoom", [
        r"\begin{table}[t]\centering\small",
        r"\caption{JPEG (quality 75) together with a $2\times$ digital zoom, on the complete test sets (all four "
        r"datasets pooled). Median depth ratio on the same pixels. Zoom: the plain zoom against the unzoomed image. "
        r"JPEG $1\times$: compression alone. JPEG after zoom (a camera that zooms and saves a JPEG) and zoom of a JPEG "
        r"(cropping a saved photo): against the plain zoom of the same image, so 1 means JPEG adds nothing.}"
        r"\label{tab:jpegzoom}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccccc}\toprule",
        r"Model & images & zoom $2\times$ & JPEG $1\times$ & JPEG after zoom & zoom of JPEG \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"])

# goodness of the power law: RMS residual in log depth, and alpha from small vs large zooms
fq = A.fit_quality(df[df.model.isin(MAIN)])
put("fitRmsMedian", fq.rms.median() if len(fq) else np.nan, "{:.3f}")
for m in ("dav2-s", "depthpro", "moge2-s", "unidepth2-s"):
    g = fq[fq.model == m]
    put(key(m, "alpha", "small"), g.alpha_small.median() if len(g) else np.nan)
    put(key(m, "alpha", "large"), g.alpha_large.median() if len(g) else np.nan)

# calibration-only reference: GeoCalib's focal exponent under the same zooms (its depth is a constant, so no alpha)
gc = df[df.model == "geocalib"]
gpi = A.per_image(gc) if len(gc) else pd.DataFrame(columns=["beta", "fx_err1", "dataset"])
put("geocalibBeta", gpi.beta.median() if len(gpi) else np.nan)
put("geocalibImages", len(gpi), "{:d}")
put("geocalibFocalErr", float(np.exp(gpi.fx_err1.median())) if len(gpi) else np.nan)
for ds in A.DATASETS:
    g = gpi[gpi.dataset == ds]
    put(key("geocalib", ds, "beta"), g.beta.median() if len(g) else np.nan)

# a deployable alternative to the true camera: Depth Pro's metres are its focal times a camera-free output, so giving it
# GeoCalib's focal estimate instead rescales each depth map exactly by f_geocalib / f_depthpro (same images and zooms)
def _swap_focal(z, f_new):
    cf = np.log(f_new / z.fx)
    c1 = z.assign(cf=cf)[np.isclose(z.zoom, 1.0)].set_index("name").cf
    return z.assign(log_ratio_self=z.log_ratio_self + cf - z.name.map(c1).values, log_ratio_gt=z.log_ratio_gt + cf)


_dpz = df[(df.model == "depthpro") & (df.kind == "zoom") & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)]
_gcz = gc[(gc.kind == "zoom") & np.isclose(gc.cx, 0.5) & np.isclose(gc.cy, 0.5)]
_m = _dpz.merge(_gcz[["name", "zoom", "fx"]], on=["name", "zoom"], suffixes=("", "_geo"))
if len(_m):
    _sw = _swap_focal(_m, _m.fx_geo)
    put("depthproGeocalibImages", _sw.name.nunique(), "{:d}")
    put("depthproGeocalibAlpha", A.per_image(_sw).alpha.median())
    put("depthproGeocalibScale", np.median(np.abs(_sw[np.isclose(_sw.zoom, 1.0)].log_ratio_gt)), "{:.3f}")

# zoom response by semantic region: objects of familiar size vs surfaces (paired per image, Wilcoxon)
reg_files = list(A.RESULTS.glob("*__regions.csv"))
reg = pd.concat([pd.read_csv(f) for f in reg_files], ignore_index=True) if reg_files else pd.DataFrame()
per_r, wide_r = A.region_alpha(reg) if len(reg) else (pd.DataFrame(), pd.DataFrame())
for m in ("dav2-s", "moge2-s", "depthpro"):
    w = wide_r[wide_r.model == m] if len(wide_r) else pd.DataFrame()
    put(key(m, "region", "object"), w.object.median() if len(w) else np.nan)
    put(key(m, "region", "surface"), w.surface.median() if len(w) else np.nan)
    put(key(m, "region", "diff"), w["diff"].median() if len(w) else np.nan)
    put(key(m, "region", "pairs"), len(w), "{:d}")
    put(key(m, "region", "p"), wilcoxon(w["diff"]).pvalue if len(w) > 10 else np.nan, "{:.2g}")
    put(key(m, "region", "objectLarger"), 100 * float((w["diff"] > 0).mean()) if len(w) else np.nan, "{:.0f}")

# UniDepthV2-S on DIODE-outdoor: how far its unzoomed metric scale is off (its pooled AbsRel is high)
u1 = df[(df.model == "unidepth2-s") & (df.dataset == "diode-out") & (df.kind == "zoom") & np.isclose(df.zoom, 1.0)
        & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)]
put("udDiodeOutScale", float(np.exp(np.median(u1.log_ratio_gt))) if len(u1) else np.nan)
put("udDiodeOutOverTwo", 100 * float((u1.log_ratio_gt > np.log(2)).mean()) if len(u1) else np.nan, "{:.0f}")

# ------------------------------------------------------------------ camera intervention: the same network, given the true K
# Depth Pro and DA3-Nested turn a camera-free network output into metres with their own focal: depth = f * net(I)
# (ml-depth-pro infer(); DA3 apply_metric_scaling). Given the true focal f*, the depth is exactly depth * f*/f, so
# their given-K run follows from the audit CSVs. MoGe-2 and UniDepthV2 use the camera differently and are rerun with
# it (models "-k"); they are compared with the plain run on the same images.
CAM = [("depthpro", None), ("da3-nested", None), ("moge2-s", "moge2-s-k"), ("unidepth2-s", "unidepth2-s-k")]
zc = df[(df.kind == "zoom") & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)]
rows = []
for m, mk in CAM:
    own = zc[zc.model == m]
    given = A.given_focal(own) if mk is None else zc[zc.model == mk]
    names_ = set(own.name) & set(given.name)
    own, given = own[own.name.isin(names_)], given[given.name.isin(names_)].assign(model=m + "+K")
    cells = []
    for tag, z in (("own", own), ("given", given)):
        p = A.per_image(z) if len(z) else pd.DataFrame(columns=["alpha"])
        one = z[np.isclose(z.zoom, 1.0)]
        put(key(m, "cam", tag, "alpha"), p.alpha.median() if len(p) else np.nan)
        put(key(m, "cam", tag, "scale"), np.median(np.abs(one.log_ratio_gt)) if len(one) else np.nan, "{:.3f}")
        cells += [macros[key(m, "cam", tag, "alpha")], macros[key(m, "cam", tag, "scale")]]
    put(key(m, "cam", "images"), len(names_), "{:d}")
    # the pooled medians above depend on the mix of datasets; per dataset, own vs true focal
    for tag, z in (("own", own), ("given", given)):
        for ds, g in z[np.isclose(z.zoom, 1.0)].groupby("dataset"):
            put(key(m, "cam", tag, "scale", ds), np.median(np.abs(g.log_ratio_gt)), "{:.3f}")
    how = "exact" if mk is None else "rerun"
    rows.append(f"{A.PRETTY.get(m, m)} & {how} & {len(names_)} & " + " & ".join(cells) + r" \\")
write_table("table_camera", [
    r"\begin{table}[t]\centering\small",
    r"\caption{The same camera-predicting network with its own focal and given the true (zoomed) camera. $\alpha$: depth "
    r"exponent. Scale: median over images of $|\operatorname{median}_p \log \hat D/D^{*}|$ at $1\times$ (0 = correct metric "
    r"scale). \emph{exact}: the model's metres are its own focal times a camera-free output, so the given-camera depth "
    r"follows exactly from the audit; \emph{rerun}: run again with the camera as input, on the same images as the "
    r"plain run.}\label{tab:camera}",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{llrcccc}\toprule",
    r"& & & \multicolumn{2}{c}{own focal} & \multicolumn{2}{c}{true camera} \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Model & how & images & $\alpha$ & scale & $\alpha$ & scale \\ \midrule",
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

# per-dataset ZoomFix table (supplement)
rows = []
for m in [x for x in MAIN if len(fix) and x in set(fix.model)]:
    cells = []
    for ds in ["kitti", "nyu", "diode-in", "diode-out"]:
        r = fix[(fix.model == m) & (fix.dataset == ds)]
        if len(r):
            r = r.iloc[0]
            cells.append(f"{fmt(r.exponent)} & {fmt(r.absrel_zoomed, 3)} & {fmt(r.absrel_fixed, 3)} & "
                         f"{fmt(r.absrel_unzoomed_same_pixels, 3)}")
        else:
            cells.append("-- & -- & -- & --")
    rows.append(f"{A.PRETTY[m]} & " + " & ".join(cells) + r" \\")
write_table("table_zoomfix", [
    r"\begin{table*}[t]\centering\scriptsize",
    r"\caption{ZoomFix per dataset at $3\times$, cross-fitted: the exponent $\hat e$ is measured without labels on "
    r"half of the images and the correction scored on the other half. AbsRel of the zoomed prediction as is, after "
    r"multiplying by $s^{\hat e}$, and of the unzoomed prediction on the same pixels (the target).}\label{tab:zoomfix}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{l cccc cccc cccc cccc}\toprule",
    r"& \multicolumn{4}{c}{KITTI} & \multicolumn{4}{c}{NYUv2} & \multicolumn{4}{c}{DIODE-in} & "
    r"\multicolumn{4}{c}{DIODE-out} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}",
    r"Model" + r" & $\hat e$ & as is & fixed & target" * 4 + r" \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ principal point under off-centre zooms
off = A.load("*__offc.csv") if list(A.RESULTS.glob("*__offc.csv")) else pd.DataFrame()
ppt = A.principal_point(off) if len(off) else pd.DataFrame()
for m in ["moge2-s", "unidepth2-s", "depthpro"]:
    g = ppt[(ppt.model == m) & (ppt.offset == "quadrant") & (ppt.zoom == 2.0)] if len(ppt) else pd.DataFrame()
    put(key(m, "ppQuadTwo"), g.model_error.median() if len(g) else np.nan, "{:.3f}")
    put(key(m, "ppGuessTwo"), g.centred_guess_error.median() if len(g) else np.nan, "{:.3f}")

# ------------------------------------------------------------------ fine-tuning (Table 2)
ft = df[df.model.isin(FT + SCALE_FT + SEEDS)]
sft = A.summary(A.per_image(ft)) if len(ft) else pd.DataFrame(columns=["model", "dataset"])
rows = []
for m in FT + SCALE_FT:
    for ds in A.DATASETS:
        r = sft[(sft.model == m) & (sft.dataset == ds)]
        get = (lambda c: r[c].iloc[0]) if len(r) else (lambda c: np.nan)
        put(key(m, ds, "alpha"), get("alpha"))
        put(key(m, ds, "absrelOne"), get("absrel1"), "{:.3f}")
        put(key(m, ds, "absrelThree"), get("absrel3"), "{:.3f}")
        bl = blur.get((m, ds), np.nan)
        put(key(m, ds, "blur"), 100 * (np.exp(bl) - 1) if np.isfinite(bl) else np.nan, "{:.1f}")
# seed spread of EFT+B: mean and sample sd over the training seeds that are present for every dataset
seeds = [m for m in SEEDS if (sft.model == m).sum() == len(A.DATASETS)]
put("eftbSeeds", len(seeds), "{:d}")
for ds in A.DATASETS:
    r = sft[sft.model.isin(seeds) & (sft.dataset == ds)] if len(seeds) == len(SEEDS) else sft.iloc[:0]
    for c, name, f in (("alpha", "alpha", "{:.2f}"), ("absrel1", "absrelOne", "{:.3f}")):
        put(key("eftbSeeds", ds, name, "mean"), r[c].mean() if len(r) else np.nan, f)
        put(key("eftbSeeds", ds, name, "sd"), r[c].std(ddof=1) if len(r) else np.nan, f)
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

# EFT-S against the frozen model and EFT+B (its own table: Table 2 is full)
SCALE_TAB = ["dav2-s-fixed", "dav2-s-eftb", "dav2-s-scale", "dav2-s-scalenb"]
names.update({"dav2-s-eftb": "EFT+B", "dav2-s-scale": "EFT-S (ours)", "dav2-s-scalenb": "EFT-S, no blur term"})
rows = []
for ds in A.DATASETS:
    cells = []
    for m in SCALE_TAB:
        cells += [macros[key(m, ds, "alpha")], macros[key(m, ds, "absrelOne")], macros[key(m, ds, "blur")] +
                  (r"\%" if macros[key(m, ds, "blur")] != "--" else "")]
    rows.append(A.PRETTY_DS[ds] + (r"$^\dagger$" if ds.startswith("diode") else "") + " & " + " & ".join(cells) + r" \\")
write_table("table_scale", [
    r"\begin{table*}[t]\centering\scriptsize",
    r"\caption{Scale-only equivariance tuning (EFT-S): DA-V2-S stays frozen and a small head learns one log-scale "
    r"correction per image from its tokens, with the EFT+B objective. $\alpha$, AbsRel on unzoomed images, and the depth "
    r"change under the blur control. $^\dagger$never used in training.}\label{tab:scale}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{l" + " ccc" * len(SCALE_TAB) + r"}\toprule",
    "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{names[m]}}}" for m in SCALE_TAB) + r" \\",
    "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(SCALE_TAB))),
    "Dataset" + r" & $\alpha$ & Abs$_{1\times}$ & blur" * len(SCALE_TAB) + r" \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ EFT-F: Depth Pro with its focal head tuned
# same images as the plain Depth Pro run; the depth network is the original, so alpha_K stays as it was
for m in ("depthpro", "depthpro-eftf"):
    z = zc[zc.model == m]
    p_ = A.per_image(z) if len(z) else pd.DataFrame(columns=["alpha", "beta", "dataset", "absrel1"])
    put(key(m, "fhead", "alpha"), p_.alpha.median() if len(p_) else np.nan)
    put(key(m, "fhead", "beta"), p_.beta.median() if len(p_) else np.nan)
    put(key(m, "fhead", "images"), len(p_), "{:d}")
    for ds in A.DATASETS:
        g = p_[p_.dataset == ds]
        put(key(m, "fhead", ds, "alpha"), g.alpha.median() if len(g) else np.nan)
        put(key(m, "fhead", ds, "absrelOne"), g.absrel1.mean() if len(g) else np.nan, "{:.3f}")
        put(key(m, "fhead", ds, "beta"), g.beta.median() if len(g) else np.nan)
        put(key(m, "fhead", ds, "absrelThree"), g.absrel3.mean() if len(g) else np.nan, "{:.3f}")
    one = z[np.isclose(z.zoom, 1.0)]
    put(key(m, "fhead", "scale"), np.median(np.abs(one.log_ratio_gt)) if len(one) else np.nan, "{:.3f}")
    put(key(m, "fhead", "focalErr"), float(np.exp(p_.fx_err1.median())) if len(p_) else np.nan)

eftf_log = ROOT / "outputs" / "eft" / "train_log_depthpro-eftf.csv"
if eftf_log.exists():  # held-out |log tan(fov/2) - target| of zoomed views before and after tuning
    lg = pd.read_csv(eftf_log)
    put("eftfValZoomBefore", lg.val_zoom.iloc[0], "{:.3f}")
    put("eftfValZoomAfter", lg.val_zoom.iloc[-1], "{:.3f}")
    put("eftfValPlainAfter", lg.val_plain.iloc[-1], "{:.3f}")
    put("eftfEpochs", int(lg.epoch.iloc[-1]), "{:d}")
write_table("table_eftf", [
    r"\begin{table}[t]\centering\small",
    r"\caption{EFT-F: Depth Pro with only its focal head tuned on unlabeled KITTI and NYUv2 training images (the "
    r"depth network is unchanged). Same 200 images per set; median $\alpha$, $\beta$, mean AbsRel at $1\times$ and "
    r"$3\times$; ideal $\alpha=0$, $\beta=1$.}\label{tab:eftf}",
    r"\footnotesize\setlength{\tabcolsep}{2pt}",
    r"\begin{tabular}{lcccccccc}\toprule",
    r"& \multicolumn{2}{c}{$\alpha$} & \multicolumn{2}{c}{$\beta$} & \multicolumn{2}{c}{Abs$_{1\times}$} & "
    r"\multicolumn{2}{c}{Abs$_{3\times}$} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
    r"Set & DP & EFT-F & DP & EFT-F & DP & EFT-F & DP & EFT-F \\ \midrule",
    *[f"{A.PRETTY_DS[ds]} & " + " & ".join(macros[key(m, "fhead", ds, q)] for q in ("alpha", "beta", "absrelOne", "absrelThree")
                                          for m in ("depthpro", "depthpro-eftf")) + r" \\" for ds in A.DATASETS],
    r"\bottomrule\end{tabular}\end{table}"])

# ------------------------------------------------------------------ complete test sets (check of the 200-image sample)
full_files = list(A.RESULTS.glob("*__full.csv"))
dfull = A.load("*__full.csv") if full_files else pd.DataFrame()
FULL = ["dav2-s", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s"]
rows = []
for ds in A.DATASETS:
    put(key("full", ds, "images"), np.nan)
if len(dfull):
    pf = A.per_image(dfull)
    shifts = []
    for m in [x for x in FULL if x in set(pf.model)]:
        put(key(m, "full", "images"), int((pf.model == m).sum()), "{:d}")
        cells = []
        for ds in A.DATASETS:
            g = pf[(pf.model == m) & (pf.dataset == ds)]
            a_sub = pi[(pi.model == m) & (pi.dataset == ds)].alpha.median()
            a_full = g.alpha.median() if len(g) else np.nan
            put(key(m, "full", ds, "alpha"), a_full)
            if np.isfinite(a_sub) and np.isfinite(a_full):
                shifts.append(abs(a_full - a_sub))
            cells.append(f"{fmt(a_sub)} / {fmt(a_full)}")
        rows.append(f"{A.PRETTY[m]} & " + " & ".join(cells) + r" \\")
    for ds in A.DATASETS:
        put(key("full", ds, "images"), int(pf[pf.dataset == ds].name.nunique()), "{:d}")
    # per dataset, not pooled: the complete sets differ in size (KITTI dominates), the 200-image sample does not
    put("fullMaxShift", max(shifts) if shifts else np.nan)
    put("fullMedianShift", float(np.median(shifts)) if shifts else np.nan)
write_table("table_full", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Depth exponent $\alpha$ on the 200-image sample per set / on the complete test sets (KITTI val "
    r"selection, NYUv2 test, DIODE val indoor and outdoor; \FullKittiImages, \FullNyuImages, \FullDiodeInImages{} and "
    r"\FullDiodeOutImages{} images), median over images.}\label{tab:full}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\resizebox{\linewidth}{!}{\begin{tabular}{lcccc}\toprule",
    "Model & " + " & ".join(A.PRETTY_DS[d] for d in A.DATASETS) + r" \\ \midrule",
    *rows, r"\bottomrule\end{tabular}}\end{table}"])

# ------------------------------------------------------------------ FocalUnshrink: inverting the focal head's shrinkage
# exact for models whose metres are their focal times a camera-free output; for MoGe-2 and UniDepthV2 the depth
# columns assume the same (an approximation), the focal columns are exact for every model
UNSHRINK = [("depthpro", "exact"), ("da3-nested", "exact"), ("moge2-s", "approx."), ("unidepth2-s", "approx."),
            ("geocalib", "focal only")]
us_df = df[df.model.isin([m for m, _ in UNSHRINK])]
us = A.deshrink_summary(A.focal_deshrink(us_df)) if len(us_df) else pd.DataFrame()
rows = []
for m, how in UNSHRINK:
    r = us[us.model == m]
    if not len(r):
        continue
    r = r.iloc[0]
    for c in ("beta", "alpha", "alpha_c", "scale1", "scale1_c", "scale3", "scale3_c", "scale_all", "scale_all_c",
              "focal1", "focal1_c", "focal3", "focal3_c", "focal_all", "focal_all_c"):
        put(key(m, "unshrink", c.replace("_c", "corr").replace("_", "")), r[c], "{:.3f}" if "scale" in c or
            "focal" in c else "{:.2f}")
    dep = how != "focal only"
    cells = [fmt(r.beta), f"{fmt(r.alpha)} $\\to$ {fmt(r.alpha_c)}" if dep else "--",
             *[f"{fmt(r[c], 3)} $\\to$ {fmt(r[c + '_c'], 3)}" if dep else "--" for c in ("scale1", "scale3", "scale_all")],
             f"{fmt(r.focal3, 3)} $\\to$ {fmt(r.focal3_c, 3)}"]
    rows.append(f"{A.PRETTY.get(m, m)} & {how} & " + " & ".join(cells) + r" \\")
write_table("table_unshrink", [
    r"\begin{table*}[t]\centering\small",
    r"\caption{FocalUnshrink: the focal head's shrinkage inverted without labels, training or the zoom factor, "
    r"$\log f_c=\mu+(\log\hat f-\mu)/\hat\beta$ with $\hat\beta$ from self-zoom (pooled over datasets) and $\mu$ the mean "
    r"unzoomed log focal of the target dataset, both measured on one half of the images and applied to the other "
    r"(cross-fitted). Before $\to$ after. Scale: median $|\log\hat D/D^{*}|$ on the crop at $1\times$, $3\times$ and "
    r"over all zooms; focal: median $|\log\hat f/f^{*}|$ at $3\times$. \emph{exact}: metres are the focal times a "
    r"camera-free output, so the corrected depth is exact; \emph{approx.}: the depth is rescaled as if it were.}"
    r"\label{tab:unshrink}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{llcccccc}\toprule",
    r"Model & depth & $\hat\beta$ & $\alpha$ & scale $1\times$ & scale $3\times$ & scale, all & focal $3\times$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ real optical zoom (SR-RAW)
sr_files = list(A.RESULTS.glob("*__srraw.csv"))
sr = pd.concat([pd.read_csv(f, dtype={"seq": str}) for f in sr_files], ignore_index=True) if sr_files else pd.DataFrame()
SR = ["dav2-s", "dav2-s-fixed", "dav2-s-eftb", "depthpro", "depthpro-eftf", "moge2-s", "unidepth2-s", "metric3d-s"]
rows = []
if len(sr):
    sa = A.srraw_alpha(sr)
    put("srrawSequences", int(sr.seq.nunique()), "{:d}")
    put("srrawShots", int((sr.view == "optical").groupby(sr.model).sum().max()), "{:d}")
    for m in [x for x in SR if x in set(sa.model)]:
        cells = []
        for v in ("optical", "digital"):
            g = sa[(sa.model == m) & (sa.view == v)]
            put(key(m, "srraw", v, "alpha"), g.alpha.median() if len(g) else np.nan)
            put(key(m, "srraw", v, "beta"), g.beta.median() if len(g) and g.beta.notna().any() else np.nan)
        w = sa[sa.model == m].pivot_table(index="seq", columns="view", values="alpha").dropna()
        put(key(m, "srraw", "seqs"), len(w), "{:d}")
        put(key(m, "srraw", "p"), wilcoxon(w.optical - w.digital).pvalue if len(w) > 10 else np.nan, "{:.2g}")
        cells = [macros[key(m, "srraw", v, q)] for q in ("alpha", "beta") for v in ("optical", "digital")]
        # dav2-s-fixed is the frozen model through the fine-tuning's fixed 392-px input, the reference for EFT+B
        name = "DA-V2-S, 392-px input" if m == "dav2-s-fixed" else A.PRETTY.get(m, m)
        rows.append(f"{name} & {len(w)} & " + " & ".join(cells) + r" \\")
write_table("table_srraw", [
    r"\begin{table}[t]\centering\small",
    r"\caption{Real optical zoom (SR-RAW~\cite{zhang2019zoom}): the same static scene shot at several focal lengths "
    r"($s\le3.2$), against the same zoom done digitally from the widest shot. Median over sequences; ideal "
    r"$\alpha=0$, $\beta=1$. The 392-px row is the frozen reference for EFT+B (same input size).}\label{tab:srraw}",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\begin{tabular}{lrcccc}\toprule",
    r"& & \multicolumn{2}{c}{$\alpha$} & \multicolumn{2}{c}{$\beta$} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"Model & seq. & optical & digital & optical & digital \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table}"])

# ------------------------------------------------------------------ one image change at a time (from the runs above)
# each column changes one factor of a 2x digital zoom; entries are depth factors on the same pixels (1 = unchanged)
FACT = [("dav2-s", "DA-V2-S"), ("depthpro", "Depth Pro"), ("moge2-s", "MoGe-2-S"), ("metric3d-s", r"Metric3D v2-S, true $K'$"),
        ("metric3d-s-staleK", r"Metric3D v2-S, stale $K$")]
rows = []
for m, name in FACT:
    a = pooled.alpha.get(m, np.nan) if "alpha" in pooled else np.nan
    put(key(m, "zoom", "two", "factor"), 2 ** (-a) if np.isfinite(a) else np.nan)
    cells = [macros.get(key(m, "zoom", "two", "factor"), "--")]
    for c in ("CueMagnify", "CueShrink", "Optical", "CueContext"):
        cells.append(macros.get(key(m) + c, "--"))
    blur = macros.get(key(m) + "Blur", "--")
    cells.append(blur + r"\%" if blur != "--" else "--")
    rows.append(f"{name} & " + " & ".join(cells) + r" \\")
write_table("table_factorial", [
    r"\begin{table*}[t]\centering\small",
    r"\caption{A $2\times$ digital zoom taken apart, one change at a time (numbers from the runs of \cref{tab:audit}, "
    r"\cref{tab:cues} and \cref{tab:controls}). Entries: factor by which depth changes on the same pixels (1 = "
    r"unchanged; apparent size alone: 0.5 for zoom and magnification, 2 for shrink). Zoom: $2^{-\alpha}$ from the "
    r"audit. Crop only: the crop at its own place and size, original $K$ (context removed). Magnification: zoom against "
    r"crop only. Shrink: the whole image $2\times$ smaller. Optical: the same field of view with real detail. Resize "
    r"only: the blur control (detail lost, field of view and camera unchanged), absolute change. Only the factors that "
    r"change the camera move depth; detail loss, compression and resampling do not.}\label{tab:factorial}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{lcccccc}\toprule",
    r"& \multicolumn{4}{c}{camera changes} & \multicolumn{2}{c}{camera fixed} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-7}",
    r"Model & zoom $2\times$ & magnification & shrink & optical $2\times$ & crop only & resize only \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ uncertainty of the exponents, small vs large zooms
UNC = ["dav2-s", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "da3-nested", "metric3d-s"]
unc = A.exponent_uncertainty(df[df.model.isin(UNC)]) if len(df) else pd.DataFrame()


def ci(r, q, d=2):
    return f"{fmt(r[q], d)} [{fmt(r[q + '_lo'], d)}, {fmt(r[q + '_hi'], d)}]" if np.isfinite(r[q]) else "--"


rows = []
for m in [x for x in UNC if x in set(unc.model)]:
    r = unc[unc.model == m].iloc[0]
    for q in ("alpha", "beta", "sum", "alpha_small", "alpha_large", "beta_small", "beta_large"):
        for suf, c in (("", q), ("Lo", q + "_lo"), ("Hi", q + "_hi")):
            put(key(m, "unc", q.replace("_", "-"), suf), r[c])
    rows.append(f"{A.PRETTY.get(m, m)} & {int(r.images)} & {ci(r, 'alpha')} & {ci(r, 'beta')} & {ci(r, 'sum')} & "
                f"{fmt(r.alpha_small)} / {fmt(r.alpha_large)} & {fmt(r.beta_small)} / {fmt(r.beta_large)} & "
                f"{fmt(r.alpha_small + r.beta_small)} / {fmt(r.alpha_large + r.beta_large)} \\\\")
write_table("table_uncert", [
    r"\begin{table*}[t]\centering\small",
    r"\caption{The exponents with 95\% intervals from a scene-level bootstrap (median over images, centred zooms), and "
    r"fitted separately on the small zooms ($s\le1.5$) and the large ones ($s\ge2$). For the camera-predicting models "
    r"$\alpha$ grows with the zoom because $\beta$ falls, while $\alpha+\beta$ stays close to 1 in both ranges.}"
    r"\label{tab:uncert}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrcccccc}\toprule",
    r"& & \multicolumn{3}{c}{all zooms} & \multicolumn{3}{c}{small / large zooms} \\ "
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
    r"Model & images & $\alpha$ & $\beta$ & $\alpha+\beta$ & $\alpha$ & $\beta$ & $\alpha+\beta$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

# ------------------------------------------------------------------ FocalUnshrink: sensitivity to beta, mu and the prior
rows = []
for m, how in UNSHRINK[:4]:
    d = df[df.model == m]
    if not len(d):
        continue
    b, blo, bhi = A.focal_beta_ci(d)
    res = {tag: A.deshrink_summary(A.focal_deshrink(d, beta=bb)).iloc[0] for tag, bb in (("lo", blo), ("hi", bhi))}
    res["mu"] = A.deshrink_summary(A.focal_deshrink(d, mu_shift=True)).iloc[0]
    base = us[us.model == m].iloc[0]
    for tag, v in (("betaCi", b), ("betaLo", blo), ("betaHi", bhi), ("alphaAtLo", res["lo"].alpha_c),
                   ("alphaAtHi", res["hi"].alpha_c), ("muShiftScaleOne", res["mu"].scale1_c),
                   ("muShiftAlpha", res["mu"].alpha_c)):
        put(key(m, "sens", tag), v, "{:.3f}" if "Scale" in tag else "{:.2f}")
    rows.append(f"{A.PRETTY.get(m, m)} & {fmt(b)} [{fmt(blo)}, {fmt(bhi)}] & {fmt(base.alpha)} & "
                f"{fmt(res['hi'].alpha_c)} / {fmt(base.alpha_c)} / {fmt(res['lo'].alpha_c)} & "
                f"{fmt(base.scale1, 3)} & {fmt(base.scale1_c, 3)} & {fmt(res['mu'].scale1_c, 3)} & "
                f"{fmt(res['mu'].alpha_c)} \\\\")
write_table("table_unshrink_sens", [
    r"\begin{table*}[t]\centering\small",
    r"\caption{Sensitivity of FocalUnshrink on real data. $\hat\beta$ with a 95\% scene-bootstrap interval; $\alpha$ "
    r"after the correction with $\hat\beta$ at the upper end of its interval, at its estimate and at the lower end. "
    r"Scale: median $|\log\hat D/D^{*}|$ at $1\times$, before, after, and after with a misspecified prior centre $\mu$ "
    r"taken from the \emph{other} datasets instead of the target one. $\mu$ does not enter $\alpha$ (it cancels in "
    r"the ratio of two zooms), so a wrong prior centre costs scale, not equivariance.}\label{tab:unshrink_sens}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lccccccc}\toprule",
    r"& & \multicolumn{2}{c}{$\alpha$} & \multicolumn{3}{c}{scale $1\times$} & \\ "
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
    r"Model & $\hat\beta$ [95\%] & before & after ($\hat\beta_{\text{hi}}$ / $\hat\beta$ / $\hat\beta_{\text{lo}}$) & "
    r"before & after & wrong $\mu$ & $\alpha$, wrong $\mu$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}\end{table*}"])

rows = []
for p in ("gaussian", "laplace", "uniform", "bimodal"):
    for rho in (0.65, 0.4):
        s_ = A.shrinkage_sim(p, rho)
        put(key("sim", p, "rho", str(int(rho * 100)), "zerrc"), s_["zerr_c"], "{:.2f}")
        put(key("sim", p, "rho", str(int(rho * 100)), "zerr"), s_["zerr"], "{:.2f}")
        rows.append(f"{p} & {rho:.2f} & {s_['beta']:.2f} & {s_['zerr']:.2f} $\\to$ {s_['zerr_c']:.2f} & "
                    f"{s_['err1']:.2f} $\\to$ {s_['err1_c']:.2f} & {s_['err3']:.2f} $\\to$ {s_['err3_c']:.2f} \\\\")
write_table("table_shrink_sim", [
    r"\begin{table}[t]\centering\small",
    r"\caption{FocalUnshrink when the camera prior is not Gaussian (simulation, \cref{sec:supp_unshrink}): $\log f$ "
    r"from a unit-variance prior, a noisy cue of Gaussian reliability $\rho$, the exact posterior-mean focal head. "
    r"Zoom error: per-image RMS of $\log\hat f_3-\log\hat f_1-\log 3$; error: RMS of $\log\hat f$ against the truth. "
    r"Before $\to$ after the correction.}\label{tab:shrink_sim}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\resizebox{\linewidth}{!}{\begin{tabular}{lccccc}\toprule",
    r"prior & $\rho$ & $\beta$ & zoom error & error $1\times$ & error $3\times$ \\ \midrule",
    *rows, r"\bottomrule\end{tabular}}\end{table}"])

(GEN / "numbers.tex").write_text("% generated by scripts/make_paper_numbers.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
missing = sum(v == "--" for v in macros.values())
print(f"{len(macros)} macros ({missing} still '--') -> {GEN / 'numbers.tex'}")
