"""Turn the audit CSVs into the paper's numbers.

Two exponents summarise a model's behaviour under a zoom by s (s = fx_zoomed / fx_original,
taken from the exact K, so rounding of the crop box is accounted for):

    depth:  D_s / D_1 ~ s ** (-alpha)      ideal alpha = 0, "apparent size only" alpha = 1
    focal:  f_s / f_1 ~ s ** beta          ideal beta  = 1, "ignores zoom"       beta  = 0

Both are fitted per image by least squares through the origin in log-log space, then
summarised by the median over images with a bootstrap confidence interval.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(os.environ.get("ZE_RESULTS", Path(__file__).resolve().parents[1] / "outputs" / "results"))

GROUP = {"dav2-l": "pixels only", "moge2-b": "predicts camera", "moge2-l": "predicts camera",
         "unidepth2-b": "predicts camera", "unidepth2-l": "predicts camera", "unik3d-l": "predicts camera",
         "da3-nested": "predicts camera", "da3-metric-l": "takes camera (true K)", "metric3d-l": "takes camera (true K)",
         "da3-l": "predicts camera (rel.)", "vggt": "predicts camera (rel.)",
         "dav2-s": "pixels only", "dav2-b": "pixels only", "zoedepth": "pixels only",
         "depthpro": "predicts camera", "moge2-s": "predicts camera", "unidepth2-s": "predicts camera",
         "metric3d-s": "takes camera (true K)", "metric3d-s-staleK": "takes camera (stale K)",
         "dav2-s-fixed": "pixels only", "dav2-s-eft": "pixels only + EFT",
         "dav2-s-anchor": "pixels only + anchor-only FT", "dav2-s-eftb": "pixels only + EFT+B"}
PRETTY = {"dav2-l": "DA-V2-L", "moge2-b": "MoGe-2-B", "moge2-l": "MoGe-2-L", "unidepth2-b": "UniDepthV2-B",
          "unidepth2-l": "UniDepthV2-L", "unik3d-l": "UniK3D-L", "da3-nested": "DA3-Nested", "da3-metric-l": "DA3-Metric-L",
          "metric3d-l": "Metric3D v2-L", "da3-l": "DA3-L", "vggt": "VGGT-1B", "dav2-s-zhong": "DA-V2-S + Zhong et al.-style",
          "dav2-s": "DA-V2-S", "dav2-b": "DA-V2-B", "zoedepth": "ZoeDepth", "depthpro": "Depth Pro",
          "moge2-s": "MoGe-2-S", "unidepth2-s": "UniDepthV2-S", "metric3d-s": "Metric3D v2-S",
          "metric3d-s-staleK": "Metric3D v2-S (stale K)", "dav2-s-fixed": "DA-V2-S (frozen)",
          "dav2-s-eft": "DA-V2-S + EFT", "dav2-s-anchor": "DA-V2-S + anchor-only FT",
          "dav2-s-eftb": "DA-V2-S + EFT+B (ours)", "dav2-s-blur": "DA-V2-S + blur-only FT",
          "dav2-s-eftb3": "DA-V2-S + EFT+B (anchor x3)", "dav2-s-scale": "DA-V2-S + EFT-S (ours)",
          "dav2-s-scalenb": "DA-V2-S + EFT-S, no blur term", "depthpro-eftf": "Depth Pro + EFT-F (ours)",
          "geocalib": "GeoCalib"}
DATASETS = ["kitti", "nyu", "diode-in", "diode-out"]
PRETTY_DS = {"kitti": "KITTI", "nyu": "NYUv2", "diode-in": "DIODE-in", "diode-out": "DIODE-out"}


CENTRES = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]


def repair_cx(df):
    """The first audit-a2 run wrote the predicted principal point into "cx", overwriting the
    crop centre. Rows are written in a fixed order (per image: centre zooms + blur row, then
    the four quadrants), so the crop centre can be restored exactly; cy was never touched."""
    if "cx_gt" not in df.columns or "pp_x" in df.columns:
        return df
    df = df.rename(columns={"cx": "pp_x", "cx_base": "pp_x_base", "cx_gt": "pp_x_gt", "cx_gt_base": "pp_x_gt_base"})
    cx = np.full(len(df), np.nan)
    for _, idx in df.groupby("name", sort=False).groups.items():
        k = 0
        for i in idx:
            r = df.loc[i]
            if r.kind == "blur":
                cx[df.index.get_loc(i)] = 0.5
                continue
            cx[df.index.get_loc(i)] = CENTRES[k // len(ZOOMS)][0]
            k += 1
    df["cx"] = cx
    return df


ZOOMS = (1.0, 1.25, 1.5, 2.0, 3.0)


def load(pattern="*__*.csv", folder=None):
    folder = Path(folder or RESULTS)
    # side runs (off-centre, controls, fine ZoomFix, cue split) share model names with the main run, so they
    # are only loaded when asked for by name, never mixed into the main audit
    side = ("__offc", "__ctrl", "__fix", "__cues", "__regions", "__full", "__srraw", "__jz")
    files = [f for f in sorted(folder.glob(pattern)) if not any(t in f.name and t[2:] not in pattern for t in side)]
    df = pd.concat([repair_cx(pd.read_csv(f)) for f in files], ignore_index=True)
    first = df[df.kind == "zoom"].groupby(["model", "dataset", "name", "cx", "cy"])["fx_gt"].transform("min")
    df.loc[df.kind == "zoom", "s_true"] = df.loc[df.kind == "zoom", "fx_gt"] / first
    return df


def slope_through_origin(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return float((x * y).sum() / (x * x).sum()) if len(x) >= 2 else np.nan


KEY = ["model", "dataset", "name", "cx", "cy"]


def _slope_by_group(x, y, keys):
    """Least-squares slope through the origin of y on x within each group, ignoring
    non-finite pairs; NaN for groups with fewer than two usable points."""
    ok = np.isfinite(x) & np.isfinite(y)
    parts = pd.DataFrame({"xy": np.where(ok, x * y, 0.0), "xx": np.where(ok, x * x, 0.0), "n": ok})
    g = parts.groupby([k.values for k in keys]).sum()
    return (g.xy / g.xx).where(g.n >= 2)


def per_image(df):
    """One row per (model, dataset, image, centre) with alpha, beta and a few errors.

    Vectorised: the same numbers as fitting each image in a loop (tests/test_analysis.py
    checks this against per_image_loop), about 100x faster on the full audit.
    """
    z = df[df.kind == "zoom"].copy()
    z["zoom"] = z.zoom.round(4)
    z["log_s"] = np.log(z.s_true)
    zz = z[z.zoom > 1]
    keys = [zz[k] for k in KEY]

    out = pd.DataFrame(index=z.groupby(KEY).size().index)
    out["alpha"] = -_slope_by_group(zz.log_s.values, zz.log_ratio_self.values, keys)
    log_f = np.log(zz.fx.astype(float) / zz.fx_base.astype(float)).values
    has_fx = zz.fx.notna().groupby(keys).all()
    out["beta"] = _slope_by_group(zz.log_s.values, log_f, keys).where(has_fx)

    def at(s, col):
        if col not in z:
            return np.nan
        return z[z.zoom == s].set_index(KEY)[col].astype(float).groupby(level=list(range(len(KEY)))).first()

    out["die2"] = at(2.0, "log_ratio_self").abs()
    for s, tag in ((1.0, "1"), (2.0, "2"), (3.0, "3")):
        out[f"absrel{tag}"] = at(s, "abs_rel")
        out[f"d1_{tag}"] = at(s, "delta1")
    # unzoomed prediction on the same crop pixels (NaN for runs before this column)
    out["base2"], out["base3"] = at(2.0, "abs_rel_base"), at(3.0, "abs_rel_base")
    out["scale1"], out["scale2"] = at(1.0, "log_ratio_gt"), at(2.0, "log_ratio_gt")
    all_fx = z.fx.notna().groupby([z[k] for k in KEY]).all()
    out["fx_err1"] = np.abs(np.log(at(1.0, "fx") / at(1.0, "fx_gt"))).where(all_fx)
    # training-free correction D * s**e (only in runs that have these columns)
    for e in (0.25, 0.5, 0.75, 1.0):
        for s in (2.0, 3.0):
            out[f"fix{int(s)}_{e}"] = at(s, f"abs_rel_fix{e}")
    return out.reset_index()


def per_image_loop(df):
    """Reference implementation of per_image (slow); kept for the equivalence test."""
    z = df[(df.kind == "zoom")].copy()
    z["log_s"] = np.log(z["s_true"])
    rows = []
    for key, g in z.groupby(["model", "dataset", "name", "cx", "cy"]):
        g = g.sort_values("zoom")
        zoomed = g[g.zoom > 1]
        alpha = -slope_through_origin(zoomed.log_s.values, zoomed.log_ratio_self.values)
        beta = np.nan
        if zoomed.fx.notna().all() and len(zoomed):
            beta = slope_through_origin(zoomed.log_s.values, np.log(zoomed.fx.values / zoomed.fx_base.values))
        at = lambda s, col: float(g.loc[np.isclose(g.zoom, s), col].iloc[0]) if np.isclose(g.zoom, s).any() else np.nan  # noqa: E731
        rows.append(dict(model=key[0], dataset=key[1], name=key[2], cx=key[3], cy=key[4], alpha=alpha, beta=beta,
                         die2=abs(at(2.0, "log_ratio_self")),
                         absrel1=at(1.0, "abs_rel"), absrel2=at(2.0, "abs_rel"), absrel3=at(3.0, "abs_rel"),
                         d1_1=at(1.0, "delta1"), d1_2=at(2.0, "delta1"), d1_3=at(3.0, "delta1"),
                         # unzoomed prediction on the same crop pixels (NaN for runs before this column)
                         base2=at(2.0, "abs_rel_base") if "abs_rel_base" in g else np.nan,
                         base3=at(3.0, "abs_rel_base") if "abs_rel_base" in g else np.nan,
                         scale1=at(1.0, "log_ratio_gt"), scale2=at(2.0, "log_ratio_gt"),
                         fx_err1=abs(np.log(at(1.0, "fx") / at(1.0, "fx_gt"))) if g.fx.notna().all() else np.nan))
    return pd.DataFrame(rows)


def blur_control(df):
    b = df[df.kind == "blur"]
    return b.groupby(["model", "dataset"])["log_ratio_self"].apply(lambda v: float(np.median(np.abs(v)))).rename("blur2")


_NYU_SCENES = None


def scene_of(name):
    """Scene id of an image, for scene-level resampling: frames of one KITTI drive, one
    DIODE scene, or one NYUv2 scene are not independent samples."""
    global _NYU_SCENES
    ds, stem = name.split("/", 1)
    if ds == "kitti":
        return "kitti/" + stem.split("_sync")[0]
    if ds.startswith("diode"):
        return f"{ds}/{stem.split('_')[0]}"
    if ds == "nyu":
        if _NYU_SCENES is None:
            f = RESULTS / "nyu_scenes.csv"
            _NYU_SCENES = dict(pd.read_csv(f).values) if f.exists() else {}
        return "nyu/" + _NYU_SCENES.get(name, stem)
    return name


def boot_median_scene(values, names, n=2000, seed=0):
    """Median with a 95% CI from a cluster bootstrap: whole scenes are resampled."""
    d = pd.DataFrame({"v": np.asarray(values, float), "scene": [scene_of(x) for x in names]}).dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    groups = [g.v.values for _, g in d.groupby("scene")]
    rng = np.random.default_rng(seed)
    meds = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return float(np.median(d.v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def boot_median(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summary(pi, by=("model", "dataset"), scene_ci=False):
    """scene_ci: also a cluster-bootstrap CI over scenes (slower; used for the paper tables)."""
    with warnings.catch_warnings():
        # columns that a run does not have (e.g. beta for pixels-only models) are all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return _summary(pi, by, scene_ci)


def _summary(pi, by, scene_ci):
    rows = []
    for key, g in pi.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        a, alo, ahi = boot_median(g.alpha)
        _, salo, sahi = boot_median_scene(g.alpha, g.name, n=500) if scene_ci else (np.nan, np.nan, np.nan)
        b, blo, bhi = boot_median(g.beta)
        rows.append(dict(zip(by, key), n=len(g), n_scenes=g.name.map(scene_of).nunique() if scene_ci else np.nan,
                         alpha=a, alpha_lo=alo, alpha_hi=ahi, alpha_lo_scene=salo, alpha_hi_scene=sahi,
                         beta=b, beta_lo=blo, beta_hi=bhi, die2=np.nanmedian(g.die2),
                         absrel1=np.nanmean(g.absrel1), absrel2=np.nanmean(g.absrel2), absrel3=np.nanmean(g.absrel3),
                         base2=np.nanmean(g.base2), base3=np.nanmean(g.base3),
                         d1_1=np.nanmean(g.d1_1), d1_2=np.nanmean(g.d1_2), d1_3=np.nanmean(g.d1_3),
                         fx_err1=np.nanmedian(g.fx_err1)))
    return pd.DataFrame(rows)


FIX_EXPONENTS = (0.25, 0.5, 0.75, 1.0)


def zoomfix(pi, s=3):
    """Training-free correction when the zoom factor is known (e.g. EXIF DigitalZoomRatio):
    multiply the depth by s**e. The exponent e is chosen leave-one-dataset-out, i.e. on the
    other three datasets, so the dataset it is scored on never influences the choice.

    Returns one row per (model, held-out dataset): chosen e, AbsRel at zoom s without the fix,
    with the fix, and of the unzoomed prediction on the same pixels (the target to reach)."""
    cols = [f"fix{s}_{e}" for e in FIX_EXPONENTS]
    have = pi.dropna(subset=cols)
    rows = []
    for model, g in have.groupby("model"):
        per_ds = g.groupby("dataset")[cols + [f"absrel{s}", f"base{s}"]].mean()
        for ds in per_ds.index:
            others = per_ds.drop(index=ds)
            if others.empty:
                continue
            best = others[cols].mean().idxmin()
            rows.append(dict(model=model, dataset=ds, exponent=float(best.split("_")[1]),
                             absrel_zoomed=per_ds.loc[ds, f"absrel{s}"], absrel_fixed=per_ds.loc[ds, best],
                             absrel_unzoomed_same_pixels=per_ds.loc[ds, f"base{s}"]))
    return pd.DataFrame(rows)


def zoomfix_crossfit(df, s=3.0, seed=0):
    """ZoomFix with a label-free exponent from the *same* domain: split each dataset's images in
    two halves; on one half measure alpha (no labels), round it to the nearest exponent that was
    evaluated, and score D * s**alpha on the other half; then swap. Uses the rows of a run made with
    --fine-fix (columns abs_rel_fix<e>)."""
    z = df[(df.kind == "zoom") & np.isclose(df.zoom, s)]
    fix_cols = sorted([c for c in z.columns if c.startswith("abs_rel_fix") and z[c].notna().any()],
                      key=lambda c: float(c[len("abs_rel_fix"):]))
    if not fix_cols:
        return pd.DataFrame()
    col_of = {float(c[len("abs_rel_fix"):]): c for c in fix_cols}
    grid = np.array(sorted(col_of))
    pi = per_image(df).set_index(KEY[:3])
    rows = []
    rng = np.random.default_rng(seed)
    for (model, dataset), g in z.groupby(["model", "dataset"]):
        names = g.name.unique()
        half = set(rng.permutation(names)[: len(names) // 2])
        for fit, test in ((half, set(names) - half), (set(names) - half, half)):
            a = pi.loc[[(model, dataset, n) for n in fit if (model, dataset, n) in pi.index], "alpha"].median()
            e = grid[np.argmin(np.abs(grid - a))]
            t = g[g.name.isin(test)]
            rows.append(dict(model=model, dataset=dataset, alpha_fit=a, exponent=e,
                             absrel_zoomed=t.abs_rel.mean(), absrel_fixed=t[col_of[e]].mean(),
                             absrel_unzoomed_same_pixels=t.abs_rel_base.mean()))
    return pd.DataFrame(rows).groupby(["model", "dataset"]).mean(numeric_only=True).reset_index()


def principal_point(df):
    """Off-centre zooms: where the model puts the principal point vs where the law puts it,
    in units of the zoomed image width. Models that assume a centred principal point show
    an error that grows with the crop offset."""
    z = df[(df.kind == "zoom") & (df.zoom > 1)].copy()
    if "pp_x" not in z or z.pp_x.isna().all():
        return pd.DataFrame()
    z = z.dropna(subset=["pp_x"])
    width = 2 * z.pp_x_gt_base  # cx of the unzoomed camera is about half the image width
    z["pp_err"] = (z.pp_x - z.pp_x_gt) / width
    z["pp_centre_err"] = (width / 2 - z.pp_x_gt) / width  # error of "always the image centre"
    z["offset"] = np.where((z.cx == 0.5) & (z.cy == 0.5), "centre", "quadrant")
    return z.groupby(["model", "dataset", "offset", "zoom"]).agg(
        model_error=("pp_err", lambda v: float(np.median(np.abs(v)))),
        centred_guess_error=("pp_centre_err", lambda v: float(np.median(np.abs(v))))).reset_index()


def given_focal(z):
    """Zoom rows of a model whose metres are its own focal times a camera-free output (Depth Pro,
    DA3-Nested), turned into the rows it would give with the true focal: depth * fx_gt / fx.

    Exact for such models, so their "given the camera" run needs no rerun. Rows must be one
    crop centre per image (the 1x row of each image is the reference)."""
    z = z.copy()
    corr = np.log(z.fx_gt / z.fx)
    corr1 = z[np.isclose(z.zoom, 1.0)].set_index("name").pipe(lambda b: np.log(b.fx_gt / b.fx))
    z["log_ratio_self"] = z.log_ratio_self + corr - z.name.map(corr1).values
    z["log_ratio_gt"] = z.log_ratio_gt + corr
    z["fx"], z["fx_base"] = z.fx_gt, z.groupby("name").fx_gt.transform("min")
    return z


def twoway_boot_spearman(pairs, x, y, n=2000, seed=0):
    """Spearman rho over (model, dataset) pairs with a 95% interval that resamples models and datasets
    together: pairs that share a model or a dataset are not independent, so resampling pairs is too narrow."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    models, sets = pairs.model.unique(), pairs.dataset.unique()
    by = pairs.set_index(["model", "dataset"])
    rho = spearmanr(pairs[x], pairs[y]).statistic
    reps = []
    for _ in range(n):
        ms, ds = rng.choice(models, len(models)), rng.choice(sets, len(sets))
        idx = [(m, d) for m in ms for d in ds if (m, d) in by.index]
        b = by.loc[idx]
        if len(b) > 4 and b[x].nunique() > 1 and b[y].nunique() > 1:
            reps.append(spearmanr(b[x], b[y]).statistic)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return rho, lo, hi


def fit_quality(df):
    """How well the per-image power law describes the depth ratio: per image, the RMS residual of
    log(depth ratio) around -alpha*log(s), and alpha fitted on the small zooms (s<=1.5) and the large
    ones (s>=2) separately. One row per (model, dataset, image) at the centre."""
    z = df[(df.kind == "zoom") & (df.zoom > 1) & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)].copy()
    z["log_s"] = np.log(z.s_true)
    keys = ["model", "dataset", "name"]

    def slope(g):
        return -float((g.log_s * g.log_ratio_self).sum() / (g.log_s ** 2).sum())

    rows = []
    for k, g in z.groupby(keys):
        g = g.dropna(subset=["log_ratio_self"])
        if len(g) < 4:
            continue
        a = slope(g)
        rms = float(np.sqrt(np.mean((g.log_ratio_self + a * g.log_s) ** 2)))
        rows.append(dict(zip(keys, k), alpha=a, rms=rms, alpha_small=slope(g[g.zoom <= 1.5]),
                         alpha_large=slope(g[g.zoom >= 2])))
    return pd.DataFrame(rows)


def region_alpha(reg):
    """Per image and region group, the depth exponent from the 2x and 3x zooms (fit through the origin),
    and per image the paired difference object - surface (images with both groups only)."""
    reg = reg.dropna(subset=["log_ratio_self"]).assign(log_s=lambda d: np.log(d.s_true))
    keys = ["model", "dataset", "name", "group"]
    g = reg.groupby(keys)
    a = -(g.apply(lambda d: (d.log_s * d.log_ratio_self).sum() / (d.log_s ** 2).sum(), include_groups=False)
          .rename("alpha"))
    n = g.size().rename("n_zooms")
    per = pd.concat([a, n], axis=1).reset_index()
    per = per[per.n_zooms == reg.zoom.nunique()]
    wide = per.pivot_table(index=["model", "dataset", "name"], columns="group", values="alpha").dropna()
    wide["diff"] = wide["object"] - wide["surface"]
    return per, wide.reset_index()


def srraw_alpha(df):
    """Real optical vs digital zoom on SR-RAW: per model, sequence and view, alpha from the log depth ratios and
    beta from the focal ratios (both fit through the origin over the zoomed shots)."""
    z = df[df.view != "base"].assign(log_s=lambda d: np.log(d.s),
                                     log_f=lambda d: np.log(d.fx.astype(float) / d.fx_base.astype(float)))
    keys = [z.model, z.seq, z.view]
    a = -_slope_by_group(z.log_s.values, z.log_ratio.values, keys)
    b = _slope_by_group(z.log_s.values, z.log_f.values, keys)
    out = pd.concat([a.rename("alpha"), b.rename("beta")], axis=1)
    out.index.names = ["model", "seq", "view"]
    return out.reset_index()


def merge_shards(folder=None, remove=False):
    """Merge the per-shard CSVs of a sharded audit (run_audit.py --shard i --shards k --out-tag TAG-si), named
    model__set__TAG-sI.csv, into model__set__TAG.csv. A shard must not repeat another shard's images.
    Returns {merged file name: number of images}."""
    folder = Path(folder or RESULTS)
    groups = {}
    for f in sorted(folder.glob("*__*-s[0-9]*.csv")):
        base, _, k = f.stem.rpartition("-s")
        if k.isdigit() and "__" in base:
            groups.setdefault(base, []).append(f)
    merged = {}
    for base, fs in groups.items():
        parts = [pd.read_csv(f) for f in fs]
        names = [set(p["name"]) for p in parts]
        overlap = set.union(*[a & b for i, a in enumerate(names) for b in names[i + 1:]] or [set()])
        if overlap:
            raise ValueError(f"{base}: {len(overlap)} images in more than one shard, e.g. {sorted(overlap)[0]}")
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(folder / f"{base}.csv", index=False)
        merged[f"{base}.csv"] = df["name"].nunique()
        if remove:
            for f in fs:
                f.unlink()
    return merged


def focal_beta(z):
    """Label-free focal exponent of a set of zoom rows: least-squares slope through the origin of
    log(f_s / f_1) on log s, pooled over images (only the model's own focal, no ground truth)."""
    z = z[(z.kind == "zoom") & z.fx.notna()] if "kind" in z else z[z.fx.notna()]
    f1 = z[np.isclose(z.zoom, 1.0)].set_index(["model", "dataset", "name"]).fx
    r = z[z.zoom > 1]
    f1 = f1.reindex(pd.MultiIndex.from_frame(r[["model", "dataset", "name"]])).values
    s = r.s_true.values if "s_true" in r else r.zoom.values
    return slope_through_origin(np.log(s), np.log(r.fx.values / f1))


def exponent_uncertainty(df, n=500, seed=0):
    """Per model, with 95% scene-bootstrap intervals (median over images): alpha, beta and alpha+beta over all
    zooms, and alpha and beta fitted on the small zooms (s<=1.5) and the large ones (s>=2) separately.
    Centred crops only. Returns one row per model; each quantity q has columns q, q_lo, q_hi."""
    z = df[(df.kind == "zoom") & (df.zoom > 1) & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)].copy()
    z["log_s"] = np.log(z.s_true)
    z["log_f"] = np.log(z.fx.astype(float) / z.fx_base.astype(float))
    keys = [z[k] for k in ("model", "name")]
    out = pd.DataFrame(index=z.groupby(["model", "name"]).size().index)
    for tag, m in (("", z.zoom > 0), ("_small", z.zoom <= 1.5), ("_large", z.zoom >= 2)):
        zz, kk = z[m], [k[m] for k in keys]
        out["alpha" + tag] = -_slope_by_group(zz.log_s.values, zz.log_ratio_self.values, kk)
        has_fx = zz.fx.notna().groupby(kk).all()
        out["beta" + tag] = _slope_by_group(zz.log_s.values, zz.log_f.values, kk).where(has_fx)
    out["sum"] = out.alpha + out.beta
    out = out.reset_index()
    rows = []
    for model, g in out.groupby("model"):
        row = dict(model=model, images=len(g))
        for q in ("alpha", "beta", "sum", "alpha_small", "alpha_large", "beta_small", "beta_large"):
            ok = g[q].notna()
            row[q], row[q + "_lo"], row[q + "_hi"] = (boot_median_scene(g[q][ok], g.name[ok], n=n, seed=seed)
                                                      if ok.sum() >= 10 else (np.nan,) * 3)
        rows.append(row)
    return pd.DataFrame(rows)


def focal_deshrink(df, seed=0, beta=None, mu_shift=False):
    """Training-free, label-free correction of a focal head that under-scales under zoom (FocalUnshrink).

    A focal head with exponent beta < 1 behaves like a shrinkage estimator, log f_hat = mu + beta (log f - mu) + noise,
    pulled towards a prior centre mu. Both are measured without labels on unlabeled images: beta by self-zoom
    (pooled over all domains, so it is stable) and mu as the mean log focal on unzoomed images of the target
    domain. The correction inverts the shrinkage, log f_c = mu + (log f_hat - mu) / beta, and needs neither the zoom
    factor nor the camera. For models whose metres are their own focal times a camera-free output (Depth Pro,
    DA3-Nested) the corrected depth is exactly D_hat * f_c / f_hat.

    Cross-fitted: in each dataset, beta and mu come from one half of the images and are applied to the other half,
    then the halves swap. Returns the zoom rows (centred crops) with columns log_fc (corrected log focal) and
    log_corr (log f_c - log f_hat).

    Sensitivity options: beta fixes the exponent instead of measuring it (e.g. at the ends of its interval);
    mu_shift takes mu from the other datasets instead of the target one (a misspecified prior centre)."""
    z = df[(df.kind == "zoom") & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5) & df.fx.notna()].copy()
    rng = np.random.default_rng(seed)
    out = []
    for model, zm in z.groupby("model"):
        names = zm.name.unique()
        half = set(rng.permutation(names)[: len(names) // 2])
        for fit in (half, set(names) - half):
            cal, ev = zm[zm.name.isin(fit)], zm[~zm.name.isin(fit)].copy()
            b = focal_beta(cal) if beta is None else beta
            for dataset, g in ev.groupby("dataset"):
                same = cal.dataset != dataset if mu_shift else cal.dataset == dataset
                c = cal[same & np.isclose(cal.zoom, 1.0)]
                mu = np.log(c.fx).mean()
                ev.loc[g.index, "log_fc"] = mu + (np.log(g.fx) - mu) / b
                ev.loc[g.index, "beta_fit"] = b
            out.append(ev)
    if not out:
        return pd.DataFrame()
    r = pd.concat(out)
    r["log_corr"] = r.log_fc - np.log(r.fx)
    return r


def focal_beta_ci(df, n=500, seed=0):
    """95% scene-bootstrap interval of the pooled focal exponent (focal_beta) of one model's zoom rows."""
    z = df[(df.kind == "zoom") & np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5) & df.fx.notna()]
    scenes = z.name.map(scene_of)
    groups = [g for _, g in z.groupby(scenes.values)]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        # resampled copies of one scene get distinct names, so focal_beta pairs each zoom with its own 1x row
        parts = [groups[i].assign(name=groups[i].name + f"#{j}") for j, i in enumerate(pick)]
        bs.append(focal_beta(pd.concat(parts)))
    return focal_beta(z), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def shrinkage_sim(prior, rho, n=4000, zooms=(1.25, 1.5, 2.0, 3.0), seed=0):
    """FocalUnshrink when the camera prior is not Gaussian. log f is drawn from `prior` (unit variance), the cue
    is x = log f + noise with noise variance set so that the Gaussian reliability would be rho, and the focal head is
    the exact posterior mean E[log f | x] (squared-error optimum, computed on a grid). A zoom adds log s to x.
    Returns the head's pooled beta; the per-image zoom error at the largest zoom, RMS of
    log f_s - log f_1 - log s, before and after FocalUnshrink (beta and mu measured as on real data; exactly zero after
    it only for a Gaussian prior); and the RMS error of log focal against the truth at 1x and at the largest zoom."""
    rng = np.random.default_rng(seed)
    sig = np.sqrt(1 / rho - 1)
    grid = np.linspace(-8, 8, 4001)
    if prior == "gaussian":
        dens, draw = np.exp(-grid ** 2 / 2), lambda k: rng.normal(size=k)
    elif prior == "laplace":
        b = 1 / np.sqrt(2)
        dens, draw = np.exp(-np.abs(grid) / b), lambda k: rng.laplace(scale=b, size=k)
    elif prior == "uniform":
        h = np.sqrt(3)
        dens, draw = (np.abs(grid) <= h).astype(float), lambda k: rng.uniform(-h, h, size=k)
    elif prior == "bimodal":  # two camera families (e.g. phones and telephoto), unit variance overall
        m, sd = 0.9, np.sqrt(1 - 0.81)
        dens = np.exp(-(grid - m) ** 2 / (2 * sd ** 2)) + np.exp(-(grid + m) ** 2 / (2 * sd ** 2))
        draw = lambda k: rng.choice([-m, m], size=k) + rng.normal(scale=sd, size=k)  # noqa: E731
    else:
        raise ValueError(prior)

    def head(x):
        w = dens[None, :] * np.exp(-(x[:, None] - grid[None, :]) ** 2 / (2 * sig ** 2))
        return (w * grid).sum(1) / w.sum(1)

    lf = draw(n)
    x = lf + rng.normal(scale=sig, size=lf.size)
    h1 = head(x)
    ls = np.log(np.array(zooms))
    hs = np.stack([head(x + v) for v in ls], 1)
    beta = float((ls[None] * (hs - h1[:, None])).sum() / (lf.size * (ls ** 2).sum()))
    mu = h1.mean()
    c1, cs = mu + (h1 - mu) / beta, mu + (hs - mu) / beta
    zerr = float(np.sqrt(np.mean((hs[:, -1] - h1 - ls[-1]) ** 2)))
    zerr_c = float(np.sqrt(np.mean((cs[:, -1] - c1 - ls[-1]) ** 2)))
    return dict(prior=prior, rho=rho, beta=beta, zerr=zerr, zerr_c=zerr_c, err1=float(np.sqrt(np.mean((h1 - lf) ** 2))),
                err1_c=float(np.sqrt(np.mean((c1 - lf) ** 2))),
                err3=float(np.sqrt(np.mean((hs[:, -1] - lf - ls[-1]) ** 2))),
                err3_c=float(np.sqrt(np.mean((cs[:, -1] - lf - ls[-1]) ** 2))))


def deshrink_summary(r):
    """Per model: alpha, scale error |median log D/D*| at 1x and 3x and over all zooms, and focal error
    |log f/f*| at 1x and 3x, before and after FocalUnshrink (the depth columns assume D proportional to f)."""
    r = r.copy()
    base = r[np.isclose(r.zoom, 1.0)].set_index(["model", "name"]).log_corr
    r["self_c"] = r.log_ratio_self + r.log_corr - base.reindex(pd.MultiIndex.from_frame(r[["model", "name"]])).values
    r["gt_c"] = r.log_ratio_gt + r.log_corr
    r["ls"] = np.log(r.s_true if "s_true" in r else r.zoom)
    rows = []
    for model, g in r.groupby("model"):
        zz = g[g.zoom > 1]
        sl = _slope_by_group(zz.ls.values, zz.log_ratio_self.values, [zz.name])
        sc = _slope_by_group(zz.ls.values, zz.self_c.values, [zz.name])
        one, three = g[np.isclose(g.zoom, 1.0)], g[np.isclose(g.zoom, 3.0)]
        lfg = np.log(g.fx_gt)
        rows.append(dict(
            model=model, images=g.name.nunique(), beta=g.beta_fit.mean(),
            alpha=float((-sl).median()), alpha_c=float((-sc).median()),
            scale1=one.log_ratio_gt.abs().median(), scale1_c=one.gt_c.abs().median(),
            scale3=three.log_ratio_gt.abs().median(), scale3_c=three.gt_c.abs().median(),
            scale_all=g.log_ratio_gt.abs().median(), scale_all_c=g.gt_c.abs().median(),
            focal1=(np.log(one.fx) - np.log(one.fx_gt)).abs().median(),
            focal1_c=(one.log_fc - np.log(one.fx_gt)).abs().median(),
            focal3=(np.log(three.fx) - np.log(three.fx_gt)).abs().median(),
            focal3_c=(three.log_fc - np.log(three.fx_gt)).abs().median(),
            focal_all=(np.log(g.fx) - lfg).abs().median(), focal_all_c=(g.log_fc - lfg).abs().median()))
    return pd.DataFrame(rows)
