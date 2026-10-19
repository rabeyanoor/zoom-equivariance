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
          "dav2-s-eftb3": "DA-V2-S + EFT+B (anchor x3)"}
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
    # side runs (off-centre, controls, fine ZoomFix) share model names with the main run, so they
    # are only loaded when asked for by name, never mixed into the main audit
    side = ("__offc", "__ctrl", "__fix")
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
