"""Turn the audit CSVs into the paper's numbers.

Two exponents summarise a model's behaviour under a zoom by s (s = fx_zoomed / fx_original,
taken from the exact K, so rounding of the crop box is accounted for):

    depth:  D_s / D_1 ~ s ** (-alpha)      ideal alpha = 0, "apparent size only" alpha = 1
    focal:  f_s / f_1 ~ s ** beta          ideal beta  = 1, "ignores zoom"       beta  = 0

Both are fitted per image by least squares through the origin in log-log space, then
summarised by the median over images with a bootstrap confidence interval.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import os

RESULTS = Path(os.environ.get("ZE_RESULTS", Path(__file__).resolve().parents[1] / "outputs" / "results"))

GROUP = {"dav2-s": "pixels only", "dav2-b": "pixels only", "zoedepth": "pixels only",
         "depthpro": "predicts camera", "moge2-s": "predicts camera", "unidepth2-s": "predicts camera",
         "metric3d-s": "takes camera (true K)", "metric3d-s-staleK": "takes camera (stale K)",
         "dav2-s-fixed": "pixels only", "dav2-s-eft": "pixels only + EFT",
         "dav2-s-anchor": "pixels only + anchor-only FT", "dav2-s-eftb": "pixels only + EFT+B"}
PRETTY = {"dav2-s": "DA-V2-S", "dav2-b": "DA-V2-B", "zoedepth": "ZoeDepth", "depthpro": "Depth Pro",
          "moge2-s": "MoGe-2-S", "unidepth2-s": "UniDepthV2-S", "metric3d-s": "Metric3D v2-S",
          "metric3d-s-staleK": "Metric3D v2-S (stale K)", "dav2-s-fixed": "DA-V2-S (frozen)",
          "dav2-s-eft": "DA-V2-S + EFT", "dav2-s-anchor": "DA-V2-S + anchor-only FT",
          "dav2-s-eftb": "DA-V2-S + EFT+B (ours)"}
DATASETS = ["kitti", "nyu", "diode-in", "diode-out"]
PRETTY_DS = {"kitti": "KITTI", "nyu": "NYUv2", "diode-in": "DIODE-in", "diode-out": "DIODE-out"}


def load(pattern="*__*.csv", folder=None):
    folder = Path(folder or RESULTS)
    files = [f for f in sorted(folder.glob(pattern)) if "__offc" not in f.name or "offc" in pattern]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    first = df[df.kind == "zoom"].groupby(["model", "dataset", "name", "cx", "cy"])["fx_gt"].transform("min")
    df.loc[df.kind == "zoom", "s_true"] = df.loc[df.kind == "zoom", "fx_gt"] / first
    return df


def slope_through_origin(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return float((x * y).sum() / (x * x).sum()) if len(x) >= 2 else np.nan


def per_image(df):
    """One row per (model, dataset, image, centre) with alpha, beta and a few errors."""
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


def boot_median(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summary(pi, by=("model", "dataset")):
    rows = []
    for key, g in pi.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        a, alo, ahi = boot_median(g.alpha)
        b, blo, bhi = boot_median(g.beta)
        rows.append(dict(zip(by, key), n=len(g), alpha=a, alpha_lo=alo, alpha_hi=ahi,
                         beta=b, beta_lo=blo, beta_hi=bhi, die2=np.nanmedian(g.die2),
                         absrel1=np.nanmean(g.absrel1), absrel2=np.nanmean(g.absrel2), absrel3=np.nanmean(g.absrel3),
                         base2=np.nanmean(g.base2), base3=np.nanmean(g.base3),
                         d1_1=np.nanmean(g.d1_1), d1_2=np.nanmean(g.d1_2), d1_3=np.nanmean(g.d1_3),
                         fx_err1=np.nanmedian(g.fx_err1)))
    return pd.DataFrame(rows)
