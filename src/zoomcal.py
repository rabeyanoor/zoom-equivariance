"""ZoomCal 2.0 proof of concept: diagnose a depth model's error from an orbit of exact zooms, and correct only
what the orbit can identify. The maths is in docs/zoomcal2/SPEC.md; the section numbers below refer to it.

Error model in log depth (SPEC 1). View v of image I is the zoom Z_v with exact factor s_v (v = 0: s_0 = 1).
On a pixel p of the original image that survives every crop,

    l_v(p) = d(p) + g + h_v + e_v(p),    h_0 = 0,

d the true log depth, g a shared error (same in every view), h_v a view-dependent global error and e_v a
residual that varies over pixels. The orbit only shows differences to the unzoomed view,

    r_v(p) = l_v(p) - l_0(p) = h_v + e_v(p) - e_0(p),

so g cancels from every label-free quantity (SPEC Prop. 1), while h_v is the median of r_v up to the residual.
"""
import numpy as np
import pandas as pd

ZOOMS = (1.25, 1.5, 2.0, 3.0)
EPS = 1e-6


# ------------------------------------------------------------------ map level (SPEC 2)
def view_residual(logd_view_on_crop, logd_base_on_crop):
    """r_v on the pixels both maps cover: (h_v estimate = median, shape = median |r - median|, n pixels)."""
    a, b = np.asarray(logd_view_on_crop, float), np.asarray(logd_base_on_crop, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan, np.nan, int(ok.sum())
    r = a[ok] - b[ok]
    h = float(np.median(r))
    return h, float(np.median(np.abs(r - h))), int(ok.sum())


def fit_alpha(log_s, h):
    """Least-squares slope through the origin of -h on log s (the paper's alpha) and the RMS residual."""
    log_s, h = np.asarray(log_s, float), np.asarray(h, float)
    ok = np.isfinite(log_s) & np.isfinite(h)
    if ok.sum() < 2:
        return np.nan, np.nan
    x, y = log_s[ok], h[ok]
    a = -float((x * y).sum() / (x * x).sum())
    return a, float(np.sqrt(np.mean((y + a * x) ** 2)))


def leave_one_out_residuals(log_s, h):
    """For each view, h minus the power-law prediction fitted on the other views (SPEC 3, E2)."""
    log_s, h = np.asarray(log_s, float), np.asarray(h, float)
    out = np.full(len(h), np.nan)
    for i in range(len(h)):
        m = np.arange(len(h)) != i
        a, _ = fit_alpha(log_s[m], h[m])
        out[i] = h[i] + a * log_s[i]
    return out


def drop_one_rms(log_s, h):
    """For each view, the RMS of the power-law fit on the *other* views. The view whose removal leaves the best fit
    is the one to blame. A large leave-one-out residual alone can point at the wrong view: in a fit through the origin
    the largest zoom has the most leverage, so an error at s = 2 also leaves a large residual at s = 3."""
    log_s, h = np.asarray(log_s, float), np.asarray(h, float)
    return np.array([fit_alpha(log_s[np.arange(len(h)) != i], h[np.arange(len(h)) != i])[1] for i in range(len(h))])


# ------------------------------------------------------------------ per image from audit rows (SPEC 3)
def orbit_table(df):
    """Audit rows (one file or many) -> one row per (model, dataset, image) with the label-free orbit
    h_s = log_ratio_self at every zoom, the focal gap log(f_s / f_1) - log s, shape error, the blur response,
    and the labelled columns used only for evaluation. Centred crops only."""
    z = df[np.isclose(df.cx, 0.5) & np.isclose(df.cy, 0.5)].copy()
    z["zoom"] = z.zoom.round(4)
    key = ["model", "dataset", "name"]
    zz = z[z.kind == "zoom"]
    allz = sorted(zz.zoom.unique())
    piv = lambda col: zz.pivot_table(index=key, columns="zoom", values=col, aggfunc="first",  # noqa: E731
                                     dropna=False).reindex(columns=allz)
    h, s_true = piv("log_ratio_self"), piv("s_true") if "s_true" in zz else None
    if s_true is None:
        fx_true = zz.assign(st=zz.fx_gt / zz.groupby(key).fx_gt.transform("min"))
        s_true = fx_true.pivot_table(index=key, columns="zoom", values="st", aggfunc="first")
    out = pd.DataFrame(index=h.index)
    for s in ZOOMS:
        out[f"h{s}"] = h[s]
        out[f"logs{s}"] = np.log(s_true[s])
    fx, fxb = piv("fx"), piv("fx_base")
    for s in ZOOMS:
        out[f"fgap{s}"] = np.log(fx[s] / fxb[s]) - out[f"logs{s}"] if fx[s].notna().any() else np.nan
    if "shape_err" in zz:
        sh = piv("shape_err")
        out["shape2"], out["shape3"] = sh[2.0], sh[3.0]
    b = z[z.kind == "blur"].set_index(key).log_ratio_self
    out["blur_h"] = b.groupby(level=[0, 1, 2]).first()
    for col, s, name in (("abs_rel", 1.0, "absrel1"), ("abs_rel", 3.0, "absrel3"), ("abs_rel", 2.0, "absrel2"),
                         ("abs_rel_base", 3.0, "base3"), ("abs_rel_base", 2.0, "base2"),
                         ("log_ratio_gt", 1.0, "scale1"), ("log_ratio_gt", 3.0, "scale3")):
        out[name] = piv(col)[s] if col in zz else np.nan
    for e in (0.25, 0.5, 0.75, 1.0):
        col = f"abs_rel_fix{e}"
        if col in zz:
            out[f"fix3_{e}"], out[f"fix2_{e}"] = piv(col)[3.0], piv(col)[2.0]
    return add_diagnosis(out.reset_index())


def add_diagnosis(t):
    """Label-free diagnosis per image (SPEC 3): alpha, fit RMS, leave-one-out residuals, beta, alpha_K."""
    H = t[[f"h{s}" for s in ZOOMS]].to_numpy(float)
    L = t[[f"logs{s}" for s in ZOOMS]].to_numpy(float)
    G = t[[f"fgap{s}" for s in ZOOMS]].to_numpy(float)
    num, den = np.nansum(L * H, 1), np.nansum(np.where(np.isfinite(H), L * L, 0), 1)
    t["alpha"] = -num / np.where(den > 0, den, np.nan)
    t["fit_rms"] = np.sqrt(np.nanmean((H + t.alpha.to_numpy()[:, None] * L) ** 2, 1))
    loo = np.array([leave_one_out_residuals(ls, h) for ls, h in zip(L, H)])
    for i, s in enumerate(ZOOMS):
        t[f"loo{s}"] = loo[:, i]
    t["loo_max"] = np.nanmax(np.abs(loo), 1)
    drop = np.array([drop_one_rms(ls, h) for ls, h in zip(L, H)])
    for i, s in enumerate(ZOOMS):
        t[f"drop{s}"] = drop[:, i]
    # how much the fit improves when the worst view is left out (0: no single view stands out)
    t["drop_gain"] = t.fit_rms - np.nanmin(drop, 1)
    # beta from the focal gap: log(f_s/f_1) = beta log s  <=>  gap = (beta - 1) log s
    gnum = np.nansum(L * G, 1)
    gden = np.nansum(np.where(np.isfinite(G), L * L, 0), 1)
    t["beta"] = np.where(np.isfinite(G).any(1), 1 + gnum / np.where(gden > 0, gden, np.nan), np.nan)
    # SPEC Prop. 3: for D = f * C(I) exactly alpha + beta - 1 = alpha_K, the camera-free part of the error
    t["alpha_K"] = t.alpha + t.beta - 1
    return t


# ------------------------------------------------------------------ injections (SPEC 4, E1-E3)
def _strip(u):
    """Drop the diagnosis columns so add_diagnosis can recompute them."""
    return u.drop(columns=[k for k in u if k.startswith(("loo", "drop"))] + ["alpha", "fit_rms", "beta", "alpha_K"])


def inject_shared(t, c):
    """E1: add the same log error c (scalar or per image) to every view. Label-free columns are differences of
    views and must not change (Prop. 1); the labelled scale error moves by c. Returns a new table."""
    u = t.copy()
    c = np.broadcast_to(np.asarray(c, float), (len(u),))
    for col in ("scale1", "scale3"):
        u[col] = u[col] + c
    return add_diagnosis(_strip(u))


def inject_view(t, view, delta):
    """E2: add delta to the log depth of one zoomed view only (an error that a zoom of that size alone has)."""
    u = t.copy()
    u[f"h{view}"] = u[f"h{view}"] + delta
    return add_diagnosis(_strip(u))


# ------------------------------------------------------------------ selective correction (SPEC 5)
def zoomfix_exponent(alpha_hat, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """The grid exponent closest to a measured alpha (the audit stored AbsRel of D * s**e on 0.25..1; e = 0 is no
    correction). A negative alpha maps to 0: the stored grid cannot shrink depth."""
    g = np.asarray(grid)
    return g[np.argmin(np.abs(g[None, :] - np.asarray(alpha_hat, float)[:, None]), 1)]


def crossfit_alpha(t, seed=0):
    """Label-free dataset-level alpha per (model, dataset), measured on one half of the images and assigned to
    the other half (then swapped), as for ZoomFix in the paper."""
    rng = np.random.default_rng(seed)
    a = pd.Series(np.nan, index=t.index)
    for _, g in t.groupby(["model", "dataset"]):
        idx = rng.permutation(g.index.to_numpy())
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        a[h2] = t.loc[h1, "alpha"].median()
        a[h1] = t.loc[h2, "alpha"].median()
    return a


def fixed_absrel(t, i, e, s=3):
    """AbsRel at zoom s of image row i after multiplying depth by s**e (e = 0: the uncorrected AbsRel)."""
    return t.at[i, f"absrel{s}"] if e == 0 else t.at[i, f"fix{s}_{e}"]


def selective_outcome(t, decide, s=3, e_col=None):
    """Mean AbsRel at zoom s when the correction is applied where decide is True and not elsewhere.
    e_col: column with the chosen grid exponent per image."""
    fixed = np.array([fixed_absrel(t, i, e, s) for i, e in zip(t.index, t[e_col])])
    return float(np.nanmean(np.where(decide, fixed, t[f"absrel{s}"])))


LABEL_FREE = ["alpha", "fit_rms", "loo_max", "h2.0", "h3.0", "blur_h", "shape2", "shape3"]
