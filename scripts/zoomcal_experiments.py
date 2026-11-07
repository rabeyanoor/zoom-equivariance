"""ZoomCal 2.0 experiments E1, E2, E3, E5 and the selective-correction test, from the existing audit CSVs
(no new model runs). E4 needs depth maps and is in scripts/zoomcal_stress.py. Writes outputs/zoomcal/*.csv
and prints every number that docs/zoomcal2/RESULTS.md quotes.

    python scripts/zoomcal_experiments.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import zoomcal as zc  # noqa: E402

import analysis  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "zoomcal"
MAIN6 = ["dav2-s", "zoedepth", "depthpro", "moge2-s", "unidepth2-s", "metric3d-s"]


def full_table():
    """All images of the four test sets for the six main models (the __full runs)."""
    return zc.orbit_table(analysis.load("*__full.csv"))


def e1_shared(t, seed=0):
    """E1. (a) Inject a random shared log error into every view: label-free columns must not move at all.
    (b) The natural shared error of each model (log median pred/GT at 1x) against every label-free feature."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-np.log(2), np.log(2), len(t))
    u = zc.inject_shared(t, c)
    diag = ["alpha", "fit_rms", "loo_max", "beta", "alpha_K"]
    max_change = max(float(np.nanmax(np.abs(u[k] - t[k]))) if t[k].notna().any() else 0.0 for k in diag)
    rows = []
    for (m, d), g in t.groupby(["model", "dataset"]):
        for f in zc.LABEL_FREE:
            ok = g[f].notna() & g.scale1.notna()
            if ok.sum() > 30:
                rows.append(dict(model=m, dataset=d, feature=f, n=int(ok.sum()),
                                 rho=spearmanr(g.loc[ok, f], g.loc[ok, "scale1"].abs()).statistic))
    r = pd.DataFrame(rows)
    return max_change, float(np.nanmax(np.abs(u.scale1 - t.scale1 - c))), r


def e2_view(t, deltas=(0.05, 0.1, 0.2, 0.4), seed=0):
    """E2. Inject delta into one random zoomed view per image (half of the images, the other half clean).
    Detect (i) which images were corrupted, by loo_max (AUC), and (ii) which view, by argmax |loo| (accuracy)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (m, d), g in t.groupby(["model", "dataset"]):
        g = g.dropna(subset=[f"h{s}" for s in zc.ZOOMS]).reset_index(drop=True)
        if len(g) < 40:
            continue
        for delta in deltas:
            views = rng.choice(zc.ZOOMS, len(g))
            sign = rng.choice([-1, 1], len(g))
            hit = rng.random(len(g)) < 0.5
            u = g.copy()
            for s in zc.ZOOMS:
                u[f"h{s}"] = u[f"h{s}"] + np.where(hit & (views == s), sign * delta, 0.0)
            u = zc.add_diagnosis(u.drop(columns=["alpha", "fit_rms", "beta", "alpha_K", "loo_max", "drop_gain"]
                                        + [f"{p}{s}" for p in ("loo", "drop") for s in zc.ZOOMS]))
            loo = np.abs(u[[f"loo{s}" for s in zc.ZOOMS]].to_numpy())
            drop = u[[f"drop{s}" for s in zc.ZOOMS]].to_numpy()
            which = np.array(zc.ZOOMS)[np.argmax(loo, 1)]
            which_d = np.array(zc.ZOOMS)[np.argmin(drop, 1)]
            rows.append(dict(model=m, dataset=d, delta=delta, n=len(g),
                             auc=roc_auc_score(hit, u.loo_max), auc_drop=roc_auc_score(hit, u.drop_gain),
                             view_acc=float(np.mean(which[hit] == views[hit])),
                             view_acc_drop=float(np.mean(which_d[hit] == views[hit])),
                             alpha_shift=float(np.median(np.abs(u.alpha[hit] - g.alpha[hit])))))
    return pd.DataFrame(rows)


def e3_stale(seed=0):
    """E3. Camera mismatch, real runs: Metric3D v2-S given the true zoomed K' and given the stale K, same 800
    images. (a) Detect the stale run from the orbit (alpha), cross-fitted threshold. (b) Selective correction:
    multiply by s (exponent 1, exact for a stale K) only where detected; AbsRel at 3x on the mixture."""
    df = analysis.load("metric3d-s*__*.csv")
    t = zc.orbit_table(df)
    t = t[t.model.isin(["metric3d-s", "metric3d-s-staleK"])].dropna(subset=["alpha", "absrel3", "fix3_1.0"])
    stale = (t.model == "metric3d-s-staleK").to_numpy()
    auc = roc_auc_score(stale, t.alpha)
    rng = np.random.default_rng(seed)
    names = t.name.unique()
    half = set(rng.permutation(names)[: len(names) // 2])
    flag = np.zeros(len(t), bool)
    for fit in (half, set(names) - half):
        cal, ev = t.name.isin(fit).to_numpy(), ~t.name.isin(fit).to_numpy()
        grid = np.linspace(0, 1.5, 151)
        acc = [np.mean((t.alpha.to_numpy()[cal] > th) == stale[cal]) for th in grid]
        flag[ev] = t.alpha.to_numpy()[ev] > grid[int(np.argmax(acc))]
    t = t.assign(e=1.0)
    out = dict(n=len(t), auc=auc, acc=float(np.mean(flag == stale)),
               never=float(t.absrel3.mean()), always=zc.selective_outcome(t, np.ones(len(t), bool), e_col="e"),
               selective=zc.selective_outcome(t, flag, e_col="e"), by_true_label=zc.selective_outcome(t, stale, e_col="e"),
               oracle=float(np.minimum(t.absrel3, t["fix3_1.0"]).mean()),
               stale_alpha=float(t.alpha[stale].median()), true_alpha=float(t.alpha[~stale].median()))
    return out


def e3_focal_shared(t):
    """E3, shared focal error for a model whose metres are its focal times a camera-free output (Depth Pro):
    replacing f_hat by k f_hat in every view is the shared error of E1 (exact algebra), so the orbit cannot see it.
    Measured: how much of Depth Pro's unzoomed scale error is its focal error, log(f_hat/f_gt) vs scale1."""
    df = analysis.load("depthpro__*__full.csv")
    z = df[(df.kind == "zoom") & np.isclose(df.zoom, 1.0) & np.isclose(df.cx, 0.5)]
    fe = np.log(z.fx / z.fx_gt)
    rho = spearmanr(fe, z.log_ratio_gt).statistic
    resid = z.log_ratio_gt - fe   # scale error with the true focal: what the camera-free output contributes
    return dict(n=len(z), rho=rho, mad_scale=float(np.median(np.abs(z.log_ratio_gt))),
                mad_after_true_f=float(np.median(np.abs(resid))))


def _aurc_ratio(score, risk, rng, n_rand=200):
    """Area under the risk-coverage curve when keeping the lowest-score images first, over the random order."""
    order = np.argsort(score)
    def aurc(o):
        r = np.asarray(risk)[o]
        return float(np.mean(np.cumsum(r) / np.arange(1, len(r) + 1)))
    rand = np.mean([aurc(rng.permutation(len(risk))) for _ in range(n_rand)])
    return aurc(order) / rand


def e5_cross_model(t, seed=0):
    """E5. A label-free reliability score learned on some models, tested on an unseen one (leave one model out).
    Risk = the error a 3x zoom adds, AbsRel(3x) - AbsRel(unzoomed, same pixels). Baselines: |DIE_2| = |h2| (the
    paper's training-free score), |alpha|. Learned: ridge on the label-free features, trained on the other five
    models (all datasets). Metric: AURC over random (1 = no better than random), per held-out model and dataset."""
    rng = np.random.default_rng(seed)
    feats = [f for f in zc.LABEL_FREE if f in t]
    t = t.dropna(subset=feats + ["absrel3", "base3"]).copy()
    t["risk"] = t.absrel3 - t.base3
    rows = []
    for held in MAIN6:
        tr, te = t[t.model != held], t[t.model == held]
        if te.empty:
            continue
        x_tr = np.column_stack([np.abs(tr[f]) for f in feats])
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_tr, tr.risk.rank(pct=True))
        for d, g in te.groupby("dataset"):
            x = np.column_stack([np.abs(g[f]) for f in feats])
            rows.append(dict(model=held, dataset=d, n=len(g),
                             die2=_aurc_ratio(np.abs(g["h2.0"]).to_numpy(), g.risk.to_numpy(), rng),
                             abs_alpha=_aurc_ratio(np.abs(g.alpha).to_numpy(), g.risk.to_numpy(), rng),
                             learned=_aurc_ratio(reg.predict(x), g.risk.to_numpy(), rng)))
    return pd.DataFrame(rows)


def selective_zoomfix(t, seed=0):
    """Selective correction with a known zoom factor (SPEC 5). ZoomFix multiplies the 3x depth by 3**e with e the
    cross-fitted label-free alpha (grid 0.25..1). Policies, mean AbsRel at 3x per model and dataset:
    never, always (= ZoomFix), gate (logistic regression on the label-free features, trained leave-one-model-out
    on whether the correction lowered AbsRel), oracle (apply only where it helps; needs labels, a bound)."""
    feats = [f for f in zc.LABEL_FREE if f in t]
    t = t.dropna(subset=feats + ["absrel3", "fix3_1.0"]).copy()
    t["a_cf"] = zc.crossfit_alpha(t, seed)
    t["e"] = zc.zoomfix_exponent(t.a_cf)
    t["fixed3"] = [zc.fixed_absrel(t, i, e) for i, e in zip(t.index, t.e)]
    t["helps"] = t.fixed3 < t.absrel3
    rows = []
    for held in MAIN6:
        tr, te = t[t.model != held], t[t.model == held]
        if te.empty:
            continue
        x = lambda g: np.column_stack([np.abs(g[f]) for f in feats] + [g.e])  # noqa: E731
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x(tr), tr.helps)
        for d, g in te.groupby("dataset"):
            gate = clf.predict_proba(x(g))[:, 1] > 0.5
            rows.append(dict(model=held, dataset=d, n=len(g), alpha_cf=float(g.a_cf.median()),
                             never=float(g.absrel3.mean()), always=float(g.fixed3.mean()),
                             gate=float(np.where(gate, g.fixed3, g.absrel3).mean()),
                             oracle=float(np.minimum(g.fixed3, g.absrel3).mean()),
                             gate_rate=float(gate.mean()), helps_rate=float(g.helps.mean())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    t = full_table()
    t = t[t.model.isin(MAIN6)]
    t.to_csv(OUT / "orbit_full.csv", index=False)
    print(f"images: {t.groupby('model').size().to_dict()}")

    mc, sc, r1 = e1_shared(t)
    r1.to_csv(OUT / "e1_shared.csv", index=False)
    print(f"\nE1 max change of any label-free diagnosis under shared injection: {mc:.3g}; scale moved by c: {sc:.3g}")
    print("E1 |Spearman| between natural |scale error at 1x| and label-free features (per model/dataset):")
    print(r1.assign(abs_rho=r1.rho.abs()).groupby("feature").abs_rho.agg(["median", "max"]).round(3))

    r2 = e2_view(t)
    r2.to_csv(OUT / "e2_view.csv", index=False)
    print("\nE2 detection of a single corrupted view (median over model/dataset):")
    print(r2.groupby("delta")[["auc", "auc_drop", "view_acc", "view_acc_drop", "alpha_shift"]].median().round(3))
    print(r2.groupby(["model", "delta"]).auc.median().unstack().round(3))

    r3 = e3_stale()
    pd.DataFrame([r3]).to_csv(OUT / "e3_stale.csv", index=False)
    print(f"\nE3 stale-K detection and selective correction (Metric3D v2-S): {r3}")
    r3b = e3_focal_shared(t)
    print(f"E3 Depth Pro unzoomed: {r3b}")

    r5 = e5_cross_model(t)
    r5.to_csv(OUT / "e5_cross_model.csv", index=False)
    print("\nE5 AURC / random (lower is better), held-out model:")
    print(r5.groupby("model")[["die2", "abs_alpha", "learned"]].median().round(3))
    print(r5[["die2", "abs_alpha", "learned"]].median().round(3).to_dict(),
          "learned better than die2 in", int((r5.learned < r5.die2).sum()), "of", len(r5))

    rs = selective_zoomfix(t)
    rs.to_csv(OUT / "selective_zoomfix.csv", index=False)
    print("\nSelective ZoomFix, mean AbsRel at 3x:")
    print(rs.round(3).to_string(index=False))
    w = rs.n / rs.n.sum()
    print("pooled (image-weighted):", {k: round(float((rs[k] * w).sum()), 4) for k in ("never", "always", "gate", "oracle")})
    print("gate <= min(never, always) in", int((rs.gate <= np.minimum(rs.never, rs.always) + 1e-9).sum()), "of", len(rs))


if __name__ == "__main__":
    main()
