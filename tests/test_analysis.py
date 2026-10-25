"""The vectorised per-image fit gives exactly the loop's numbers; bootstrap behaves."""
import numpy as np
import pandas as pd

import analysis
import audit
from test_audit import SAMPLE, ApparentSize, Equivariant


class Noisy:
    """Partly camera-aware model with per-zoom noise, so slopes are not trivial."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, img, K=None):
        f = K[0, 0] if K is not None else 100.0
        s = f / 100.0
        return {"depth": np.full(img.shape[:2], 10.0 * s ** -0.4 * np.exp(self.rng.normal(0, 0.02))),
                "fx": 100.0 * s ** 0.6 * np.exp(self.rng.normal(0, 0.02)), "cx": 99.5}


def toy_frame():
    rows = []
    for k, (name, model) in enumerate([("eq", Equivariant()), ("size", ApparentSize()), ("noisy", Noisy(0))]):
        for i in range(3):
            sample = dict(SAMPLE, name=f"{name}{i}")
            for r in audit.zoom_rows(model, sample, 80, center=analysis.CENTRES[i % 2]):
                r.update(model=name, dataset="toy")
                rows.append(r)
    df = pd.DataFrame(rows)
    first = df[df.kind == "zoom"].groupby(["name", "cx", "cy"]).fx_gt.transform("min")
    df.loc[df.kind == "zoom", "s_true"] = df.loc[df.kind == "zoom", "fx_gt"] / first
    return df


def test_vectorised_equals_loop():
    df = toy_frame()
    new = analysis.per_image(df).set_index(analysis.KEY).sort_index()
    old = analysis.per_image_loop(df).set_index(analysis.KEY).sort_index()
    for col in ["alpha", "beta", "die2", "absrel1", "absrel3", "base3", "fx_err1"]:
        np.testing.assert_allclose(new[col].astype(float), old[col].astype(float), atol=1e-12, equal_nan=True)


def test_noisy_model_exponents_are_recovered():
    pi = analysis.per_image(toy_frame())
    noisy = pi[pi.model == "noisy"]
    assert np.allclose(noisy.alpha, 0.4, atol=0.08) and np.allclose(noisy.beta, 0.6, atol=0.08)


def test_bootstrap_interval_contains_median():
    v = np.random.default_rng(0).normal(1.0, 0.3, 400)
    med, lo, hi = analysis.boot_median(v)
    assert lo < med < hi and abs(med - 1.0) < 0.05


def test_zoomfix_picks_e1_for_apparent_size_model():
    a = toy_frame()
    b = a.copy()
    b["dataset"] = "toy2"
    pi = analysis.per_image(pd.concat([a, b], ignore_index=True))
    fix = analysis.zoomfix(pi[pi.model == "size"])
    assert (fix.exponent == 1.0).all()
    assert (fix.absrel_fixed < 0.01).all() and (fix.absrel_zoomed > 0.5).all()


def test_principal_point_error_is_zero_for_equivariant_model():
    pp = analysis.principal_point(toy_frame())
    eq = pp[pp.model == "eq"]
    assert np.allclose(eq.model_error, 0, atol=1e-9)


def test_zoomfix_crossfit_recovers_apparent_size_exponent():
    rows = []
    for i in range(6):
        sample = dict(SAMPLE, name=f"size{i}")
        for r in audit.zoom_rows(ApparentSize(), sample, 80, exponents=audit.FINE_EXPONENTS):
            r.update(model="size", dataset="toy")
            rows.append(r)
    df = pd.DataFrame(rows)
    first = df[df.kind == "zoom"].groupby("name").fx_gt.transform("min")
    df.loc[df.kind == "zoom", "s_true"] = df.loc[df.kind == "zoom", "fx_gt"] / first
    fix = analysis.zoomfix_crossfit(df)
    assert np.isclose(fix.exponent.iloc[0], 1.0) and fix.absrel_fixed.iloc[0] < 0.01


def test_fit_quality_on_an_exact_power_law():
    rows = []
    for r in audit.zoom_rows(ApparentSize(), dict(SAMPLE, name="size0"), 80):
        r.update(model="size", dataset="toy")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["s_true"] = df.fx_gt / df[df.kind == "zoom"].fx_gt.min()
    q = analysis.fit_quality(df)
    assert len(q) == 1 and abs(q.alpha[0] - 1) < 1e-6 and q.rms[0] < 1e-6
    assert abs(q.alpha_small[0] - q.alpha_large[0]) < 1e-6


def test_twoway_bootstrap_contains_the_estimate():
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame([dict(model=m, dataset=d, x=i + rng.normal(0, .5), y=i + rng.normal(0, .5))
                          for i, m in enumerate("abcdefgh") for d in "wxyz"])
    rho, lo, hi = analysis.twoway_boot_spearman(pairs, "x", "y", n=200)
    assert lo <= rho <= hi and rho > 0.8


def test_region_alpha_pairs_groups_per_image():
    import regions

    seg = np.full(SAMPLE["img"].shape[:2], 6)
    seg[:, :100] = 20
    rows = pd.DataFrame(regions.region_rows(ApparentSize(), SAMPLE, seg)).assign(model="m", dataset="d")
    per, wide = analysis.region_alpha(rows)
    assert np.allclose(per.alpha, 1) and len(wide) == 1 and abs(wide["diff"][0]) < 1e-6
