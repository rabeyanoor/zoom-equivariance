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


def test_merge_shards_joins_disjoint_shards_and_rejects_overlap(tmp_path):
    def shard(k, names):
        pd.DataFrame({"name": names, "kind": "zoom", "zoom": 1.0}).to_csv(tmp_path / f"m-s__kitti__full-s{k}.csv",
                                                                          index=False)

    shard(0, ["a", "c"])
    shard(1, ["b"])
    pd.DataFrame({"name": ["x"]}).to_csv(tmp_path / "dav2-s__kitti.csv", index=False)  # not a shard: left alone
    assert analysis.merge_shards(tmp_path, remove=True) == {"m-s__kitti__full.csv": 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dav2-s__kitti.csv", "m-s__kitti__full.csv"]
    shard(0, ["a"])
    shard(1, ["a"])
    try:
        analysis.merge_shards(tmp_path)
    except ValueError as e:
        assert "more than one shard" in str(e)
    else:
        raise AssertionError("overlapping shards were merged")


class Shrinking:
    """A focal head that shrinks log focal towards a prior: log f_hat = mu + b (log f - mu); its metres are the
    focal times a camera-free output, like Depth Pro, so its depth has alpha = 1 - b."""

    def __init__(self, b, mu=np.log(150.0)):
        self.b, self.mu = b, mu

    def predict(self, img, K=None):
        f = self.mu + self.b * (np.log(K[0, 0]) - self.mu)
        return {"depth": np.full(img.shape[:2], 10.0 * np.exp(f) / K[0, 0]), "fx": float(np.exp(f)), "cx": 99.5}


def shrink_frame(b, n=8):
    rows = []
    for i in range(n):
        for ds in ("a", "b"):
            sample = dict(SAMPLE, name=f"{ds}{i}")
            for r in audit.zoom_rows(Shrinking(b), sample, 80):
                r.update(model="shrink", dataset=ds)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["s_true"] = df.fx_gt / df.groupby("name").fx_gt.transform("min")
    return df


def test_focal_beta_is_label_free_and_recovers_the_shrinkage():
    assert abs(analysis.focal_beta(shrink_frame(0.6)) - 0.6) < 1e-3


def test_focal_deshrink_makes_a_shrinking_focal_head_equivariant():
    df = shrink_frame(0.6)
    s = analysis.deshrink_summary(analysis.focal_deshrink(df)).iloc[0]
    assert abs(s.beta - 0.6) < 1e-3
    assert abs(s.alpha - 0.4) < 0.02 and abs(s.alpha_c) < 0.02
    # without labels the 1x bias stays (the centre is the model's own unzoomed estimate), but the zoom no longer
    # adds error: after the correction the focal error at 3x equals the model's own 1x error
    assert abs(s.focal1_c - s.focal1) < 1e-6
    assert abs(s.focal3_c - s.focal1) < 1e-3 and s.focal3 > s.focal1 + 0.1


def test_focal_deshrink_leaves_an_equivariant_focal_head_unchanged():
    r = analysis.focal_deshrink(shrink_frame(1.0))
    assert np.allclose(r.log_corr, 0, atol=1e-6)
