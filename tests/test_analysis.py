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
