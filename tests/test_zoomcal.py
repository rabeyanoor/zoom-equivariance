import numpy as np
import pandas as pd

import zoomcal as zc


def _table(alpha, n=50, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    rows = {"model": "m", "dataset": "d", "name": [f"i{k}" for k in range(n)]}
    t = pd.DataFrame(rows)
    for s in zc.ZOOMS:
        t[f"logs{s}"] = np.log(s)
        t[f"h{s}"] = -alpha * np.log(s) + rng.normal(0, noise, n)
        t[f"fgap{s}"] = np.nan
    t["scale1"] = rng.normal(0, 0.2, n)
    t["scale3"] = t.scale1
    return zc.add_diagnosis(t)


def test_alpha_recovered():
    t = _table(0.7, noise=0.0)
    assert np.allclose(t.alpha, 0.7) and np.allclose(t.fit_rms, 0, atol=1e-12)


def test_shared_error_is_invisible():
    """Prop. 1: a log error shared by every view leaves every label-free quantity unchanged."""
    t = _table(0.4)
    u = zc.inject_shared(t, np.linspace(-1, 1, len(t)))
    for k in ("alpha", "fit_rms", "loo_max"):
        assert np.array_equal(u[k].to_numpy(), t[k].to_numpy())
    assert np.allclose(u.scale1 - t.scale1, np.linspace(-1, 1, len(t)))


def test_view_error_is_localised():
    t = _table(0.4, noise=0.0)
    u = zc.inject_view(t, 2.0, 0.3)
    drop = u[[f"drop{s}" for s in zc.ZOOMS]].to_numpy()
    assert (np.array(zc.ZOOMS)[drop.argmin(1)] == 2.0).all()
    # the plain leave-one-out residual blames s = 3 here (leverage of the largest zoom)
    loo = np.abs(u[[f"loo{s}" for s in zc.ZOOMS]].to_numpy())
    assert (np.array(zc.ZOOMS)[loo.argmax(1)] == 3.0).all()


def test_zoomfix_exponent_grid():
    assert list(zc.zoomfix_exponent(np.array([-0.2, 0.1, 0.3, 0.9, 1.4]))) == [0.0, 0.0, 0.25, 1.0, 1.0]
