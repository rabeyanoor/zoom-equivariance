"""The audit on toy models whose behaviour under zoom is known in closed form."""
import numpy as np
import pandas as pd

import analysis
import audit

K = np.array([[100, 0, 99.5], [0, 100, 29.5], [0, 0, 1.0]])
SAMPLE = {"img": np.zeros((60, 200, 3), np.uint8), "depth": np.full((60, 200), 10.0, np.float32),
          "K": K, "name": "toy"}


class Equivariant:
    """Knows the camera: depth fixed, focal and principal point follow the law."""

    def predict(self, img, K=None):
        return {"depth": np.full(img.shape[:2], 10.0), "fx": K[0, 0], "cx": K[0, 2]}


class ApparentSize:
    """Reads apparent size with a fixed focal: depth ~ 1/s, focal ignores the zoom."""

    def predict(self, img, K=None):
        return {"depth": np.full(img.shape[:2], 10.0 * 100 / K[0, 0]), "fx": 100.0, "cx": 99.5}


def exponents(model):
    rows = audit.zoom_rows(model, SAMPLE, max_depth=80)
    for r in rows:
        r.update(model="m", dataset="d")
    df = pd.DataFrame(rows)
    first = df[df.kind == "zoom"].groupby("name").fx_gt.transform("min")
    df.loc[df.kind == "zoom", "s_true"] = df.loc[df.kind == "zoom", "fx_gt"] / first
    pi = analysis.per_image(df)
    return float(pi.alpha.iloc[0]), float(pi.beta.iloc[0]), rows


def test_equivariant_model_scores_alpha0_beta1():
    a, b, rows = exponents(Equivariant())
    assert abs(a) < 1e-6 and abs(b - 1) < 1e-6
    assert all(r["abs_rel"] < 1e-6 for r in rows)


def test_apparent_size_model_scores_alpha1_beta0_and_is_fixed_by_e1():
    a, b, rows = exponents(ApparentSize())
    assert abs(a - 1) < 0.02 and abs(b) < 1e-6
    zoomed = [r for r in rows if r["kind"] == "zoom" and r["zoom"] > 1]
    assert all(r["abs_rel_fix1.0"] < 0.01 for r in zoomed)
    assert all(r["abs_rel"] > 0.15 for r in zoomed)


def test_blur_control_row_is_present():
    rows = audit.zoom_rows(Equivariant(), SAMPLE, max_depth=80)
    blur = [r for r in rows if r["kind"] == "blur"]
    assert len(blur) == 1 and abs(blur[0]["log_ratio_self"]) < 1e-9


def test_controls_equivariant_model():
    """A model that knows the camera: the optical and digital zooms keep depth, JPEG changes nothing."""
    rows = {r["kind"]: r for r in audit.control_rows(Equivariant(), SAMPLE, 80)}
    assert set(rows) == {"optical", "digital_lr", "bilinear", "jpeg"}
    for r in rows.values():
        assert abs(r["log_ratio_self"]) < 1e-6 and r["shape_err"] < 1e-6


def test_controls_apparent_size_model():
    rows = {r["kind"]: r for r in audit.control_rows(ApparentSize(), SAMPLE, 80)}
    # both 2x zooms (optical and digital) halve the depth of a size-reading model
    for k in ("optical", "digital_lr", "bilinear"):
        assert abs(rows[k]["log_ratio_self"] - np.log(0.5)) < 0.02
    assert abs(rows["jpeg"]["log_ratio_self"]) < 1e-6


def test_shrink_K_undoes_zoom_K():
    import zoom

    small, box = audit.shrink_image(SAMPLE["img"], 2.0)
    Ks = audit.shrink_K(K, box, (60, 200))
    np.testing.assert_allclose(zoom.zoom_K(Ks, box, (60, 200)), K, atol=1e-9)


def test_cues_equivariant_model_keeps_depth():
    rows = {r["kind"]: r for r in audit.cue_rows(Equivariant(), SAMPLE, 80)}
    assert set(rows) == {"context", "magnify", "zoom", "shrink"}
    for r in rows.values():
        assert abs(r["log_ratio_self"]) < 1e-6


def test_cues_apparent_size_model_reacts_to_magnification_only():
    rows = {r["kind"]: r for r in audit.cue_rows(ApparentSize(), SAMPLE, 80)}
    assert abs(rows["context"]["log_ratio_self"]) < 1e-6
    assert abs(rows["magnify"]["log_ratio_self"] - np.log(0.5)) < 0.02
    assert abs(rows["zoom"]["log_ratio_self"] - np.log(0.5)) < 0.02
    assert abs(rows["shrink"]["log_ratio_self"] - np.log(2.0)) < 0.02


def test_given_focal_makes_a_focal_head_model_equivariant():
    # ApparentSize is a fixed focal times a correct camera-free depth (depth = f_hat * 10 / f_true),
    # the Depth Pro / DA3-Nested structure: with the true focal its whole zoom error goes away
    _, _, rows = exponents(ApparentSize())
    for r in rows:
        r.update(model="m", dataset="d")
    df = pd.DataFrame(rows)
    z = df[df.kind == "zoom"].copy()
    z["s_true"] = z.fx_gt / z.fx_gt.min()
    g = analysis.given_focal(z)
    pi = analysis.per_image(g)
    assert abs(pi.alpha.iloc[0]) < 1e-6 and abs(pi.beta.iloc[0] - 1) < 1e-6
    assert np.allclose(g.log_ratio_gt, 0, atol=1e-6)


def test_region_rows_split_the_zoom_response_by_group():
    import regions

    seg = np.full(SAMPLE["img"].shape[:2], 6)  # road (surface)
    seg[:, :100] = 20  # left half: car (object)
    rows = pd.DataFrame(regions.region_rows(ApparentSize(), SAMPLE, seg))
    for _, r in rows.iterrows():  # apparent size: depth ~ 1/s on every region
        assert abs(r.log_ratio_self + np.log(r.s_true)) < 1e-6
    assert set(rows.group) == {"object", "surface"}
    eq = pd.DataFrame(regions.region_rows(Equivariant(), SAMPLE, seg))
    assert np.allclose(eq.log_ratio_self, 0, atol=1e-6)
