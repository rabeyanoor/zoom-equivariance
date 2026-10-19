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
