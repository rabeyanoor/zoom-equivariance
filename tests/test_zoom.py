"""The zoom law and our implementation of it."""
import numpy as np
import pytest

import zoom

H, W = 352, 1216
K = np.array([[721.5, 0, 596.5], [0, 721.5, 149.9], [0, 0, 1]])


def project(K, P):
    p = (K @ P.T).T
    return p[:, :2] / p[:, 2:]


@pytest.mark.parametrize("s", [1.25, 2.0, 3.0])
@pytest.mark.parametrize("centre", [(0.5, 0.5), (0.25, 0.75), (0.9, 0.1)])
def test_zoom_K_matches_crop_and_resize(s, centre):
    rng = np.random.default_rng(0)
    P = np.stack([rng.uniform(-10, 10, 2000), rng.uniform(-2, 3, 2000), rng.uniform(3, 80, 2000)], 1)
    box = zoom.zoom_box(H, W, s, *centre)
    x0, y0, cw, ch = box
    uv = project(K, P)
    # pixel u in the crop goes to (u - x0 + 1/2) * W / cw - 1/2 after an align_corners=False resize
    expected = np.stack([(uv[:, 0] - x0 + 0.5) * W / cw - 0.5, (uv[:, 1] - y0 + 0.5) * H / ch - 0.5], 1)
    np.testing.assert_allclose(project(zoom.zoom_K(K, box, (H, W)), P), expected, atol=1e-6)


@pytest.mark.parametrize("s", [1.0, 1.5, 3.0])
def test_box_stays_inside_and_has_the_right_size(s):
    for c in [(0, 0), (1, 1), (0.5, 0.5)]:
        x0, y0, cw, ch = zoom.zoom_box(H, W, s, *c)
        assert 0 <= x0 and x0 + cw <= W and 0 <= y0 and y0 + ch <= H
        assert abs(cw - W / s) <= 1 and abs(ch - H / s) <= 1


def test_constant_depth_survives_unzoom():
    box = zoom.zoom_box(H, W, 2.0)
    back = zoom.unzoom_depth(np.full((H, W), 7.0, np.float32), box)
    assert back.shape == (box[3], box[2])
    np.testing.assert_allclose(back, 7.0)


def test_zoom_and_blur_keep_size_and_dtype():
    img = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    for out in (zoom.zoom_image(img, zoom.zoom_box(H, W, 2.0)), zoom.blur_control(img, 2.0)):
        assert out.shape == img.shape and out.dtype == np.uint8


def test_zoom_1x_is_identity_in_K():
    np.testing.assert_allclose(zoom.zoom_K(K, (0, 0, W, H), (H, W)), K)
