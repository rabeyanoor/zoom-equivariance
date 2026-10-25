"""SR-RAW optical-zoom scoring: focal from EXIF, matched centre boxes, alpha and beta of a known model."""
import numpy as np
import pandas as pd
from PIL import Image

import analysis
import srraw


class CameraPower:
    """Depth and focal follow the given K with known exponents: depth ~ s^-a, fx ~ s^b."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def predict(self, img, K=None):
        s = K[0, 0] / 100.0
        return {"depth": np.full(img.shape[:2], 5.0 * s ** -self.a), "fx": 100.0 * s ** self.b}


def write_shot(path, f_mm, f35, size=(120, 80)):
    img = Image.fromarray(np.random.default_rng(int(f_mm)).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    exif = Image.Exif()
    ifd = exif.get_ifd(srraw.EXIF_IFD)
    ifd[srraw.FOCAL_TAG] = f_mm
    ifd[srraw.F35_TAG] = f35
    img.save(path, exif=exif)


def test_focal_and_k_from_exif(tmp_path):
    write_shot(tmp_path / "00001.JPG", 24.0, 24)
    img, f, f35 = srraw.read_shot(tmp_path / "00001.JPG", long_side=60)
    assert img.shape == (40, 60, 3) and f == 24.0 and f35 == 24.0
    K = srraw.K_from_f35(36.0, 40, 60)  # 36 mm equivalent: focal = the long side
    assert K[0, 0] == 60 and K[0, 2] == 30 and K[1, 2] == 20


def test_centre_box_and_median():
    d = np.ones((100, 200))
    d[45:55, 90:110] = np.e
    assert srraw.centre_box(100, 200, 0.1) == (45, 55, 90, 110)
    assert srraw.median_log(d, 0.1) == (1.0, 200)


def test_alpha_beta_of_known_model(tmp_path):
    seq = tmp_path / "seq1"
    seq.mkdir()
    for i, f in enumerate((24.0, 35.0, 50.0, 70.0, 100.0)):  # 100 mm is beyond S_MAX and is skipped
        write_shot(seq / f"{i + 1:05d}.JPG", f, int(f))
    assert srraw.sequences(tmp_path) == [seq]
    rows = pd.DataFrame(srraw.sequence_rows(CameraPower(0.4, 0.6), seq, needs_K=True)).assign(model="m")
    assert set(rows.view) == {"base", "optical", "digital"} and rows.s.max() <= srraw.S_MAX
    assert np.allclose(sorted(rows[rows.view == "optical"].s), [35 / 24, 50 / 24, 70 / 24])
    out = analysis.srraw_alpha(rows)
    assert np.allclose(out.alpha, 0.4) and np.allclose(out.beta, 0.6)
