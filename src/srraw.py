"""Real optical zoom: the zoom audit on SR-RAW (Zhang et al., "Zoom to Learn, Learn to Zoom", CVPR 2019).

Each SR-RAW sequence is one static scene shot with a Sony FE 24-240 mm zoom lens at 7 focal lengths
(00001.JPG ... 00007.JPG, focal length in EXIF). The camera does not move, so the true depth of the scene
is the same in every shot: a model with the right geometry keeps its depth (alpha = 0) and scales its focal
with the lens (beta = 1). No depth labels are needed.

For a shot with focal f_i against the widest shot f_0 (s = f_i / f_0), the zoomed frame shows the centred
1/s box of the wide frame (the alignment the SR-RAW code assumes too). We compare the median log depth over
the inner Q of the zoomed frame with the median over the matching centred Q/s box of the wide frame's
prediction; the inner Q leaves a margin for the small shift of the lens centre while zooming. The same
crop of the wide image, done digitally, is scored alongside, so optical and digital zoom are compared on
the same scenes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

FOCAL_TAG, F35_TAG, EXIF_IFD = 37386, 41989, 0x8769  # FocalLength, FocalLengthIn35mmFilm, Exif sub-IFD
LONG_SIDE = 1024  # working resolution of every shot (the full-resolution JPGs are downsampled with LANCZOS)
Q = 0.8           # inner fraction of the zoomed frame that is compared
S_MAX = 3.2       # the zoom range of the main audit (s <= 3)


def read_shot(path, long_side=LONG_SIDE):
    """RGB uint8 at the working size (EXIF orientation applied), focal length in mm, 35 mm-equivalent focal."""
    img = Image.open(path)
    ifd = img.getexif().get_ifd(EXIF_IFD)
    f = float(ifd[FOCAL_TAG]) if FOCAL_TAG in ifd else np.nan
    f35 = float(ifd[F35_TAG]) if ifd.get(F35_TAG) else np.nan
    img = ImageOps.exif_transpose(img).convert("RGB")
    k = long_side / max(img.size)
    img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return np.asarray(img), f, f35


def sequences(root):
    """Folders that hold an SR-RAW sequence (00001.JPG and at least one more shot). Each test sequence also has an
    aligned/ subfolder (the authors' pre-aligned crops, without EXIF); it is not a sequence of its own."""
    return sorted(p.parent for p in Path(root).rglob("00001.JPG")
                  if p.parent.name != "aligned" and len(list(p.parent.glob("0000[2-7].JPG"))))


def centre_box(h, w, frac):
    ch, cw = max(1, round(h * frac)), max(1, round(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return y0, y0 + ch, x0, x0 + cw


def median_log(depth, frac):
    y0, y1, x0, x1 = centre_box(*depth.shape, frac)
    d = depth[y0:y1, x0:x1]
    d = d[np.isfinite(d) & (d > 0)]
    return (float(np.median(np.log(d))), int(d.size)) if d.size else (np.nan, 0)


def digital_zoom(img, s):
    """Centred crop of 1/s, resampled back to the full working size (bilinear, as in the main audit)."""
    h, w = img.shape[:2]
    y0, y1, x0, x1 = centre_box(h, w, 1 / s)
    return np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((w, h), Image.BILINEAR))


def K_from_f35(f35, h, w):
    """Pinhole K from the 35 mm-equivalent focal: the 36 mm film width maps to the long image side."""
    if not np.isfinite(f35):
        return None
    fx = f35 / 36.0 * max(h, w)
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


def sequence_rows(model, seq, needs_K=False):
    shots = [(p, *read_shot(p)) for p in sorted(Path(seq).glob("0000[1-7].JPG"))]
    shots = [s for s in shots if np.isfinite(s[2])]
    if len(shots) < 2:
        return []
    shots.sort(key=lambda s: s[2])
    _, base, f0, f35_0 = shots[0]
    h, w = base.shape[:2]

    def run(img, f35):
        K = K_from_f35(f35, *img.shape[:2])
        if needs_K and K is None:
            return None
        return model.predict(img, K=K)

    out0 = run(base, f35_0)
    if out0 is None:
        return []
    rows = [dict(seq=Path(seq).name, view="base", f_mm=f0, s=1.0, log_ratio=0.0, n_px=int(base.size // 3),
                 fx=out0.get("fx"), fx_base=out0.get("fx"))]
    for _, img, f, f35 in shots[1:]:
        s = f / f0
        if s > S_MAX or img.shape[:2] != (h, w):
            continue
        m0, _ = median_log(out0["depth"], Q / s)
        # optical: the real zoomed shot; digital: the wide shot cropped by the same factor (K scales with s)
        f35_dig = f35_0 * s if np.isfinite(f35_0) else np.nan
        for view, im, f35_v in (("optical", img, f35), ("digital", digital_zoom(base, s), f35_dig)):
            out = run(im, f35_v)
            if out is None:
                continue
            m, n = median_log(out["depth"], Q)
            rows.append(dict(seq=Path(seq).name, view=view, f_mm=f, s=s, log_ratio=m - m0, n_px=n,
                             fx=out.get("fx"), fx_base=out0.get("fx")))
    return rows


def run(model_name, root, out_dir, every=1):
    import models

    model = models.load(model_name, "Outdoor")
    needs_K = models.REGISTRY[model_name][0].startswith("takes camera")
    rows = []
    seqs = sequences(root)[::every]
    print(f"{model_name}: {len(seqs)} SR-RAW sequences", flush=True)
    for i, seq in enumerate(seqs):
        rows += sequence_rows(model, seq, needs_K)
        if (i + 1) % 10 == 0:
            print(f"{model_name}: {i + 1}/{len(seqs)}", flush=True)
    out = Path(out_dir) / f"{model_name}__srraw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(model=model_name).to_csv(out, index=False)
    return out
