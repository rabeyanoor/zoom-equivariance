"""The zoom audit: run a model on an image and on digitally zoomed copies of it, and record
how depth and focal change. Results go to one CSV row per (image, zoom); the run can be
stopped and restarted, finished rows are skipped.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import models
import zoom

ZOOMS = (1.0, 1.25, 1.5, 2.0, 3.0)
# exponents for the training-free correction D * s**e (used when the zoom factor is known,
# e.g. from the EXIF DigitalZoomRatio tag); e is chosen on other datasets in the analysis
CORRECTION_EXPONENTS = (0.25, 0.5, 0.75, 1.0)
# finer grid (used by the dedicated ZoomFix run): alpha can exceed 1 (DA-V2-S on KITTI)
FINE_EXPONENTS = tuple(round(0.125 * k, 3) for k in range(1, 13))


def depth_errors(pred, gt, max_depth):
    m = (gt > 1e-3) & (gt <= max_depth) & np.isfinite(pred) & (pred > 1e-3)
    if m.sum() < 50:
        return dict(abs_rel=np.nan, delta1=np.nan, log_ratio_gt=np.nan, n_gt=int(m.sum()))
    p, g = pred[m], gt[m]
    ratio = np.maximum(p / g, g / p)
    return dict(abs_rel=float(np.mean(np.abs(p - g) / g)),
                delta1=float(np.mean(ratio < 1.25)),
                # median log(pred / gt): the global scale error, sign included
                log_ratio_gt=float(np.median(np.log(p / g))),
                n_gt=int(m.sum()))


def self_consistency(d_a, d_b):
    """Median log ratio of two depth maps on the same pixels (the global scale change), and the
    shape error left after removing it: median |log(d_a/d_b) - median log ratio|."""
    ok = np.isfinite(d_a) & np.isfinite(d_b) & (d_a > 1e-3) & (d_b > 1e-3)
    if ok.sum() <= 50:
        return np.nan, np.nan
    r = np.log(d_a[ok] / d_b[ok])
    m = float(np.median(r))
    return m, float(np.median(np.abs(r - m)))


def control_rows(model, sample, max_depth, centre=(0.5, 0.5)):
    """Confounder controls, each against the right reference:

    optical   : a camera with half the resolution of the dataset image is "zoomed" 2x optically
                by taking the same field of view from the full-resolution image (real detail);
    digital_lr: the same 2x zoom done digitally on the half-resolution image (crop + upsample);
                both are compared with the half-resolution camera's own unzoomed prediction,
                so optical vs digital_lr separates the field-of-view change from detail loss;
    bilinear  : the main 2x digital zoom with bilinear instead of bicubic resampling;
    jpeg      : JPEG quality 75 at 1x (no zoom; compression only).
    """
    img, gt, K = sample["img"], sample["depth"], sample["K"]
    h, w = img.shape[:2]
    rows = []

    def row(kind, zoom_, out, ref_out, d_s, d_ref, gt_box, fx_gt):
        lr, shape = self_consistency(d_s, d_ref)
        r = dict(name=sample["name"], kind=kind, zoom=zoom_, cx=centre[0], cy=centre[1], sec=np.nan,
                 log_ratio_self=lr, shape_err=shape, fx=out["fx"], fx_base=ref_out["fx"], fx_gt=fx_gt)
        r.update(depth_errors(d_s, gt_box, max_depth))
        rows.append(r)

    # half-resolution camera and its 2x zooms
    hc, wc = h // 2, w // 2
    cam = zoom.downsample(img, 2)
    K_cam = zoom.scale_K(K, (h, w), (hc, wc))
    base_cam = model.predict(cam, K_cam)
    box_c = zoom.zoom_box(hc, wc, 2.0, *centre)
    x0, y0, cw, ch = box_c
    ref = zoom.crop(base_cam["depth"], box_c)
    K_dig = zoom.zoom_K(K_cam, box_c, (hc, wc))
    dig = model.predict(zoom.zoom_image(cam, box_c), K_dig)
    box_full = (2 * x0, 2 * y0, 2 * cw, 2 * ch)
    opt_img = zoom.crop(img, box_full)
    if opt_img.shape[:2] != (hc, wc):  # odd image sizes: at most one pixel of resampling
        opt_img = zoom.resize_image(opt_img, (hc, wc))
    K_opt = zoom.zoom_K(K, box_full, (hc, wc))
    opt = model.predict(opt_img, K_opt)
    gt_box = zoom.crop(gt, box_full)
    for kind, out in (("optical", opt), ("digital_lr", dig)):
        d_box = zoom.unzoom_depth(out["depth"], box_c)                      # on the camera's crop grid
        d_full = zoom.resize_depth(out["depth"], gt_box.shape)              # for the GT errors
        lr, shape = self_consistency(d_box, ref)
        r = dict(name=sample["name"], kind=kind, zoom=2.0, cx=centre[0], cy=centre[1], sec=np.nan,
                 log_ratio_self=lr, shape_err=shape, fx=out["fx"], fx_base=base_cam["fx"], fx_gt=float(K_opt[0, 0]))
        r.update(depth_errors(d_full, gt_box, max_depth))
        rows.append(r)

    # interpolation and compression on the full-resolution image
    base = model.predict(img, K)
    box = zoom.zoom_box(h, w, 2.0, *centre)
    Kz = zoom.zoom_K(K, box, (h, w))
    bil = model.predict(zoom.zoom_image(img, box, mode="bilinear"), Kz)
    row("bilinear", 2.0, bil, base, zoom.unzoom_depth(bil["depth"], box), zoom.crop(base["depth"], box),
        zoom.crop(gt, box), float(Kz[0, 0]))
    jp = model.predict(zoom.jpeg(img, 75), K)
    row("jpeg", 1.0, jp, base, jp["depth"], base["depth"], gt, float(K[0, 0]))
    return rows


def zoom_rows(model, sample, max_depth, zooms=ZOOMS, center=(0.5, 0.5), stale_K=False, blur=(2.0,),
              exponents=CORRECTION_EXPONENTS):
    """stale_K: give a 'takes camera' model the unzoomed K (what you get if EXIF is not
    updated after a digital zoom). Ignored by models that do not take K."""
    img, gt, K = sample["img"], sample["depth"], sample["K"]
    h, w = img.shape[:2]
    base = model.predict(img, K)
    rows = []
    for s in zooms:
        box = zoom.zoom_box(h, w, s, *center)
        t0 = time.time()
        if s == 1.0:
            out = base
        else:
            Kz = K if stale_K else zoom.zoom_K(K, box, (h, w))
            out = model.predict(zoom.zoom_image(img, box), Kz)
        sec = time.time() - t0
        d_s = zoom.unzoom_depth(out["depth"], box) if s != 1.0 else out["depth"]
        d_1 = zoom.crop(base["depth"], box)
        gt_box = zoom.crop(gt, box)
        lr, shape = self_consistency(d_s, d_1)
        row = dict(name=sample["name"], kind="zoom", zoom=s, cx=center[0], cy=center[1], sec=sec,
                   # label-free: how much the depth of the same pixels moved because of zoom, and how
                   # much their relative shape changed once that global factor is removed
                   log_ratio_self=lr, shape_err=shape,
                   fx=out["fx"], fx_base=base["fx"], fx_gt=float(zoom.zoom_K(K, box, (h, w))[0, 0]))
        row.update(depth_errors(d_s, gt_box, max_depth))
        # the unzoomed prediction scored on exactly the same pixels: the fair reference
        row.update({f"{k}_base": v for k, v in depth_errors(d_1, gt_box, max_depth).items()})
        # principal point: predicted vs where the law puts it ("cx" is already the crop centre)
        Kz_true = zoom.zoom_K(K, box, (h, w))
        row.update(pp_x=out.get("cx"), pp_x_base=base.get("cx"),
                   pp_x_gt=float(Kz_true[0, 2]), pp_x_gt_base=float(K[0, 2]))
        if s != 1.0:
            s_true = Kz_true[0, 0] / K[0, 0]
            for e in exponents:
                row[f"abs_rel_fix{e}"] = depth_errors(d_s * s_true ** e, gt_box, max_depth)["abs_rel"]
        rows.append(row)
    # control: same detail loss, no change of field of view
    for s in blur:
        out = model.predict(zoom.blur_control(img, s), K)
        ok = np.isfinite(out["depth"]) & np.isfinite(base["depth"]) & (out["depth"] > 1e-3) & (base["depth"] > 1e-3)
        row = dict(name=sample["name"], kind="blur", zoom=s, cx=center[0], cy=center[1], sec=np.nan,
                   log_ratio_self=float(np.median(np.log(out["depth"][ok] / base["depth"][ok]))),
                   fx=out["fx"], fx_base=base["fx"], fx_gt=float(K[0, 0]))
        row.update(depth_errors(out["depth"], gt, max_depth))
        rows.append(row)
    return rows


def paste_crop(img, box):
    """The crop at its own place and scale on a plain canvas (the image's mean colour).

    The camera does not change, so K stays the same; only the context around the crop is gone.
    """
    x0, y0, cw, ch = box
    out = np.empty_like(img)
    out[:] = img.reshape(-1, img.shape[-1]).mean(0).astype(img.dtype)
    out[y0:y0 + ch, x0:x0 + cw] = img[y0:y0 + ch, x0:x0 + cw]
    return out


def shrink_image(img, s):
    """The whole image made s times smaller and centred on a plain canvas: a zoom *out* that
    keeps all the context. Returns the image and the box where the shrunk image sits."""
    h, w = img.shape[:2]
    box = zoom.zoom_box(h, w, s)
    x0, y0, cw, ch = box
    out = np.empty_like(img)
    out[:] = img.reshape(-1, img.shape[-1]).mean(0).astype(img.dtype)
    out[y0:y0 + ch, x0:x0 + cw] = zoom.resize_image(img, (ch, cw))
    return out, box


def shrink_K(K, box, in_hw):
    """Intrinsics of shrink_image: the inverse map of zoom_K on that box."""
    x0, y0, cw, ch = box
    H, W = in_hw
    sx, sy = cw / W, ch / H
    K = np.array(K, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5 + x0
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5 + y0
    return K


def cue_rows(model, sample, max_depth, s=2.0):
    """Split a digital zoom into the two things it changes, one at a time, with the right K each:

    context : the crop pasted at its own place and scale (context gone, pixels and camera the same);
    magnify : the zoomed image against the pasted crop (same content, s times larger);
    zoom    : the full digital zoom, the sum of the two in log depth;
    shrink  : the whole image s times smaller on a canvas (smaller, all context kept).

    A model that respects the camera keeps the depth of every pixel in all four.
    """
    img, gt, K = sample["img"], sample["depth"], sample["K"]
    h, w = img.shape[:2]
    box = zoom.zoom_box(h, w, s)
    base = model.predict(img, K)
    d_base = zoom.crop(base["depth"], box)
    Kz = zoom.zoom_K(K, box, (h, w))
    zoomed = model.predict(zoom.zoom_image(img, box), Kz)
    d_zoom = zoom.unzoom_depth(zoomed["depth"], box)
    pasted = model.predict(paste_crop(img, box), K)
    d_paste = zoom.crop(pasted["depth"], box)
    small, sbox = shrink_image(img, s)
    Ks = shrink_K(K, sbox, (h, w))
    shrunk = model.predict(small, Ks)
    d_shrink = zoom.resize_depth(zoom.crop(shrunk["depth"], sbox), (h, w))
    gt_box = zoom.crop(gt, box)
    rows = []
    for kind, d, ref, out, ref_out, fx_gt, g in (
            ("context", d_paste, d_base, pasted, base, float(K[0, 0]), gt_box),
            ("magnify", d_zoom, d_paste, zoomed, pasted, float(Kz[0, 0]), gt_box),
            ("zoom", d_zoom, d_base, zoomed, base, float(Kz[0, 0]), gt_box),
            ("shrink", d_shrink, base["depth"], shrunk, base, float(Ks[0, 0]), gt)):
        lr, shape = self_consistency(d, ref)
        r = dict(name=sample["name"], kind=kind, zoom=s, cx=0.5, cy=0.5, sec=np.nan, log_ratio_self=lr,
                 shape_err=shape, fx=out["fx"], fx_base=ref_out["fx"], fx_gt=fx_gt)
        r.update(depth_errors(d, g, max_depth))
        rows.append(r)
    return rows


def run(model_name, dataset, n, out_dir, zooms=ZOOMS, centers=((0.5, 0.5),), stale_K=False, log_every=10, suffix="",
        controls_only=False, fine_fix=False, cues=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}{'-staleK' if stale_K else ''}__{dataset}{suffix}"
    out_csv = out_dir / f"{tag}.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["name"], prev["cx"], prev["cy"]))

    load = datasets.Loader(dataset)
    ids = datasets.sample_ids(dataset, n)
    model = models.load(model_name, datasets.DOMAIN[dataset])
    t0 = time.time()
    for k, i in enumerate(ids):
        sample = load(i)
        todo = [c for c in centers if (sample["name"], c[0], c[1]) not in done]
        for c in todo:
            if cues:
                rows = cue_rows(model, sample, datasets.MAX_DEPTH[dataset])
            elif controls_only:
                rows = control_rows(model, sample, datasets.MAX_DEPTH[dataset], c)
            else:
                # the blur control only needs to run once per image
                blur = (2.0,) if c == centers[0] else ()
                rows = zoom_rows(model, sample, datasets.MAX_DEPTH[dataset], zooms, c, stale_K, blur,
                                 FINE_EXPONENTS if fine_fix else CORRECTION_EXPONENTS)
            for r in rows:
                r.update(model=model_name + ("-staleK" if stale_K else ""), dataset=dataset)
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        if log_every and (k + 1) % log_every == 0:
            print(f"{tag}: {k + 1}/{len(ids)} images, {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_csv
