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


def zoom_rows(model, sample, max_depth, zooms=ZOOMS, center=(0.5, 0.5), stale_K=False, blur=(2.0,)):
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
        ok = np.isfinite(d_s) & np.isfinite(d_1) & (d_s > 1e-3) & (d_1 > 1e-3)
        row = dict(name=sample["name"], kind="zoom", zoom=s, cx=center[0], cy=center[1], sec=sec,
                   # label-free: how much the depth of the same pixels moved because of zoom
                   log_ratio_self=float(np.median(np.log(d_s[ok] / d_1[ok]))) if ok.sum() > 50 else np.nan,
                   fx=out["fx"], fx_base=base["fx"], fx_gt=float(zoom.zoom_K(K, box, (h, w))[0, 0]))
        row.update(depth_errors(d_s, gt_box, max_depth))
        # the unzoomed prediction scored on exactly the same pixels: the fair reference
        row.update({f"{k}_base": v for k, v in depth_errors(d_1, gt_box, max_depth).items()})
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


def run(model_name, dataset, n, out_dir, zooms=ZOOMS, centers=((0.5, 0.5),), stale_K=False, log_every=10, suffix=""):
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
            # the blur control only needs to run once per image
            blur = (2.0,) if c == centers[0] else ()
            rows = zoom_rows(model, sample, datasets.MAX_DEPTH[dataset], zooms, c, stale_K, blur)
            for r in rows:
                r.update(model=model_name + ("-staleK" if stale_K else ""), dataset=dataset)
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        if log_every and (k + 1) % log_every == 0:
            print(f"{tag}: {k + 1}/{len(ids)} images, {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_csv
