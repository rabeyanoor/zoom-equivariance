"""Zoom response by semantic region: does depth shrink more on objects of familiar size (people, cars,
furniture) than on surfaces (road, floor, walls)? An apparent-size reader would key on the first.

Each image is segmented once (SegFormer-B5, ADE20K classes); for each zoom the depth ratio
zoomed/unzoomed on the same pixels is summarised per region group, as in audit.zoom_rows.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import datasets
import models
import zoom

SEGMENTER = "nvidia/segformer-b5-finetuned-ade-640-640"
# ADE20K class ids (config.json id2label of the checkpoint above)
GROUPS = {
    "object": [12, 20, 83, 80, 102, 127, 116, 19, 15, 7, 23, 10, 14, 30, 33, 36, 98],  # person, car, truck, bus,
    # van, bicycle, minibike, chair, table, bed, sofa, cabinet, door, armchair, desk, lamp, bottle
    "surface": [0, 1, 3, 5, 6, 11, 9, 13],  # wall, building, floor, ceiling, road, sidewalk, grass, earth
}
ZOOMS = (2.0, 3.0)
MIN_PIXELS = 200


class Segmenter:
    def __init__(self):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.proc = AutoImageProcessor.from_pretrained(SEGMENTER)
        self.net = SegformerForSemanticSegmentation.from_pretrained(SEGMENTER).eval().to(models.DEVICE)

    @torch.no_grad()
    def __call__(self, img):
        x = self.proc(images=img, return_tensors="pt").to(models.DEVICE)
        logits = self.net(**x).logits
        up = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()


def group_map(labels):
    out = np.full(labels.shape, "", dtype=object)
    for g, ids in GROUPS.items():
        out[np.isin(labels, ids)] = g
    return out


def region_rows(model, sample, seg, zooms=ZOOMS, centre=(0.5, 0.5)):
    img, K = sample["img"], sample["K"]
    h, w = img.shape[:2]
    base = model.predict(img, K)
    groups = group_map(seg)
    rows = []
    for s in zooms:
        box = zoom.zoom_box(h, w, s, *centre)
        Kz = zoom.zoom_K(K, box, (h, w))
        out = model.predict(zoom.zoom_image(img, box), Kz)
        d_s, d_1, g = zoom.unzoom_depth(out["depth"], box), zoom.crop(base["depth"], box), zoom.crop(groups, box)
        ok = np.isfinite(d_s) & np.isfinite(d_1) & (d_s > 1e-3) & (d_1 > 1e-3)
        for name in GROUPS:
            m = ok & (g == name)
            lr = float(np.median(np.log(d_s[m] / d_1[m]))) if m.sum() >= MIN_PIXELS else np.nan
            rows.append(dict(name=sample["name"], zoom=s, s_true=float(Kz[0, 0] / K[0, 0]), group=name,
                             n_px=int(m.sum()), log_ratio_self=lr))
    return rows


def run(model_name, dataset, n, out_dir, every=1, log_every=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}__{dataset}__regions.csv"
    done = set(pd.read_csv(out).name) if out.exists() else set()
    load = datasets.Loader(dataset)
    model, seg = models.load(model_name, datasets.DOMAIN[dataset]), Segmenter()
    t0 = time.time()
    for k, i in enumerate(datasets.sample_ids(dataset, n)[::every]):
        sample = load(i)
        if sample["name"] in done:
            continue
        rows = region_rows(model, sample, seg(sample["img"]))
        for r in rows:
            r.update(model=model_name, dataset=dataset)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        if k % log_every == 0:
            print(f"{model_name} {dataset} {k} {time.time() - t0:.0f}s", flush=True)
    return out
