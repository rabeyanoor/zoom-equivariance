"""Per-image scene statistics for the audited images, and depth maps for Figure 1.

    python scripts/scene_stats.py --n 200 --out outputs/results

depth_spread = IQR of log ground-truth depth. The zoom-squash argument (paper, Sec. 3)
predicts that equivariance improves with depth spread, so we need it per image.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
p.add_argument("--figure1", action="store_true", help="also save depth maps for Figure 1")
p.add_argument("--figure-only", action="store_true")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

rows = []
for ds in ([] if args.figure_only else ["kitti", "nyu", "diode-in", "diode-out"]):
    load = datasets.Loader(ds)
    for i in datasets.sample_ids(ds, args.n):
        s = load(i)
        g = s["depth"]
        g = np.log(g[(g > 1e-3) & (g <= datasets.MAX_DEPTH[ds])])
        q25, q50, q75 = np.percentile(g, [25, 50, 75]) if len(g) else (np.nan,) * 3
        rows.append(dict(dataset=ds, name=s["name"], depth_spread=q75 - q25, median_depth=float(np.exp(q50)),
                         gt_coverage=len(g) / s["depth"].size))
    print(ds, "done", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(out / "scene_stats.csv", index=False)

if args.figure1 or args.figure_only:
    import models
    import zoom

    picks = {"kitti": datasets.sample_ids("kitti", args.n)[3], "nyu": datasets.sample_ids("nyu", args.n)[5]}
    maps = {}
    for name in ("dav2-s", "moge2-s", "metric3d-s"):
        for ds, i in picks.items():
            m = models.load(name, datasets.DOMAIN[ds])
            s = datasets.Loader(ds)(i)
            h, w = s["img"].shape[:2]
            for zf in (1.0, 2.0, 3.0):
                box = zoom.zoom_box(h, w, zf)
                img = zoom.zoom_image(s["img"], box) if zf > 1 else s["img"]
                maps[f"{name}|{ds}|{zf}"] = m.predict(img, zoom.zoom_K(s["K"], box, (h, w)))["depth"].astype(np.float16)
                maps[f"img|{ds}|{zf}"] = img
                maps[f"gt|{ds}|{zf}"] = zoom.crop(s["depth"], box).astype(np.float16)
            del m
        print(name, "figure maps done", flush=True)
    np.savez_compressed(out / "figure1_maps.npz", **maps)
