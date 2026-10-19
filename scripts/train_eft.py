"""Equivariance Fine-Tuning of DA-V2-S metric on unlabeled images.

    python scripts/train_eft.py outdoor --images "data/raw/kitti_raw/*/*/image_02/data/*.png"
    python scripts/train_eft.py indoor  --nyu-train

Only RGB is read. Depth labels of the training images are never loaded.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import eft  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("domain", choices=["outdoor", "indoor"])
p.add_argument("--images", help="glob of unlabeled training images")
p.add_argument("--nyu-train", action="store_true", help="RGB of the 795 official NYUv2 train images")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--w-anchor", type=float, default=1.0)
p.add_argument("--w-zoom", type=float, default=1.0, help="0 = anchor-only baseline")
p.add_argument("--w-blur", type=float, default=0.0, help="blur-invariance term (EFT+B)")
p.add_argument("--target", default="teacher", choices=["teacher", "student"],
               help="student = Zhong et al. style self-consistency target")
p.add_argument("--tag", default="eft", help="checkpoint name: dav2-s-<tag>-<domain>.pt")
p.add_argument("--s-max", type=float, default=3.0)
p.add_argument("--out", default=str(ROOT / "outputs" / "eft"))
args = p.parse_args()

if args.nyu_train:
    import h5py
    from scipy.io import loadmat

    raw = ROOT / "data" / "raw" / "nyu"
    f = h5py.File(raw / "nyu_depth_v2_labeled.mat", "r")
    train = loadmat(raw / "splits.mat")["trainNdxs"].ravel() - 1
    images = [np.ascontiguousarray(np.array(f["images"][int(j)]).transpose(2, 1, 0)) for j in train]
    get = lambda i: images[i]  # noqa: E731
    n_imgs = len(images)
else:
    paths = sorted(glob.glob(args.images))
    get = lambda i: np.array(Image.open(paths[i]).convert("RGB"))  # noqa: E731
    n_imgs = len(paths)
print(f"{n_imgs} unlabeled training images", flush=True)

from transformers import AutoModelForDepthEstimation  # noqa: E402

name = "Outdoor" if args.domain == "outdoor" else "Indoor"
model = AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-Metric-{name}-Small-hf")
h0, w0 = get(0).shape[:2]
trainer = eft.EFT(model, eft.input_hw(h0, w0), w_anchor=args.w_anchor, w_zoom=args.w_zoom, w_blur=args.w_blur,
                  s_max=args.s_max, target=args.target)

rng = np.random.default_rng(0)
log, t0 = [], time.time()
for step in range(1, args.steps + 1):
    batch = [get(int(i)) for i in rng.integers(0, n_imgs, args.batch)]
    l_zoom, l_anchor = trainer.step(batch)
    log.append(dict(step=step, l_zoom=l_zoom, l_anchor=l_anchor))
    if step % 100 == 0:
        recent = pd.DataFrame(log[-100:]).mean()
        print(f"step {step}: zoom {recent.l_zoom:.4f}  anchor {recent.l_anchor:.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save(trainer.student.cpu().state_dict(), out / f"dav2-s-{args.tag}-{args.domain}.pt")
pd.DataFrame(log).to_csv(out / f"train_log_{args.tag}_{args.domain}.csv", index=False)
print("saved", out, flush=True)
