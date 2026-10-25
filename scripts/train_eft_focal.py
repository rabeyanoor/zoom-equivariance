"""EFT-F: tune Depth Pro's focal head with the focal-consistency loss of src/eft_focal.py.

    python scripts/train_eft_focal.py --kitti "data/raw/kitti_raw/*/*/image_02/data/*.png" --nyu-train

Only RGB of unlabeled training images is read (the KITTI raw drives disjoint from the evaluation drives and
the official NYUv2 train split, as for EFT). One model covers both domains, so the two sets are mixed.
Writes <out>/depthpro-eftf.pt (the tuned layers only) and <out>/train_log_depthpro-eftf.csv.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import eft_focal as ef  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--kitti", help="glob of unlabeled KITTI raw frames")
p.add_argument("--nyu-train", action="store_true", help="RGB of the 795 official NYUv2 train images")
p.add_argument("--n-per-set", type=int, default=300, help="training images per set, evenly spaced")
p.add_argument("--views", type=int, default=4, help="random views per image")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--w-blur", type=float, default=1.0)
p.add_argument("--w-anchor", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="depthpro-eftf")
p.add_argument("--repo", default="apple/DepthPro-hf")
p.add_argument("--out", default=str(ROOT / "outputs" / "eft"))
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)


def evenly(items, n):
    idx = np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int)
    return [items[i] for i in idx]


getters = []
if args.kitti:
    from PIL import Image

    getters += [lambda q=q: np.array(Image.open(q).convert("RGB")) for q in evenly(sorted(glob.glob(args.kitti)), args.n_per_set)]
if args.nyu_train:
    import h5py
    from scipy.io import loadmat

    raw = ROOT / "data" / "raw" / "nyu"
    f = h5py.File(raw / "nyu_depth_v2_labeled.mat", "r")
    train = evenly(list(loadmat(raw / "splits.mat")["trainNdxs"].ravel() - 1), args.n_per_set)
    getters += [lambda j=j: np.ascontiguousarray(np.array(f["images"][int(j)]).transpose(2, 1, 0)) for j in train]
order = rng.permutation(len(getters))
n_val = max(1, len(getters) // 10)
print(f"{len(getters)} unlabeled training images ({n_val} held out)", flush=True)

from transformers import AutoImageProcessor, DepthProForDepthEstimation  # noqa: E402

proc = AutoImageProcessor.from_pretrained(args.repo)
# the hub weights load in fp16; the tuned layers are trained in fp32 (the cache is still made under fp16 autocast)
model = DepthProForDepthEstimation.from_pretrained(args.repo).float().eval().to(DEVICE)
fm = model.fov_model
amp = dict(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda")


def pixels(img):
    return proc(images=img, return_tensors="pt").pixel_values.to(DEVICE)


# the cached path must reproduce the model's own field of view
with torch.no_grad():
    pv = pixels(getters[order[0]]())
    full = model(pixel_values=pv).field_of_view
    g, tok = ef.encode(model, pv)
    mine = ef.fov_from_cache(fm, g, tok, *pv.shape[-2:])
print(f"check: model fov {full.item():.4f} deg, cached path {mine.item():.4f} deg", flush=True)
assert abs(full.item() - mine.item()) < 1e-3, "fov_from_cache does not match the model"

t0 = time.time()
cache = {"g": [], "tok": [], "target": [], "kind": [], "val": []}
KIND = {"plain": 0, "blur": 1, "zoom": 2}
for n, k in enumerate(order):
    img = getters[k]()
    with torch.no_grad(), torch.autocast(**amp):
        pv = pixels(img)
        t_img = ef.log_tan_half(ef.fov_from_cache(fm, *ef.encode(model, pv), *pv.shape[-2:])).float().item()
        for _ in range(args.views):
            view, kind, s = ef.random_view(img, rng)
            pv = pixels(view)
            g, tok = ef.encode(model, pv)
            cache["g"].append(g.half().cpu())
            cache["tok"].append(tok.half().cpu())
            cache["target"].append(t_img - np.log(s))
            cache["kind"].append(KIND[kind])
            cache["val"].append(n < n_val)
    if (n + 1) % 50 == 0:
        print(f"cached {n + 1}/{len(order)} images, {(time.time() - t0) / 60:.1f} min", flush=True)
H, W = pv.shape[-2:]
G, TOK = torch.cat(cache["g"]), torch.cat(cache["tok"])
target = torch.tensor(cache["target"], dtype=torch.float32)
kind = torch.tensor(cache["kind"])
val = torch.tensor(cache["val"])
tr_idx, va_idx = torch.where(~val)[0], torch.where(val)[0]
print(f"cache: {len(target)} views, {(G.nbytes + TOK.nbytes) / 2 ** 30:.1f} GB, {(time.time() - t0) / 60:.1f} min", flush=True)

for p_ in model.parameters():
    p_.requires_grad_(False)
params = ef.trainable_parameters(fm)
for p_ in params.values():
    p_.requires_grad_(True)
opt = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=1e-4)
print(f"tuning {sum(p_.numel() for p_ in params.values()) / 1e6:.2f}M parameters", flush=True)


def evaluate(idx):
    errs = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for b in idx.split(args.batch):
            pred = ef.log_tan_half(ef.fov_from_cache(fm, G[b].to(DEVICE).float(), TOK[b].to(DEVICE).float(), H, W))
            e = (pred.cpu() - target[b]).abs()
            for c in errs:
                errs[c] += e[kind[b] == c].tolist()
    return {c: float(np.mean(v)) if v else np.nan for c, v in errs.items()}


log = []
e0 = evaluate(va_idx)
log.append(dict(epoch=0, train_loss=np.nan, val_zoom=e0[2], val_blur=e0[1], val_plain=e0[0]))
print(f"epoch 0: val |err| zoom {e0[2]:.4f} blur {e0[1]:.4f} plain {e0[0]:.4f}", flush=True)
fm.train()
for ep in range(1, args.epochs + 1):
    perm = tr_idx[torch.randperm(len(tr_idx))]
    tot = 0.0
    for b in perm.split(args.batch):
        pred = ef.log_tan_half(ef.fov_from_cache(fm, G[b].to(DEVICE).float(), TOK[b].to(DEVICE).float(), H, W))
        loss, _ = ef.loss_terms(pred, target[b].to(DEVICE), kind[b].to(DEVICE), args.w_blur, args.w_anchor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    fm.eval()
    e = evaluate(va_idx)
    fm.train()
    log.append(dict(epoch=ep, train_loss=tot / len(tr_idx), val_zoom=e[2], val_blur=e[1], val_plain=e[0]))
    print(f"epoch {ep}: train {tot / len(tr_idx):.4f}, val |err| zoom {e[2]:.4f} blur {e[1]:.4f} plain {e[0]:.4f}",
          flush=True)
fm.eval()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({n: p_.detach().cpu() for n, p_ in params.items()}, out / f"{args.tag}.pt")
pd.DataFrame(log).to_csv(out / f"train_log_{args.tag}.csv", index=False)
print("wrote", out / f"{args.tag}.pt", f"in {(time.time() - t0) / 60:.1f} min", flush=True)
