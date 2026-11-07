"""EFT-S without back-propagating through the frozen model: cache, then fit the head.

The EFT-S student is D0(x) * exp(c(x)) with D0 frozen, and the EFT+B loss of one view is

    |c_x|  +  mean_p |r_z(p) + c_z|  +  mean_p |r_b(p) + c_b|

with r_z = log D0(zoomed) on the crop grid - log D0(x) in the crop, and r_b = log D0(blurred) -
log D0(x). c is one number per image, so each mean only needs the distribution of r: its quantiles
give the loss for any c. One pass of the frozen model per view therefore stores everything, and the
head trains on the cache alone (the full EFT trainer would run the frozen model ~10^4 times
per run on a CPU).

    python scripts/eft_scale.py cache outdoor --images "data/raw/kitti_raw/*/*/image_02/data/*.png" --shard 0 --shards 8
    python scripts/eft_scale.py cache indoor --nyu-train --shard 0 --shards 8
    python scripts/eft_scale.py fit outdoor --cache outputs/eft/cache --w-blur 1 --tag scale
"""
import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import eft  # noqa: E402

Q = 100  # residual quantiles per view


def base_model(domain):
    from transformers import AutoModelForDepthEstimation

    name = "Outdoor" if domain == "outdoor" else "Indoor"
    return AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-Metric-{name}-Small-hf")


def training_images(args):
    """(get(i), count) for the unlabeled training images; RGB only, as in train_eft.py."""
    if args.nyu_train:
        import h5py
        from scipy.io import loadmat

        raw = ROOT / "data" / "raw" / "nyu"
        f = h5py.File(raw / "nyu_depth_v2_labeled.mat", "r")
        train = loadmat(raw / "splits.mat")["trainNdxs"].ravel() - 1
        return (lambda i: np.ascontiguousarray(np.array(f["images"][int(train[i])]).transpose(2, 1, 0))), len(train)
    from PIL import Image

    paths = sorted(glob.glob(args.images))
    return (lambda i: np.array(Image.open(paths[i]).convert("RGB"))), len(paths)


def quantiles(r):
    return torch.quantile(r.flatten().float(), (torch.arange(Q) + 0.5) / Q).numpy()


@torch.no_grad()
def cache(args):
    get, n_imgs = training_images(args)
    model = base_model(args.domain).to(eft.DEVICE).eval()
    h0, w0 = get(0).shape[:2]
    h, w = eft.input_hw(h0, w0)
    rng = np.random.default_rng(1000 + args.shard)
    n = args.views // args.shards
    feats, rz, rb, zs, bs = [], [], [], [], []
    t0 = time.time()
    for k in range(n):
        img = get(int(rng.integers(0, n_imgs)))
        if rng.random() < 0.5:
            img = img[:, ::-1]
        x = eft.to_input(np.ascontiguousarray(img), (h, w))
        # the same augmentations as eft.EFT.step
        s, (x0, y0, cw, ch) = eft.random_box(h, w, rng, args.s_max)
        z = F.interpolate(x[..., y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        sb = math.exp(rng.uniform(0, math.log(args.s_max)))
        small = F.interpolate(x, size=(max(14, int(h / sb)), max(14, int(w / sb))), mode="area")
        b = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        out = model(pixel_values=torch.cat([x, z, b]), output_hidden_states=True)
        log0 = eft.log_depth_of(out.predicted_depth.float()).cpu()
        back = F.interpolate(log0[1:2], size=(ch, cw), mode="area")
        rz.append(quantiles(back - log0[0:1, :, y0:y0 + ch, x0:x0 + cw]))
        rb.append(quantiles(log0[2] - log0[0]))
        feats.append(eft.ScaleAdapter.features(out.hidden_states[-1]).cpu().numpy())
        zs.append(s)
        bs.append(sb)
        if (k + 1) % 50 == 0:
            print(f"{args.domain} shard {args.shard}: {k + 1}/{n} views, {(time.time() - t0) / 60:.1f} min", flush=True)
    out_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.domain}_{args.shard:02d}.npz", feats=np.stack(feats).astype(np.float32),
                        rz=np.stack(rz).astype(np.float32), rb=np.stack(rb).astype(np.float32),
                        s=np.array(zs), sb=np.array(bs))


def load_cache(folder, domain):
    parts = [np.load(f) for f in sorted(Path(folder).glob(f"{domain}_*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def view_loss(c, rz, rb, w_zoom, w_anchor, w_blur):
    """Per-view EFT+B loss. c: N x 3 log-scales for (original, zoomed, blurred); rz, rb: N x Q."""
    loss = w_anchor * c[:, 0].abs() + w_zoom * (rz + c[:, 1:2]).abs().mean(1)
    if w_blur:
        loss = loss + w_blur * (rb + c[:, 2:3]).abs().mean(1)
    return loss


def fit(args):
    torch.manual_seed(0)
    data = load_cache(args.cache, args.domain)
    feats = torch.from_numpy(data["feats"])                      # N x 3 x 2dim
    rz, rb = torch.from_numpy(data["rz"]), torch.from_numpy(data["rb"])
    n = len(feats)
    perm = torch.randperm(n)
    val, tr = perm[: n // 10], perm[n // 10:]                    # held-out views: label-free loss only
    adapter = eft.ScaleAdapter(base_model(args.domain))
    head = adapter.head
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    w = dict(w_zoom=args.w_zoom, w_anchor=args.w_anchor, w_blur=args.w_blur)

    def loss_on(idx):
        return view_loss(head(feats[idx]).squeeze(-1), rz[idx], rb[idx], **w).mean()

    with torch.no_grad():
        print(f"frozen: train {loss_on(tr):.4f}  val {loss_on(val):.4f}", flush=True)
    log = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        for idx in tr[torch.randperm(len(tr))].split(args.batch):
            loss = loss_on(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            log.append(dict(epoch=epoch, train=float(loss_on(tr)), val=float(loss_on(val))))
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train {log[-1]['train']:.4f}  val {log[-1]['val']:.4f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), out / f"dav2-s-{args.tag}-{args.domain}.pt")
    import pandas as pd

    pd.DataFrame(log).to_csv(out / f"train_log_{args.tag}_{args.domain}.csv", index=False)
    print("saved", out, flush=True)


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["cache", "fit"])
p.add_argument("domain", choices=["outdoor", "indoor"])
p.add_argument("--images", help="glob of unlabeled training images")
p.add_argument("--nyu-train", action="store_true", help="RGB of the 795 official NYUv2 train images")
p.add_argument("--views", type=int, default=4000, help="random zoom/blur views in total (all shards)")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--shards", type=int, default=1)
p.add_argument("--s-max", type=float, default=3.0)
p.add_argument("--cache", default=str(ROOT / "outputs" / "eft" / "cache"))
p.add_argument("--w-zoom", type=float, default=1.0)
p.add_argument("--w-anchor", type=float, default=1.0)
p.add_argument("--w-blur", type=float, default=0.0)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--tag", default="scale")
p.add_argument("--out", default=str(ROOT / "outputs" / "eft"))
args = p.parse_args()
(cache if args.cmd == "cache" else fit)(args)
