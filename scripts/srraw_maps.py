"""Depth maps of one SR-RAW sequence for the optical-zoom figure (notebooks/08_srraw_figure.py).

    python scripts/srraw_maps.py --seq 00134 --models dav2-s dav2-s-eftb depthpro depthpro-eftf

The sequence is chosen outside this script by a stated rule (the one with the median alpha of DA-V2-S), so the
figure is not a hand-picked example. Saves the shots (s <= S_MAX) and every model's depth on each of them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import models  # noqa: E402
import srraw  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seq", required=True, help="sequence folder name")
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--root", default=str(ROOT / "data" / "raw" / "srraw"))
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
args = p.parse_args()

seq = [d for d in srraw.sequences(args.root) if d.name == args.seq]
assert len(seq) == 1, f"sequence {args.seq} found {len(seq)} times"
shots = sorted((srraw.read_shot(q) for q in sorted(seq[0].glob("0000[1-7].JPG"))), key=lambda t: t[1])
f0 = shots[0][1]
shots = [t for t in shots if t[1] / f0 <= srraw.S_MAX]
maps = {f"img|{f:g}": img for img, f, _ in shots}
for name in args.models:
    m = models.load(name, "Outdoor")
    needs_K = models.REGISTRY[name][0].startswith("takes camera")
    for img, f, f35 in shots:
        maps[f"{name}|{f:g}"] = m.predict(img, K=srraw.K_from_f35(f35, *img.shape[:2]) if needs_K else None)[
            "depth"].astype(np.float16)
    del m
    print(name, "done", flush=True)
out = Path(args.out) / f"srraw_maps_{args.seq}.npz"
np.savez_compressed(out, focals=np.array([f for _, f, _ in shots]), **maps)
print("wrote", out, flush=True)
