"""Run the zoom audit for one model on one or more datasets.

    python scripts/run_audit.py moge2-s kitti nyu diode-in diode-out --n 100

Heavy part of the project, so it lives in a script (runs in the background, resumable);
the notebooks only read the CSVs it writes to outputs/results/.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import audit  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("datasets", nargs="+")
p.add_argument("--n", type=int, default=100)
p.add_argument("--stale-k", action="store_true")
p.add_argument("--off-centre", action="store_true", help="also zoom into the four quadrants")
p.add_argument("--fine-fix", action="store_true", help="ZoomFix exponents 0.125..1.5")
p.add_argument("--controls-only", action="store_true", help="optical-vs-digital, bilinear and JPEG controls")
p.add_argument("--cues", action="store_true", help="split the 2x zoom into context loss and magnification")
p.add_argument("--every", type=int, default=1, help="every k-th image of the --n sample (paired subset)")
p.add_argument("--shard", type=int, default=0, help="run only images shard, shard+shards, ... of the sample")
p.add_argument("--shards", type=int, default=1)
p.add_argument("--out-tag", default="", help="suffix for the CSV name, e.g. offc")
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
args = p.parse_args()

centres = [(0.5, 0.5)]
if args.off_centre:
    centres += [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

for ds in args.datasets:
    out = audit.run(args.model, ds, args.n, args.out, centers=tuple(centres), stale_K=args.stale_k,
                    suffix=f"__{args.out_tag}" if args.out_tag else "", controls_only=args.controls_only,
                    fine_fix=args.fine_fix, cues=args.cues, every=args.every,
                    shard=args.shard, shards=args.shards)
    print("wrote", out, flush=True)
