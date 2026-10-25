"""Zoom response by semantic region (src/regions.py).

    python scripts/run_regions.py dav2-s kitti nyu --n 200
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import regions  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("datasets", nargs="+")
p.add_argument("--n", type=int, default=200)
p.add_argument("--every", type=int, default=1)
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
args = p.parse_args()
for ds in args.datasets:
    print("wrote", regions.run(args.model, ds, args.n, args.out, every=args.every), flush=True)
