"""Real optical zoom on SR-RAW for one model (src/srraw.py).

    python scripts/run_srraw.py dav2-s --root data/raw/srraw/test
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import srraw  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--root", default=str(ROOT / "data" / "raw" / "srraw" / "test"))
p.add_argument("--every", type=int, default=1, help="every k-th sequence")
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
args = p.parse_args()
print("wrote", srraw.run(args.model, args.root, args.out, every=args.every), flush=True)
