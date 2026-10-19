"""Parameter count of every model wrapper (for the model-ladder figure and the efficiency table).

    python scripts/count_params.py --out outputs/results
"""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import models  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
p.add_argument("--only", nargs="*")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
path = out / "params.json"
counts = json.loads(path.read_text()) if path.exists() else {}
for name in args.only or models.REGISTRY:
    if name in counts or name.startswith("dav2-s-"):
        continue
    try:
        m = models.load(name, "Outdoor")
        counts[name] = sum(p.numel() for p in m.model.parameters())
        print(f"{name:14s} {counts[name] / 1e6:8.1f} M", flush=True)
    except Exception as e:  # one model failing should not stop the table
        print(f"{name}: {e}", flush=True)
    finally:
        m = None
        gc.collect()
    path.write_text(json.dumps(counts, indent=1))
