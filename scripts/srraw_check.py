"""Inventory of the SR-RAW sequences: which shots the optical-zoom audit (src/srraw.py) can use, and why not.

    python scripts/srraw_check.py --root data/raw/srraw

One row per shot: sequence, file, EXIF focal (mm), 35 mm-equivalent focal, working size, s against the widest
shot, and the reason the shot is not compared (empty if it is). Writes srraw_inventory.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import srraw  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default=str(ROOT / "data" / "raw" / "srraw"))
p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
args = p.parse_args()

rows = []
for seq in srraw.sequences(args.root):
    shots = []
    for q in sorted(seq.glob("0000[1-7].JPG")):
        img, f, f35 = srraw.read_shot(q)
        shots.append(dict(seq=seq.name, path=str(q.relative_to(args.root)), file=q.name, f_mm=f, f35=f35,
                          h=img.shape[0], w=img.shape[1]))
    ok = [s for s in shots if np.isfinite(s["f_mm"])]
    base = min(ok, key=lambda s: s["f_mm"]) if len(ok) >= 2 else None
    for s in shots:
        s["s"] = s["f_mm"] / base["f_mm"] if base else np.nan
        if not np.isfinite(s["f_mm"]):
            s["reason"] = "no EXIF focal"
        elif base is None:
            s["reason"] = "fewer than 2 shots with a focal"
        elif s is base:
            s["reason"] = "base"
        elif s["s"] > srraw.S_MAX:
            s["reason"] = "s > S_MAX"
        elif (s["h"], s["w"]) != (base["h"], base["w"]):
            s["reason"] = "size differs from base"
        else:
            s["reason"] = ""
    rows += shots
df = pd.DataFrame(rows)
out = Path(args.out) / "srraw_inventory.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"{df.seq.nunique()} sequence folders, {len(df)} shots; duplicate names: "
      f"{df.drop_duplicates('path').groupby('seq').path.apply(lambda v: v.str.rsplit('/', n=2).str[0].nunique()).gt(1).sum()}")
print(df.reason.replace("", "compared").value_counts().to_string())
print("sequences with >=1 compared shot:", df[df.reason == ""].seq.nunique())
