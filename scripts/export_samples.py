"""Write the image ids of every evaluation sample to outputs/samples/ and check that the unlabeled KITTI training
drives of EFT are disjoint from the evaluated ones.

    python scripts/export_samples.py

main_<set>.txt: the fixed 200-image sample per set (seed 0) that every main model saw; offcentre_<set>.txt: the
50-image subsets (KITTI, NYUv2) zoomed about the quadrant centres.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis as A  # noqa: E402

OUT = ROOT / "outputs" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATE = (ROOT / "kaggle" / "templates" / "eft_main.py").read_text()
TRAIN_DRIVES = set(re.findall(r'"(20\d\d_\d\d_\d\d_drive_\d{4})"', TEMPLATE.split("TRAIN_DRIVES =", 1)[1].split("]", 1)[0]))

for ds in A.DATASETS:
    names = sorted(pd.read_csv(A.RESULTS / f"dav2-s__{ds}.csv").name.unique())
    (OUT / f"main_{ds}.txt").write_text("\n".join(names) + "\n")
    f = A.RESULTS / f"dav2-s__{ds}__offc.csv"
    off = sorted(pd.read_csv(f).name.unique()) if f.exists() else []
    if off:
        (OUT / f"offcentre_{ds}.txt").write_text("\n".join(off) + "\n")
    print(f"{ds}: {len(names)} images, {len(off)} off-centre")

kitti = [n for ds in A.DATASETS if ds == "kitti" for n in (OUT / f"main_{ds}.txt").read_text().split()]
evaluated = {re.match(r"kitti/(20\d\d_\d\d_\d\d_drive_\d{4})", n).group(1) for n in kitti}
clash = evaluated & TRAIN_DRIVES
assert len(TRAIN_DRIVES) == 7 and not clash, f"training drives overlap the evaluated ones: {clash}"
print(f"KITTI: {len(evaluated)} evaluated drives, {len(TRAIN_DRIVES)} training drives, disjoint")
