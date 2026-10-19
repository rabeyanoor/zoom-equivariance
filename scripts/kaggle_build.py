"""Build a one-file Kaggle kernel: project code (base64) + job list + kaggle/<kind>_main.py.

    python scripts/kaggle_build.py <kaggle-username> audit [smoke]
    python scripts/kaggle_build.py <kaggle-username> eft [smoke]
    kaggle kernels push -p kaggle/build_<kind>[_smoke]
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user, kind = sys.argv[1], sys.argv[2]
SMOKE = len(sys.argv) > 3 and sys.argv[3] == "smoke"   # tiny run to catch bugs fast
SETS = "kitti nyu diode-in diode-out"
N = 3 if SMOKE else 200

if kind == "audit":
    JOBS = [f"{m} {SETS} --n {N}" for m in
            ("dav2-s", "dav2-b", "zoedepth", "moge2-s", "unidepth2-s", "metric3d-s", "depthpro")]
    JOBS += [f"metric3d-s {SETS} --n {N} --stale-k"]
    # off-centre zooms (principal point), smaller sample
    JOBS += [f"{m} kitti nyu --n {2 if SMOKE else 50} --off-centre --out-tag offc"
             for m in ("moge2-s", "unidepth2-s", "depthpro", "dav2-s")]
else:
    steps = 20 if SMOKE else 3000
    res = "/kaggle/working/results"
    JOBS = [f"scripts/train_eft.py outdoor --steps {steps} --images 'data/raw/kitti_raw/*/*/image_02/data/*.png' --out /kaggle/working/eft",
            f"scripts/train_eft.py indoor --steps {steps} --nyu-train --out /kaggle/working/eft"]
    JOBS += [f"scripts/run_audit.py {m} {SETS} --n {N} --out {res}" for m in ("dav2-s-fixed", "dav2-s-eft")]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/run_audit.py", ROOT / "scripts/train_eft.py"])}

name = f"zoom-equivariance-{kind}{'-smoke' if SMOKE else ''}"
out = ROOT / "kaggle" / f"build_{kind}{'_smoke' if SMOKE else ''}"
out.mkdir(parents=True, exist_ok=True)
body = (ROOT / "kaggle" / f"{kind}_main.py").read_text()
(out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\nJOBS = {json.dumps(JOBS)}\n" + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}",
    "title": name,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
}, indent=2))
print("built", out, len(files), "files,", len(JOBS), "jobs")
