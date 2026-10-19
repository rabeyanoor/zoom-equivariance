"""Build a one-file Kaggle kernel: project code (base64) + job list + kaggle/templates/<kind>_main.py.

    python scripts/kaggle_build.py <kaggle-username> audit [smoke|a|b]   (a, b: two halves, 12 h limit)
    python scripts/kaggle_build.py <kaggle-username> eft [smoke]
    kaggle kernels push -p kaggle/build/<kind>[_<part>]
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user, kind = sys.argv[1], sys.argv[2]
part = sys.argv[3] if len(sys.argv) > 3 else ""
SMOKE = part == "smoke"   # tiny run to catch bugs fast
SETS = "kitti nyu diode-in diode-out"
N = 3 if SMOKE else 200

if kind == "audit":
    JOBS = [f"{m} {SETS} --n {N}" for m in
            ("dav2-s", "dav2-b", "zoedepth", "moge2-s", "unidepth2-s", "metric3d-s", "depthpro")]
    JOBS += [f"metric3d-s {SETS} --n {N} --stale-k"]
    # off-centre zooms (principal point), smaller sample
    JOBS += [f"{m} kitti nyu --n {2 if SMOKE else 50} --off-centre --out-tag offc"
             for m in ("moge2-s", "unidepth2-s", "depthpro", "dav2-s")]
    # Depth Pro is ~5x slower than the rest, so part b = Depth Pro + off-centre
    if part == "a":
        JOBS = [j for j in JOBS if not j.startswith("depthpro") and "--off-centre" not in j]
    elif part == "a2":
        # final run with the current audit code (same-pixel reference, principal point,
        # training-free correction): all models, plus off-centre zooms for the two models
        # that predict a principal point
        JOBS = [j for j in JOBS if "--off-centre" not in j]
        JOBS += [f"{m} kitti nyu --n 50 --off-centre --out-tag offc" for m in ("moge2-s", "unidepth2-s")]
    elif part == "b":
        JOBS = [j for j in JOBS if j.startswith("depthpro") or "--off-centre" in j]
    # model ladder: larger variants and newer families, 100 images per set
    LADDER = ("dav2-l", "moge2-b", "moge2-l", "unidepth2-b", "unidepth2-l", "unik3d-l",
              "da3-nested", "da3-metric-l", "metric3d-l", "da3-l", "vggt")
    if part == "c1":
        JOBS = [f"{m} {SETS} --n 100" for m in LADDER[:6]]
    elif part == "c2":
        JOBS = [f"{m} {SETS} --n 100" for m in LADDER[6:]]
    elif part == "csmoke":
        # CPU kernel (no GPU slot needed); Metric3D has hard-coded .cuda() calls
        JOBS = [f"{m} kitti --n 1" for m in LADDER if m != "metric3d-l"]
    JOBS = ["scripts/run_audit.py " + j for j in JOBS]
    if part == "c2":
        JOBS.append("scripts/count_params.py")
elif kind == "stats":
    # CPU only: per-image scene statistics + Figure 1 depth maps
    JOBS = [f"scripts/scene_stats.py --n {N} " + ("--figure-only" if part == "fig" else "--figure1")]
else:
    # eft: the method; eft_baseline (part "baseline"): anchor-only fine-tuning, same data and steps
    steps = 20 if SMOKE else 3000
    res = "/kaggle/working/results"
    kitti = "--images 'data/raw/kitti_raw/*/*/image_02/data/*.png'"
    if part == "baseline":
        # trains anchor-only and EFT+B, then scores all four with the current audit code
        # (the EFT checkpoints come from the attached zoom-equivariance-eft output)
        runs = ["--w-zoom 0 --tag anchor", "--w-blur 1 --tag eftb"]
        evals = ("dav2-s-fixed", "dav2-s-anchor", "dav2-s-eft", "dav2-s-eftb")
    else:
        runs, evals = ["--tag eft"], ("dav2-s-fixed", "dav2-s-eft")
    JOBS = []
    for flags in runs:
        JOBS += [f"scripts/train_eft.py outdoor --steps {steps} {kitti} {flags} --out /kaggle/working/eft",
                 f"scripts/train_eft.py indoor --steps {steps} --nyu-train {flags} --out /kaggle/working/eft"]
    JOBS += [f"scripts/run_audit.py {m} {SETS} --n {N} --out {res}" for m in evals]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/run_audit.py", ROOT / "scripts/train_eft.py",
                         ROOT / "scripts/scene_stats.py", ROOT / "scripts/count_params.py"])}

name = f"zoom-equivariance-{kind}{'-' + part if part and kind != 'stats' else ''}"
out = ROOT / "kaggle" / "build" / f"{kind}{'_' + part if part else ''}"
out.mkdir(parents=True, exist_ok=True)
body = (ROOT / "kaggle" / "templates" / f"{'eft' if kind == 'eft' else 'audit'}_main.py").read_text()
(out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\nJOBS = {json.dumps(JOBS)}\n" + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}",
    "title": name,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": kind != "stats" and part != "csmoke",
    "enable_internet": True,
    **({"kernel_sources": [f"{user}/zoom-equivariance-eft"]} if part == "baseline" else {}),
}, indent=2))
print("built", out, len(files), "files,", len(JOBS), "jobs")
