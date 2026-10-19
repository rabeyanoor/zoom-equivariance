# Body of the Kaggle audit kernel. scripts/kaggle_build.py puts the project's src/ and
# scripts/ in front of this (as base64), so the kernel is one self-contained file.
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = Path("/kaggle/temp/ze") if Path("/kaggle/temp").exists() else Path("/tmp/ze")
RAW = PROJ / "data" / "raw"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def unpack_project():
    import base64
    for rel, b64 in FILES.items():  # noqa: F821  (filled in by the builder)
        p = PROJ / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(b64))


def install():
    sh("pip install -q transformers==5.20.1 mmengine einops timm h5py termcolor tabulate trimesh wandb")
    sh("pip install -q git+https://github.com/microsoft/MoGe.git")
    tp = PROJ / "third_party"
    tp.mkdir(exist_ok=True)
    if not (tp / "UniDepth").exists():
        sh(f"git clone -q --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git {tp}/UniDepth")
        sh(f"pip install -q --no-deps -e {tp}/UniDepth")
    if not (tp / "Metric3D").exists():
        sh(f"git clone -q --depth 1 https://github.com/YvanYin/Metric3D.git {tp}/Metric3D")


def download():
    (RAW / "kitti").mkdir(parents=True, exist_ok=True)
    (RAW / "nyu").mkdir(parents=True, exist_ok=True)
    (RAW / "diode").mkdir(parents=True, exist_ok=True)
    jobs = [
        ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip", RAW / "kitti/sel.zip"),
        ("https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat", RAW / "nyu/splits.mat"),
        ("https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", RAW / "nyu/nyu_depth_v2_labeled.mat"),
        ("http://diode-dataset.s3.amazonaws.com/val.tar.gz", RAW / "diode/val.tar.gz"),
    ]
    procs = [subprocess.Popen(f"curl -sL --retry 5 -o {dst} {url}", shell=True) for url, dst in jobs if not dst.exists()]
    for p in procs:
        p.wait()
    # keep only what the loaders read
    sh(f"cd {RAW}/kitti && unzip -q -o sel.zip 'depth_selection/val_selection_cropped/*' && rm sel.zip")
    sh(f"cd {RAW}/diode && tar xzf val.tar.gz && rm val.tar.gz")


def main():
    t0 = time.time()
    unpack_project()
    install()
    download()
    print(f"setup done in {(time.time() - t0) / 60:.1f} min", flush=True)
    os.chdir(PROJ)
    out = WORK / "results"
    run = f"{sys.executable} scripts/run_audit.py"
    for job in JOBS:  # noqa: F821  (filled in by the builder)
        try:
            sh(f"{run} {job} --out {out}")
        except subprocess.CalledProcessError as e:
            # one broken model should not cost us the others
            print("FAILED:", job, e, flush=True)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


main()
