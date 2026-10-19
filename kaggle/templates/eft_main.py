# Body of the Kaggle EFT kernel: train on unlabeled images, then audit frozen vs EFT.
# scripts/kaggle_build.py puts FILES and JOBS in front of this.
import base64
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = Path("/kaggle/temp/ze") if Path("/kaggle/temp").exists() else Path("/tmp/ze")
RAW = PROJ / "data" / "raw"
S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

# KITTI raw drives used as *unlabeled* training images. None of them is one of the 13
# drives the KITTI depth val selection is drawn from; main() checks this again.
TRAIN_DRIVES = ["2011_09_26_drive_0001", "2011_09_26_drive_0009", "2011_09_26_drive_0011",
                "2011_09_26_drive_0018", "2011_09_26_drive_0022", "2011_09_26_drive_0057",
                "2011_09_30_drive_0020"]


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def unpack_project():
    for rel, b64 in FILES.items():  # noqa: F821
        p = PROJ / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(b64))


def download():
    for d in ("kitti", "kitti_raw", "nyu", "diode"):
        (RAW / d).mkdir(parents=True, exist_ok=True)
    jobs = [(f"{S3}/data_depth_selection.zip", RAW / "kitti/sel.zip"),
            ("https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat", RAW / "nyu/splits.mat"),
            ("https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", RAW / "nyu/nyu_depth_v2_labeled.mat"),
            ("http://diode-dataset.s3.amazonaws.com/val.tar.gz", RAW / "diode/val.tar.gz")]
    jobs += [(f"{S3}/raw_data/{d}/{d}_sync.zip", RAW / f"kitti_raw/{d}.zip") for d in TRAIN_DRIVES]
    procs = [subprocess.Popen(f"curl -sL --retry 5 -o {dst} {url}", shell=True) for url, dst in jobs]
    for p in procs:
        p.wait()
    sh(f"cd {RAW}/kitti && unzip -q -o sel.zip 'depth_selection/val_selection_cropped/*' && rm sel.zip")
    sh(f"cd {RAW}/diode && tar xzf val.tar.gz && rm val.tar.gz")
    for d in TRAIN_DRIVES:
        # left colour camera only, nothing with depth
        sh(f"cd {RAW}/kitti_raw && unzip -q -o {d}.zip '*/image_02/data/*' && rm {d}.zip")


def check_no_overlap():
    val = {p.name.split("_sync")[0] for p in (RAW / "kitti/depth_selection/val_selection_cropped/image").glob("*.png")}
    clash = val & set(TRAIN_DRIVES)
    print("val drives:", sorted(val), flush=True)
    assert not clash, f"training drives overlap the evaluation drives: {clash}"


def reuse_checkpoints():
    """If an earlier EFT kernel is attached as input, reuse its checkpoints instead of training."""
    import shutil

    (WORK / "eft").mkdir(parents=True, exist_ok=True)
    for ck in Path("/kaggle/input").glob("**/dav2-s-*.pt"):
        print("reusing", ck, flush=True)
        shutil.copy(ck, WORK / "eft" / ck.name)
    for log in Path("/kaggle/input").glob("**/train_log_*.csv"):
        shutil.copy(log, WORK / "eft" / log.name)


def main():
    t0 = time.time()
    unpack_project()
    reuse_checkpoints()
    sh("pip install -q transformers==5.20.1 h5py")
    download()
    check_no_overlap()
    print(f"setup done in {(time.time() - t0) / 60:.1f} min", flush=True)
    py = sys.executable
    for job in JOBS:  # noqa: F821
        try:
            sh(f"cd {PROJ} && EFT_DIR={WORK}/eft {py} {job}")
        except subprocess.CalledProcessError as e:
            print("FAILED:", job, e, flush=True)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


main()
