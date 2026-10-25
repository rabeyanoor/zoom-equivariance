# Body of the TPU VM kernel. The TPU itself is not used: the VM's CPU cores run the same PyTorch
# code, many jobs at once. scripts/kaggle_build.py puts FILES and JOBS in front of this; each job is
# {"id", "cmd", "after": [ids]} and starts once the jobs it waits for have ended.
import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = Path("/kaggle/temp/ze") if Path("/kaggle/temp").exists() else Path("/tmp/ze")
RAW = PROJ / "data" / "raw"
S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
# 6 threads per job when there are more jobs than cores allow, up to 16 when there are few (beyond that a
# single PyTorch CPU job barely speeds up); TpuV5E8 has 224 cores
THREADS = max(6, min(16, (os.cpu_count() or 96) // max(1, len(JOBS))))  # noqa: F821
SLOTS = max(1, (os.cpu_count() or 96) // THREADS)

# unlabeled training drives, disjoint from the KITTI val selection (as in eft_main.py)
TRAIN_DRIVES = ["2011_09_26_drive_0001", "2011_09_26_drive_0009", "2011_09_26_drive_0011",
                "2011_09_26_drive_0018", "2011_09_26_drive_0022", "2011_09_26_drive_0057",
                "2011_09_30_drive_0020"]


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def unpack_project():
    for rel, b64 in FILES.items():  # noqa: F821  (filled in by the builder)
        p = PROJ / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(b64))


def install():
    sh("pip install -q transformers==5.20.1 einops timm h5py wandb")  # UniDepth imports wandb at module level
    sh("pip install -q git+https://github.com/microsoft/MoGe.git")
    sh("pip install -q git+https://github.com/cvg/GeoCalib#egg=geocalib")  # calibration-only reference
    tp = PROJ / "third_party"
    tp.mkdir(exist_ok=True)
    if not (tp / "UniDepth").exists():
        sh(f"git clone -q --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git {tp}/UniDepth")
    sh(f"pip install -q --no-deps -e {tp}/UniDepth")
    if any("metric3d" in j["cmd"] for j in JOBS):  # noqa: F821
        sh("pip install -q mmengine termcolor tabulate trimesh")
        if not (tp / "Metric3D").exists():
            sh(f"git clone -q --depth 1 https://github.com/YvanYin/Metric3D.git {tp}/Metric3D")


def download():
    # an interactive session can run the smoke kernel and then the main one: download only once
    if (RAW / "done").exists():
        return
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
        sh(f"cd {RAW}/kitti_raw && unzip -q -o {d}.zip '*/image_02/data/*' && rm {d}.zip")
    val = {p.name.split("_sync")[0] for p in (RAW / "kitti/depth_selection/val_selection_cropped/image").glob("*.png")}
    assert not val & set(TRAIN_DRIVES), f"training drives overlap the evaluation drives: {val & set(TRAIN_DRIVES)}"
    (RAW / "done").touch()


def download_srraw():
    """SR-RAW test set (Zhang et al., CVPR 2019; CC BY-NC 4.0): zoom-test.zip in the authors' Google Drive folder linked from
    github.com/ceciliavision/zoom-learn-zoom; only the camera JPGs are kept (the ARW raw files are not used)."""
    if not any("run_srraw" in j["cmd"] for j in JOBS) or (RAW / "srraw" / "done").exists():  # noqa: F821
        return
    (RAW / "srraw").mkdir(parents=True, exist_ok=True)
    sh("pip install -q gdown")
    sh(f"cd {RAW}/srraw && gdown -q 11WbP15GQjCHbbsjs3bnJWkK4c-EWkbSO -O test.zip && "
       f"unzip -q -o test.zip '*.JPG' && rm test.zip")
    n = len(list((RAW / "srraw").rglob("00001.JPG")))
    print(f"SR-RAW: {n} sequences", flush=True)
    assert n, "SR-RAW download gave no sequences"
    (RAW / "srraw" / "done").touch()


def reuse_checkpoints():
    """Checkpoints of earlier EFT kernels attached as input (EFT+B, for the cue split)."""
    (WORK / "eft").mkdir(parents=True, exist_ok=True)
    for ck in [*Path("/kaggle/input").glob("**/dav2-s-*.pt"), *Path("/kaggle/input").glob("**/depthpro-*.pt")]:
        print("reusing", ck, flush=True)
        shutil.copy(ck, WORK / "eft" / ck.name)
    # CSVs of an earlier TPU kernel: run_audit.py skips the images they already hold
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    for f in Path("/kaggle/input").glob("**/results/*.csv"):
        shutil.copy(f, WORK / "results" / f.name)


def run_all(jobs):
    env = dict(os.environ, ZE_THREADS=str(THREADS), OMP_NUM_THREADS=str(THREADS), EFT_DIR=str(WORK / "eft"))
    log_dir = WORK / "job_logs"
    log_dir.mkdir(exist_ok=True)
    waiting, running, ended, failed = list(jobs), {}, set(), []
    t0 = time.time()
    while waiting or running:
        for jid, (p, f) in list(running.items()):
            if p.poll() is not None:
                f.close()
                ended.add(jid)
                del running[jid]
                status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
                if p.returncode:
                    failed.append(jid)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {jid}: {status}", flush=True)
        for j in list(waiting):
            if len(running) >= SLOTS:
                break
            if all(a in ended for a in j["after"]):
                waiting.remove(j)
                f = open(log_dir / f"{j['id']}.log", "w")
                print(f"[{(time.time() - t0) / 60:6.1f} min] start {j['id']}: {j['cmd']}", flush=True)
                running[j["id"]] = (subprocess.Popen(f"cd {PROJ} && {sys.executable} {j['cmd']}", shell=True, env=env,
                                                     stdout=f, stderr=subprocess.STDOUT), f)
        time.sleep(20)
    print("failed jobs:", failed or "none", flush=True)
    for jid in failed:
        print(f"--- tail of {jid}", flush=True)
        print("".join((log_dir / f"{jid}.log").read_text().splitlines(True)[-30:]), flush=True)


def main():
    t0 = time.time()
    unpack_project()
    reuse_checkpoints()
    install()
    download()
    download_srraw()
    print(f"setup done in {(time.time() - t0) / 60:.1f} min, {os.cpu_count()} cores", flush=True)
    run_all(JOBS)  # noqa: F821  (filled in by the builder)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


main()
