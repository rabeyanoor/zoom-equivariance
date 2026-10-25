"""Build a one-file Kaggle kernel: project code (base64) + job list + kaggle/templates/<kind>_main.py.

    python scripts/kaggle_build.py <kaggle-username> audit [smoke|a|b]   (a, b: two halves, 12 h limit)
    python scripts/kaggle_build.py <kaggle-username> eft [smoke]
    python scripts/kaggle_build.py <kaggle-username> tpu [smoke|main|cam|camgpu|calib]   (TPU VM CPUs in parallel; camgpu: GPU)
    kaggle kernels push -p kaggle/build/<kind>[_<part>]
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user, kind = sys.argv[1], sys.argv[2]
part = sys.argv[3] if len(sys.argv) > 3 else ""
SMOKE = part in ("smoke", "focalsmoke")   # tiny run to catch bugs fast
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
        JOBS = [f"{m} {SETS} --n 50" for m in LADDER[6:]]  # 50: keeps the week within the Kaggle GPU quota
    elif part == "d":
        # confounder controls (optical vs digital zoom, interpolation, JPEG) for one model per group
        JOBS = [f"{m} {SETS} --n 100 --controls-only --out-tag ctrl" for m in ("dav2-s", "depthpro", "moge2-s", "metric3d-s")]
        # ZoomFix with a finer exponent grid (alpha can exceed 1), same 100 images
        JOBS += [f"{m} {SETS} --n 100 --fine-fix --out-tag fix" for m in ("dav2-s", "depthpro", "moge2-s")]
    elif part in ("cues", "cues2", "cues3", "cuestpu", "cuesgpu"):
        # GPU quota spent: split the 2x zoom into context loss and magnification on CPU kernels, or
        # all of it on a TPU VM, whose 96 CPU cores run the same PyTorch code much faster; cuesgpu: the five
        # models still missing, on a GPU of another account (DA3-Metric-L came from cues3)
        if part == "cuesgpu":
            JOBS = [f"{m} {SETS} --n {n} --cues --out-tag cues" for m, n in
                    (("dav2-s", 100), ("zoedepth", 100), ("moge2-s", 100), ("unidepth2-s", 100), ("depthpro", 50))]
        elif part == "cuestpu":
            JOBS = [f"{m} {SETS} --n {n} --cues --out-tag cues" for m, n in
                    (("dav2-s", 100), ("zoedepth", 100), ("moge2-s", 100), ("unidepth2-s", 100),
                     ("depthpro", 50), ("da3-metric-l", 50))]
        else:
            models_, n = {"cues": (("dav2-s", "zoedepth", "moge2-s", "unidepth2-s"), 100),
                          "cues2": (("depthpro",), 50), "cues3": (("da3-metric-l",), 50)}[part]
            JOBS = [f"{m} {SETS} --n {n} --cues --out-tag cues" for m in models_]
    elif part in ("fulla", "fullb"):
        # the main models on the complete test sets (KITTI val selection, NYUv2 test, DIODE val), as a check of
        # the 200-image sample; Depth Pro alone in fullb (it is the slowest model by far)
        models_ = ("depthpro",) if part == "fullb" else ("dav2-s", "zoedepth", "moge2-s", "unidepth2-s", "metric3d-s")
        JOBS = [f"{m} {SETS} --n 100000 --out-tag full" for m in models_]
    elif part == "csmoke":
        # CPU kernel (no GPU slot needed); Metric3D has hard-coded .cuda() calls
        JOBS = [f"{m} kitti --n 1" for m in LADDER if m != "metric3d-l"]
    JOBS = ["scripts/run_audit.py " + j for j in JOBS]
    if part == "c2":
        JOBS.append("scripts/count_params.py")
elif kind == "tpu":
    # one TPU VM kernel, run on its 96 CPU cores with many jobs at once (kaggle/templates/tpu_main.py):
    # the cue split for the four models still missing, EFT-S (cache the frozen model's outputs, fit the
    # head, score it), and the cue split of the fine-tuned models
    res, ck = "--out /kaggle/working/results", "--cache /kaggle/working/eft/cache"
    n_cue, n_dp, n_eval, views, epochs = (2, 1, 2, 32, 5) if SMOKE else (100, 50, 200, 4000, 60)
    sets = ("kitti",) if SMOKE else SETS.split()
    dom = {"kitti": "outdoor", "diode-out": "outdoor", "nyu": "indoor", "diode-in": "indoor"}
    JOBS = []

    def job(id_, cmd, after=()):
        JOBS.append({"id": id_, "cmd": cmd, "after": list(after)})

    if part in ("focal", "focalsmoke"):
        # GPU: EFT-F (Depth Pro's focal head tuned with the focal-consistency loss) and its audit; then real optical
        # zoom on SR-RAW for the main models, the tuned ones and their frozen references
        kitti = "--kitti 'data/raw/kitti_raw/*/*/image_02/data/*.png'"
        job("train-eftf", f"scripts/train_eft_focal.py {kitti} --nyu-train"
            f"{' --n-per-set 4 --views 2 --epochs 2' if SMOKE else ''} --out /kaggle/working/eft")
        for ds in sets:
            job(f"audit-depthpro-eftf-{ds}", f"scripts/run_audit.py depthpro-eftf {ds} --n {3 if SMOKE else 200} {res}",
                ["train-eftf"])
        for m in ("depthpro", "depthpro-eftf", "dav2-s", "dav2-s-fixed", "dav2-s-eftb", "moge2-s", "unidepth2-s",
                  "metric3d-s"):
            job(f"srraw-{m}", f"scripts/run_srraw.py {m} --root data/raw/srraw{' --every 20' if SMOKE else ''} {res}",
                ["train-eftf"] if m == "depthpro-eftf" else [])
    elif part == "figs":
        # CPU: Figure 1 maps of the fine-tuned model (and the frozen one at the same input size), and the maps of one
        # SR-RAW sequence for the optical-zoom figure; the sequence (argv[4]) is the one with DA-V2-S's median alpha
        seq = sys.argv[4]
        job("fig1-eft", f"scripts/scene_stats.py --figure-only --figure-models dav2-s-fixed dav2-s-eftb "
            f"--maps-name figure1_maps_eft.npz {res}")
        job("srraw-check", f"scripts/srraw_check.py --root data/raw/srraw {res}")
        job("srraw-maps", f"scripts/srraw_maps.py --root data/raw/srraw --seq {seq} "
            f"--models dav2-s dav2-s-eftb depthpro depthpro-eftf moge2-s {res}")
    elif part == "calib":
        # GPU: a dedicated calibration network (GeoCalib) under the same zooms, as a reference for beta, and the
        # zoom response by semantic region (objects of familiar size vs surfaces), same 200 images per set
        for ds in sets:
            job(f"audit-geocalib-{ds}", f"scripts/run_audit.py geocalib {ds} --n 200 {res}")
        for m in ("dav2-s", "moge2-s", "depthpro"):
            for ds in sets:
                job(f"regions-{m}-{ds}", f"scripts/run_regions.py {m} {ds} --n 200 {res}")
    elif part in ("cam", "camgpu"):
        # the same camera-predicting networks given the true camera, paired with the main sample, and the
        # UniDepthV2 cue split that failed in tpu-main. camgpu: the same on a GPU kernel (one job at a time,
        # the template's slots follow the 4 CPU cores), with all 200 images per set and Depth Pro's cue split,
        # which was too slow on CPU
        gpu = part == "camgpu"
        for m in ("moge2-s-k", "unidepth2-s-k"):
            for ds in sets:
                job(f"audit-{m}-{ds}", f"scripts/run_audit.py {m} {ds} --n 200 {'' if gpu else '--every 2 '}{res}")
        for m, n in (("unidepth2-s", n_cue),) + ((("depthpro", n_dp),) if gpu else ()):
            for ds in sets:
                job(f"cues-{m}-{ds}", f"scripts/run_audit.py {m} {ds} --n {n} --cues --out-tag cues {res}")
    else:
        # longest first: Depth Pro is the slowest model by far
        for m, n in (("depthpro", n_dp), ("moge2-s", n_cue), ("unidepth2-s", n_cue), ("zoedepth", n_cue)):
            for ds in sets:
                if m == "zoedepth" and ds in ("kitti", "nyu") and not SMOKE:
                    continue  # finished in audit-cuestpu
                job(f"cues-{m}-{ds}", f"scripts/run_audit.py {m} {ds} --n {n} --cues --out-tag cues {res}")
        shards = {"outdoor": 2 if SMOKE else 8, "indoor": 2 if SMOKE else 4}
        src = {"outdoor": "--images 'data/raw/kitti_raw/*/*/image_02/data/*.png'", "indoor": "--nyu-train"}
        for d, k in shards.items():
            for i in range(k):
                job(f"cache-{d}-{i}", f"scripts/eft_scale.py cache {d} {src[d]} --views {views} --shard {i} --shards {k} {ck}")
            for tag, flags in (("scale", "--w-blur 1"), ("scalenb", "")):
                job(f"fit-{tag}-{d}", f"scripts/eft_scale.py fit {d} {ck} {flags} --epochs {epochs} --tag {tag} "
                    f"--out /kaggle/working/eft", [f"cache-{d}-{i}" for i in range(k)])
        for ds in sets:
            for m in ("dav2-s-fixed", "dav2-s-scale", "dav2-s-scalenb"):
                after = [] if m == "dav2-s-fixed" else [f"fit-{m.split('-')[-1]}-{dom[ds]}"]
                job(f"eval-{m}-{ds}", f"scripts/run_audit.py {m} {ds} --n {n_eval} {res}", after)
            for m in ("dav2-s-fixed", "dav2-s-eftb", "dav2-s-scale"):
                after = [f"fit-scale-{dom[ds]}"] if m == "dav2-s-scale" else []
                job(f"cues-{m}-{ds}", f"scripts/run_audit.py {m} {ds} --n {n_cue} --cues --out-tag cues {res}", after)
elif kind == "stats":
    # CPU only: per-image scene statistics + Figure 1 depth maps
    flag = {"fig": "--figure-only", "scenes": "--nyu-scenes"}.get(part, "--figure1")
    JOBS = [f"scripts/scene_stats.py --n {N} {flag}"]
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
    elif part == "ablate":
        # anchor-only turned out to be the frozen model (student == teacher at step 0, so the
        # loss and its gradient are exactly zero); the blur-only run is the real baseline
        runs = ["--target student --tag zhong", "--w-zoom 0 --w-blur 1 --tag blur", "--w-blur 1 --w-anchor 3 --tag eftb3"]
        evals = ("dav2-s-zhong", "dav2-s-blur", "dav2-s-eftb3")
    elif part == "seeds":
        # EFT+B again with two more seeds (seed 0 is the paper run), to measure seed variance
        runs = ["--w-blur 1 --seed 1 --tag eftb-s1", "--w-blur 1 --seed 2 --tag eftb-s2"]
        evals = ("dav2-s-eftb-s1", "dav2-s-eftb-s2")
    elif part in ("scale", "scalesmoke"):
        # EFT-S: frozen DA-V2-S + a learned per-image log-scale; the fixed model is scored again so the
        # comparison is within one kernel
        runs = ["--scale-only --lr-head 1e-3 --w-blur 1 --tag scale", "--scale-only --lr-head 1e-3 --tag scalenb"]
        evals = ("dav2-s-fixed", "dav2-s-scale", "dav2-s-scalenb")
        if part == "scalesmoke":
            steps, N = 20, 3
    else:
        runs, evals = ["--tag eft"], ("dav2-s-fixed", "dav2-s-eft")
    JOBS = []
    for flags in runs:
        JOBS += [f"scripts/train_eft.py outdoor --steps {steps} {kitti} {flags} --out /kaggle/working/eft",
                 f"scripts/train_eft.py indoor --steps {steps} --nyu-train {flags} --out /kaggle/working/eft"]
    JOBS += [f"scripts/run_audit.py {m} {SETS} --n {N} --out {res}" for m in evals]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/run_audit.py", ROOT / "scripts/train_eft.py",
                         ROOT / "scripts/run_regions.py", ROOT / "scripts/run_srraw.py", ROOT / "scripts/srraw_check.py",
                         ROOT / "scripts/train_eft_focal.py", ROOT / "scripts/srraw_maps.py",
                         ROOT / "scripts/eft_scale.py",
                         ROOT / "scripts/scene_stats.py", ROOT / "scripts/count_params.py"])}

name = f"zoom-equivariance-{kind}{'-' + part if part and kind != 'stats' else ''}"
out = ROOT / "kaggle" / "build" / f"{kind}{'_' + part if part else ''}"
out.mkdir(parents=True, exist_ok=True)
body = (ROOT / "kaggle" / "templates" / f"{kind if kind in ('eft', 'tpu') else 'audit'}_main.py").read_text()
(out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\nJOBS = {json.dumps(JOBS)}\n" + body)
GPU_TPU_PARTS = ("camgpu", "calib", "focal", "focalsmoke")  # "tpu" kernels that run on a GPU instead
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}",
    "title": name,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": (kind not in ("stats", "tpu") or part in GPU_TPU_PARTS)
    and part not in ("csmoke", "cues", "cues2", "cues3", "cuestpu"),
    **({"enable_tpu": True, "machine_shape": "TpuV5E8"}
       if part == "cuestpu" or (kind == "tpu" and part not in GPU_TPU_PARTS) else {}),
    "enable_internet": True,
    **({"kernel_sources": [f"{user}/zoom-equivariance-eft"]} if part == "baseline" else {}),
    **({"kernel_sources": [f"{user}/zoom-equivariance-eft-baseline"]
                          + ([f"{user}/zoom-equivariance-tpu-focal"] if part == "figs" else [])}
       if kind == "tpu" and part not in ("cam", "camgpu", "calib") else {}),
}, indent=2))
print("built", out, len(files), "files,", len(JOBS), "jobs")
