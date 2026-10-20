"""Wrap a built Kaggle kernel as a Colab notebook, for runs when the Kaggle GPU quota is spent.

    python scripts/kaggle_build.py <kaggle-username> audit cuestpu
    python scripts/colab_build.py kaggle/build/audit_cuestpu [--skip da3-metric-l]
Upload the .ipynb to Colab, pick a T4 runtime and run all. The results go to Google Drive
(MyDrive/zoom-equivariance/results) as each model finishes, so a dropped runtime loses at most one model.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("build", type=Path)
p.add_argument("--skip", nargs="*", default=[], help="models whose results we already have")
args = p.parse_args()

code = (args.build / "kernel.py").read_text()
head, rest = code.split("\n", 2)[:2], code.split("\n", 2)[2]
files_line, jobs_line = head
jobs = [j for j in json.loads(jobs_line.removeprefix("JOBS = ")) if j.split()[1] not in args.skip]
work = 'WORK = Path("/content/out") if Path("/content").exists() else Path("/kaggle/working")  # Colab also has /kaggle'
assert work in rest
rest = rest.replace(work, 'WORK = Path("/content/drive/MyDrive/zoom-equivariance")')
code = "\n".join([files_line, "JOBS = " + json.dumps(jobs), rest])
cells = [
    "!nvidia-smi -L\nfrom google.colab import drive\ndrive.mount('/content/drive')",
    code,
    "!ls -l /content/drive/MyDrive/zoom-equivariance/results",
]
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
}
out = args.build / f"{args.build.name}_colab.ipynb"
out.write_text(json.dumps(nb))
print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB,", len(jobs), "jobs")
