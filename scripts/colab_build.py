"""Wrap a built Kaggle kernel as a Colab notebook, for runs when the Kaggle GPU quota is spent.

    python scripts/kaggle_build.py <kaggle-username> audit cuestpu
    python scripts/colab_build.py kaggle/build/audit_cuestpu
Upload the .ipynb to Colab, pick a T4 runtime and run all; the last cell zips /content/out/results.
"""
import json
import sys
from pathlib import Path

build = Path(sys.argv[1])
code = (build / "kernel.py").read_text()
cells = [
    "!nvidia-smi -L",
    code,
    "!cd /content/out && zip -qr /content/results.zip results && ls -l /content/results.zip",
]
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
}
out = build / f"{build.name}_colab.ipynb"
out.write_text(json.dumps(nb))
print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
