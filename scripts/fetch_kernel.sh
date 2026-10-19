#!/usr/bin/env bash
# Download a finished Kaggle kernel's output and keep its log.
#   scripts/fetch_kernel.sh audit-a      -> outputs/kaggle/audit-a/ + logs/kaggle/
set -e
cd "$(dirname "$0")/.."
k="$1"
dst="outputs/kaggle/$k"
mkdir -p "$dst"
kaggle kernels output "rabeya100x/zoom-equivariance-$k" -p "$dst"
.venv/bin/python scripts/save_logs.py
