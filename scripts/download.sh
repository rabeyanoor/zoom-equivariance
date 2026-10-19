#!/usr/bin/env bash
# Downloads the three evaluation sets from their official hosts. Resumable (-C -).
set -euo pipefail
RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
get() { curl -L --retry 5 -C - -o "$2" "$1"; }
get https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip "$RAW/kitti/data_depth_selection.zip"
get https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat "$RAW/nyu/splits.mat"
get https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat "$RAW/nyu/nyu_depth_v2_labeled.mat"
get http://diode-dataset.s3.amazonaws.com/val.tar.gz "$RAW/diode/val.tar.gz"
echo DONE
