# Data

No dataset is redistributed here. `scripts/download.sh` (and each Kaggle kernel) downloads the official files from
the original sources; check each source's terms before use.

| Set | Source | What is used | Ground truth | Terms |
|---|---|---|---|---|
| KITTI depth selection, val | cvlibs.net/datasets/kitti (`data_depth_selection.zip`) | `val_selection_cropped`: 1,000 images from 13 drives, with intrinsics | projected LiDAR, capped at 80 m | CC BY-NC-SA 3.0 |
| KITTI raw (training only) | cvlibs.net/datasets/kitti, raw data | 7 drives disjoint from the 13 val drives (`TRAIN_DRIVES` in `kaggle/templates/eft_main.py`), 3,323 frames, images only (no labels) | none used | CC BY-NC-SA 3.0 |
| NYU Depth v2 | cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html (`nyu_depth_v2_labeled.mat`, `splits.mat`) | the 654 official test images (evaluation, Eigen crop); the RGB of the 795 official training images (EFT, no labels) | Kinect, capped at 10 m | see the dataset site |
| DIODE val | diode-dataset.org (`val.tar.gz`) | indoor (325) and outdoor (446) scans | laser, capped at 50 m / 150 m | see the dataset site |
| SR-RAW test | github.com/ceciliavision/zoom-learn-zoom | the 50 test sequences (static, tripod, Sony FE 24–240 mm); the `aligned/` sub-folders are not used | none (label-free exponents only) | CC BY-NC 4.0 |

## Samples

Every main model sees the same images. The fixed samples (200 per set, seed 0) are listed one id per line in
`outputs/samples/main_<set>.txt`, and the 50-image off-centre subsets (KITTI, NYUv2) in
`outputs/samples/offcentre_<set>.txt`. `python scripts/export_samples.py` rewrites both from the result files and
checks that the KITTI training drives are disjoint from the evaluated ones. `scripts/verify_results.py` checks that
all models of one sample saw the same images and that every zoom row is complete.

Image ids are `<set>/<file stem>`: KITTI and DIODE use the official file names, NYUv2 the 0-based index into the
labelled `.mat` file (`outputs/results/nyu_scenes.csv` maps it to its scene, for the scene-level bootstrap).
