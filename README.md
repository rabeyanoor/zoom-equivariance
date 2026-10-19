# Zoom equivariance of metric depth models

**Working title:** *Zoom In, Scale Out: Metric Depth Models Do Not Respect a Digital Zoom*.
Target venue is CVPR 2027 (abstract ~7-10 Nov 2026, paper ~12-16 Nov 2026).

A crop followed by a resize is an exact digital zoom: the focal length is multiplied by s and the depth stays the same. We use that law in three ways:

1. **Label-free audit.** We measure how far 7 metric depth models are from the law with two exponents. α is the depth exponent (ideal 0). β is the focal exponent (ideal 1).
2. **Theory.** We derive the α+β=1 lateral invariant and the zoom–squash ambiguity: a zoomed photo looks the same as a photo of the scene squashed along the depth axis.
3. **Fine-tuning on unlabeled images.**
   - EFT is self-consistency in the spirit of Zhong et al., ICCV 2023.
   - EFT+B adds a blur-invariance term. It closes the "blurry means zoomed" shortcut we found.

`docs/research_overview_bn.md` explains the whole project in Bangla.

## Folder layout

```
zoom-equivariance/
├── README.md
├── docs/                   research overview (Bangla), prior-work notes
├── src/                    library code (imported by scripts and notebooks)
│   ├── zoom.py             exact crop/resize geometry, zoom_K, blur control
│   ├── models.py           one predict(img, K) interface for 7 models + fine-tuned variants
│   ├── datasets.py         KITTI val selection, NYUv2 test (654), DIODE val indoor/outdoor
│   ├── audit.py            per image x zoom: self-consistency, focal, errors vs GT
│   ├── eft.py              EFT / EFT+B / anchor-only trainer
│   └── analysis.py         alpha/beta fits, bootstrap CIs, summaries
├── scripts/                command-line jobs
│   ├── run_audit.py        the audit (heavy: Kaggle GPU)
│   ├── train_eft.py        fine-tuning (Kaggle GPU)
│   ├── scene_stats.py      GT depth spread per image + Figure 1 maps (Kaggle CPU)
│   ├── kaggle_build.py     packs src/ + scripts into one Kaggle kernel
│   ├── kaggle_queue.sh     pushes waiting kernels when a GPU slot is free (max 2)
│   ├── fetch_kernel.sh     downloads a finished kernel's output and keeps its log
│   ├── save_logs.py        Kaggle JSON logs -> readable text in logs/
│   ├── make_paper_numbers.py  every number/table in the paper, from the CSVs
│   └── download.sh         local dataset download (not needed when using Kaggle)
├── notebooks/              analysis, run one at a time (jupytext .py + executed .ipynb)
│   ├── 01_zoom_geometry_check   our zoom code is exact (no model, no data)
│   ├── 02_audit_results         Table 1, alpha/beta curves
│   ├── 03_label_free_proxy      does the label-free score predict real error?
│   ├── 04_off_centre_and_stale_K
│   ├── 05_eft_results           frozen vs anchor-only vs EFT vs EFT+B
│   └── 06_figure1               teaser figure
├── kaggle/
│   ├── templates/          kernel bodies (audit_main.py, eft_main.py)
│   └── build/              generated kernels, one folder each (not in git)
├── outputs/
│   ├── results/            final CSVs the notebooks read
│   ├── eft/                fine-tuned checkpoints (.pt not in git) + training logs
│   ├── figures/  tables/   produced by the notebooks
│   └── kaggle/             raw downloads per kernel (not in git)
├── logs/
│   ├── kaggle/             one text log per Kaggle kernel run (kept, never overwritten)
│   └── local/              push queue and download logs
├── paper/                  CVPR LaTeX: main.tex, sec/, gen/ (generated numbers), main.bib
└── third_party/            UniDepth, Metric3D repos (not in git)
```

## Kaggle kernels

Each run's log is kept in `logs/kaggle/<kernel>.txt`.

| Kernel | What |
|---|---|
| `zoom-equivariance-audit-a` | 6 models x 4 sets x 200 images |
| `zoom-equivariance-audit-a2` | rerun of audit-a with the same-pixel reference column |
| `zoom-equivariance-audit-b` | Depth Pro + off-centre zooms |
| `zoom-equivariance-eft` | EFT training + frozen vs EFT |
| `zoom-equivariance-eft-baseline` | anchor-only and EFT+B training, scores all four |
| `zoom-equivariance-stats` | GT depth spread + Figure 1 maps (CPU) |

## Workflow

```bash
# build and push (the queue pushes automatically when a slot frees)
.venv/bin/python scripts/kaggle_build.py rabeya100x audit a2
kaggle kernels push -p kaggle/build/audit_a2

# when a kernel is done: download its output, keep the log
scripts/fetch_kernel.sh audit-a2
cp outputs/kaggle/audit-a2/results/*.csv outputs/results/

# analysis, one notebook at a time
cd notebooks
../.venv/bin/jupytext --to ipynb 02_audit_results.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=zoom-eq 02_audit_results.ipynb

# paper numbers and tables
.venv/bin/python scripts/make_paper_numbers.py     # -> paper/gen/
```

The paper compiles on Overleaf: upload the `paper/` folder and set `main.tex` as the main file.

Credentials: the Kaggle CLI reads `~/.kaggle/kaggle.json`. No keys are stored in this repo or in the logs.
