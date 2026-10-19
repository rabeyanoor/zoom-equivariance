# Zoom equivariance of metric depth models

Working title: **Zoom In, Scale Out: Testing and Enforcing Camera Equivariance in Metric Depth Models**
(target: CVPR 2027, abstract ~7-10 Nov 2026, paper ~12-16 Nov 2026)

A crop + resize is an exact digital zoom: focal length x s, depth unchanged. We use that law to
1. audit metric depth models without labels (alpha: depth exponent, beta: focal exponent),
2. test whether the label-free score tracks real error,
3. fix pixels-only models with Equivariance Fine-Tuning (EFT) on unlabeled images.

## Layout

| Path | What |
|---|---|
| `src/zoom.py` | exact crop/resize geometry, blur control |
| `src/models.py` | one `predict(img, K)` interface for 7 models + frozen/EFT |
| `src/datasets.py` | KITTI val selection, NYUv2 test (654), DIODE val (indoor/outdoor) |
| `src/audit.py` | per image, per zoom: self-consistency, focal, errors vs GT |
| `src/eft.py` | EFT trainer (teacher on the unzoomed image -> student on the zoom) |
| `src/analysis.py` | alpha / beta fits, bootstrap CIs |
| `scripts/run_audit.py`, `scripts/train_eft.py` | heavy jobs (Kaggle GPU) |
| `scripts/kaggle_build.py` | packs the code into one Kaggle kernel |
| `notebooks/01..05` | analysis, run one at a time |

## Running

The laptop has no GPU and little free RAM, so heavy jobs run on Kaggle:

```bash
python scripts/kaggle_build.py <kaggle-user> audit && kaggle kernels push -p kaggle/build_audit
python scripts/kaggle_build.py <kaggle-user> eft   && kaggle kernels push -p kaggle/build_eft
kaggle kernels output <kaggle-user>/zoom-equivariance-audit -p outputs/kaggle_audit
```

Then copy `results/*.csv` into `outputs/results/` (and `eft/` into `outputs/eft/`) and run the
notebooks in order:

```bash
cd notebooks
jupytext --to ipynb 02_audit_results.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=zoom-eq 02_audit_results.ipynb
```

Credentials: the Kaggle CLI reads `~/.kaggle/kaggle.json`; no keys are stored in this repo.
