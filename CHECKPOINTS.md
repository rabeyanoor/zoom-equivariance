# Checkpoints

All weights are the authors' official releases, loaded by `src/models.py` (Hugging Face ids unless noted). The
package versions used for every result are pinned in `requirements-lock.txt` (MoGe at commit
`74fbce0`; UniDepth and Metric3D from their GitHub repositories, cloned into `third_party/`).

| Key in `src/models.py` | Model | Checkpoint |
|---|---|---|
| `dav2-s`, `dav2-b`, `dav2-l` | Depth Anything V2 metric (indoor head for NYUv2 / DIODE-indoor, outdoor head otherwise) | `depth-anything/Depth-Anything-V2-Metric-{Indoor,Outdoor}-{Small,Base,Large}-hf` |
| `zoedepth` | ZoeDepth NYU+KITTI | `Intel/zoedepth-nyu-kitti` |
| `depthpro` | Depth Pro | `apple/DepthPro-hf` |
| `moge2-s`, `moge2-b`, `moge2-l` | MoGe-2 | `Ruicheng/moge-2-{vits,vitb,vitl}-normal` |
| `unidepth2-s`, `unidepth2-b`, `unidepth2-l` | UniDepthV2 | `lpiccinelli/unidepth-v2-{vits14,vitb14,vitl14}` |
| `unik3d-l` | UniK3D | `lpiccinelli/unik3d-vitl` |
| `da3-nested`, `da3-metric-l`, `da3-l` | Depth Anything 3 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `DA3METRIC-LARGE`, `DA3-LARGE-1.1` |
| `vggt` | VGGT | `facebook/VGGT-1B` |
| `metric3d-s`, `metric3d-l` | Metric3D v2 | torch hub entries `metric3d_vit_small`, `metric3d_vit_large` (github.com/YvanYin/Metric3D) |
| `geocalib` | GeoCalib (calibration only) | github.com/cvg/GeoCalib, default weights |
| `*-k` | MoGe-2-S / UniDepthV2-S given the true camera | same checkpoints as above |

## Fine-tuned variants

The fine-tuned checkpoints are not in git (size); they are produced by the training kernels and are fully
determined by the configuration below and the seed.

| Key | What | Script |
|---|---|---|
| `dav2-s-eft`, `dav2-s-eftb`, `dav2-s-eftb-s1`, `dav2-s-eftb-s2` | EFT, EFT+B (seeds 0, 1, 2) | `scripts/train_eft.py` |
| `dav2-s-blur`, `dav2-s-zhong`, `dav2-s-anchor`, `dav2-s-eftb3` | blur-only, stop-gradient target, anchor-only, EFT+B with anchor ×3 | `scripts/train_eft.py` |
| `dav2-s-scale`, `dav2-s-scalenb` | EFT-S (with / without the blur term) | `scripts/eft_scale.py` |
| `depthpro-eftf` | EFT-F, Depth Pro's focal head | `scripts/train_eft_focal.py` |

EFT / EFT+B: AdamW, learning rate 5e-5 (head) and 5e-6 (backbone), batch 4, 3,000 steps, λ = μ = 1, horizontal
flips, input height 392 px, mixed precision, one T4 GPU. Logs of every run are in `outputs/eft/` and `logs/kaggle/`.
