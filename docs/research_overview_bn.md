# Research overview (Bangla)

**Paper:** *Zoom In, Scale Out: Metric Depth Models Do Not Respect a Digital Zoom*
**Target:** CVPR 2027 (abstract ~7-10 Nov 2026, paper ~12-16 Nov 2026)
**Shesh update:** 2026-10-19

> Notun result ashle ei file update kora hoy. Number gulo ekhane shudhu bojhar jonno.
> Paper er shob number `scripts/make_paper_numbers.py` CSV theke toiri kore.

## 1. Shomossha

Metric depth model gulo (Depth Anything, Depth Pro, MoGe, UniDepth, Metric3D) ekta chhobi dekhe
bole dey jinish gulo koto **meter** dure. Kintu ek chhobi dekhe meter e dure bolte hole camera er
focal length jana lage. Model gulo ki asholei camera bujhe, naki shudhu "boro dekhale kachhe" ei
shortcut use kore?

## 2. Mul idea: digital zoom ekta free ruler

Chhobir majhkhan theke kete (crop) abar boro (resize) korle sheta ekdom shothik digital zoom:
- focal length thik **s gun** bare
- depth **bodlay na** (kono jinish kachhe ashe na)

Tai label chhara test kora jay: zoom er age ar pore model er depth ar focal tulona korlei hoy.

## 3. Shonkhya

| | Mane | Thik model | Shortcut model |
|---|---|---|---|
| α | zoom e depth koto kome (D_s/D_1 = s^-α) | 0 | 1 |
| β | zoom e focal koto bare (f_s/f_1 = s^β) | 1 | 0 |

## 4. Theory

- **Tinta invariant:**
  - α=0 hole depth thik thake
  - β=1 hole shape thik thake
  - α+β=1 hole shudhu lateral coordinate (X) thik thake, depth ba shape na. "Boro dekhale
    kachhe" reasoning thik ei line e pore.
- **Zoom-squash ambiguity:** zoom kora chhobi = scene ke depth er dik borabor s gun chepte tola
  chhobi, pixel-by-pixel ek. Size dekhe eta dhora jay na, 3D shape lage. Tai shoja dewal
  (fronto-parallel) e kono model-i zoom bujhte pare na, ar jei scene e depth beshi chhoriye ache
  sekhane bhalo bujhe. Eta `scene_stats.csv` diye test hocche.

## 5. Agerkar kaj (2026-10-19 check kora)

| Paper | Ki koreche | Amader sathe shomporko |
|---|---|---|
| Zhong et al., ICCV 2023 | crop-resize self-consistency loss, unlabeled fine-tuning | EFT er mul idea oder. Amra metric depth e adapt kori, nijer method bolchi na |
| Kim & Lee, Electronics 2026 | stale intrinsics (purono K) e camera-input model er bhul | Amader Metric3D stale-K result oder kaj ke support kore, notun na |
| Roll-robust depth (2608.00678) | camera roll | Zoom alada: intrinsics bodlay |
| KPE (2312.06594) | crop location input hishebe | Test na, training input |

**Amader notun ongsho:**
1. β exponent ar camera anuman kora model gulo zoom e focal kom scale kore
2. α+β=1 line
3. zoom-squash theory ar depth-range test
4. teen dhoroner model ek scale e
5. label-free proxy
6. **resolution shortcut** finding ar EFT+B

## 6. Experiment

- **Model:** DA-V2-S/B, ZoeDepth (pixels only); Depth Pro, MoGe-2-S, UniDepthV2-S (predicts
  camera); Metric3D v2-S (takes camera, thik K ar stale K)
- **Data:** KITTI val selection, NYUv2 test, DIODE val indoor/outdoor. Protita theke 200 ta
  chhobi, shob model e ekoi chhobi.
- **Zoom:** 1, 1.25, 1.5, 2, 3x (majhkhan). Off-centre: 4 kona, 50 chhobi.
- **Control:**
  - blur control: zoom er moto jhapsa, kintu field of view ek
  - geometry check (notebook 01, error 0)
  - same-pixel reference: 1x prediction ke-o zoom box er pixel e mapa

## 7. Result (ekhon porjonto)

**Smoke test (12 chhobi):** 3x zoom e depth ratio (thik = 1):

| Model | Depth ratio |
|---|---|
| DA-V2 | 0.27-0.36 |
| Depth Pro / MoGe / UniDepth | 0.59-0.74 (focal 1.5-1.8x, thik hobe 3x) |
| Metric3D, thik K | 1.04 |
| Metric3D, stale K | 0.35 |

**Figure 1** (`outputs/figures/06_figure1.png`):

| | GT | DA-V2 (1x → 2x → 3x) | MoGe-2 (1x → 2x → 3x) |
|---|---|---|---|
| KITTI | 21 m | 28.5 → 15.1 → 7.8 m | 22 → 16 → 11.5 m |
| NYU | 2.3 m | 3.8 → 1.7 → 0.5 m | 2.4 → 2.8 → 2.1 m |

**EFT (200 chhobi/dataset):**

| Dataset | α frozen → EFT | AbsRel 3x frozen → EFT | AbsRel 1x frozen → EFT |
|---|---|---|---|
| KITTI | 1.03 → 0.18 | 0.665 → 0.158 | 0.138 → 0.156 |
| NYU | 0.98 → 0.12 | 0.563 → 0.226 | 0.344 → 0.313 |
| DIODE-in (unseen) | 0.70 → 0.39 | 0.558 → 0.364 | 0.280 → 0.368 |
| DIODE-out (unseen) | 0.65 → 0.39 | 0.551 → 0.648 | 0.731 → 0.877 |

**Shortcut:** EFT e blur control e KITTI te depth ~30% bodlay. Mane "jhapsa = zoom" shikheche.
Ei jonno EFT+B (blur-invariance term) banano hoyeche, Kaggle e cholche.

## 8. Baki kaj

1. audit-a2, audit-b, eft-baseline (anchor-only + EFT+B) shesh howa
2. notebook 02-05 chalano
3. abstract, intro, results, conclusion asol number diye lekha
4. (optional) phone e 1x/2x/5x zoom chhobi ar tape diye mapa dure
5. Overleaf e PDF, supervisor review, submit
