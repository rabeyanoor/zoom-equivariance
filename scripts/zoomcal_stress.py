"""ZoomCal 2.0 pixel-level pilot (E2, E4 and the consensus baselines) on real depth maps.

For each image: the unzoomed prediction and centred zooms s in {1.25, 1.5, 2, 3}. Every view is brought back onto
the pixels of the 3x crop (the region all views share), in log depth: L_0 (unzoomed) ... L_4. Then

    h_v = median_p (L_v - L_0)                        per-view global offset (the zoom error, alpha log s)
    C(p) = median_v (L_v(p) - h_v)                    robust consensus, offsets removed (v = 0 included, h_0 = 0)
    r_v(p) = L_v(p) - h_v - C(p),  M(p) = max_v |r_v(p)|   pixel disagreement left after the offsets

E2  a log error delta inside a random rectangle (1/4 of the region) of one random zoomed view: pixel AUC of M,
    and recall at the threshold that flags 5% of the pixels of clean images.
E4  nuisances in the 2x view only: a grey occluder (20% of the image), the blur of a 2x down/up resample, JPEG 20.
    Error of h_2 against the clean h_2, and whether the shape spread of view 2 (median |r_2|) flags it.
H3  consensus vs. plain multi-scale averaging, AbsRel against LiDAR on the 3x region: unzoomed L_0, the mean of the
    raw views (exp mean_v L_v), and the consensus C.
Selective correction of the 3x view: D_3 exp(-h_3), applied only when view 3's spread is below a threshold set on
clean images.

    python scripts/zoomcal_stress.py --model dav2-s --n 40
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import datasets  # noqa: E402
import models  # noqa: E402
import zoom  # noqa: E402

ZOOMS = (1.25, 1.5, 2.0, 3.0)
OUT = ROOT / "outputs" / "zoomcal"


def _log(d):
    """Log depth; pixels without a valid prediction (MoGe-2 marks them with NaN/inf) become NaN."""
    d = np.asarray(d, np.float64)
    return np.where(np.isfinite(d) & (d > 0), np.log(np.maximum(d, 1e-3)), np.nan)


def on_common(depth_view, box_v, box_c):
    """A zoomed view's depth, back on its crop grid, then cut to the common (3x) box given in original pixels."""
    d = zoom.unzoom_depth(depth_view, box_v)
    x0, y0 = box_c[0] - box_v[0], box_c[1] - box_v[1]
    return d[y0:y0 + box_c[3], x0:x0 + box_c[2]]


def view(model, img, K, hw, s, edit=None):
    """Log depth of the zoom by s (s = 1: the unzoomed image) on the common (3x) box."""
    h, w = hw
    bc = zoom.zoom_box(h, w, 3.0)
    if s == 1.0:
        return _log(zoom.crop(model.predict(img, K)["depth"], bc))
    box = zoom.zoom_box(h, w, s)
    zi = zoom.zoom_image(img, box)
    if edit is not None:
        zi = edit(zi)
    d = model.predict(zi, zoom.zoom_K(K, box, hw))["depth"]
    return _log(on_common(d, box, bc))


def views(model, img, K, hw):
    """Log depth of the unzoomed view and every zoom on the common box."""
    return np.stack([view(model, img, K, hw, s) for s in (1.0,) + ZOOMS]), zoom.zoom_box(*hw, 3.0)


def diagnose(L):
    h = np.array([0.0] + [float(np.nanmedian(L[v] - L[0])) for v in range(1, len(L))])
    C = np.nanmedian(L - h[:, None, None], 0)
    r = L - h[:, None, None] - C[None]
    return h, C, r


def absrel(logd, gt, max_depth=80.0):
    m = (gt > 1e-3) & (gt <= max_depth) & np.isfinite(logd)
    if m.sum() < 50:
        return np.nan
    p = np.exp(logd[m])
    return float(np.mean(np.abs(p - gt[m]) / gt[m]))


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    return float(roc_auc_score(label[ok], score[ok]))


def occlude(img, rng, frac=0.2):
    h, w = img.shape[:2]
    bh, bw = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
    y0, x0 = rng.integers(0, h - bh), rng.integers(0, w - bw)
    out = img.copy()
    out[y0:y0 + bh, x0:x0 + bw] = 128
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dav2-s")
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    load = datasets.Loader(args.dataset)
    model = models.load(args.model, datasets.DOMAIN[args.dataset])
    rows, clean_M, clean_spread3 = [], [], []
    t0 = time.time()
    for k, i in enumerate(datasets.sample_ids(args.dataset, args.n, seed=1)):
        smp = load(i)
        img, K, gt = smp["img"], smp["K"], smp["depth"]
        hw = img.shape[:2]
        L, bc = views(model, img, K, hw)
        g = zoom.crop(gt, bc)
        h, C, r = diagnose(L)
        M = np.abs(r).max(0)  # NaN where any view has no valid depth
        clean_M.append(M[np.isfinite(M)])
        spread = np.nanmedian(np.abs(r), axis=(1, 2))
        clean_spread3.append(spread[4])
        row = dict(name=smp["name"], **{f"h{s}": h[j + 1] for j, s in enumerate(ZOOMS)},
                   **{f"spread{s}": spread[j + 1] for j, s in enumerate(ZOOMS)},
                   ar_unzoomed=absrel(L[0], g), ar_mean_views=absrel(L.mean(0), g), ar_consensus=absrel(C, g),
                   ar_zoom3=absrel(L[4], g), ar_zoom3_corrected=absrel(L[4] - h[4], g))
        # E2: a local error in one zoomed view
        H, W = M.shape
        for delta in (0.1, 0.2, 0.4):
            v = int(rng.integers(1, len(ZOOMS) + 1))
            rh, rw = H // 2, W // 2
            y0, x0 = int(rng.integers(0, H - rh)), int(rng.integers(0, W - rw))
            mask = np.zeros((H, W), bool)
            mask[y0:y0 + rh, x0:x0 + rw] = True
            Lc = L.copy()
            Lc[v][mask] += delta * rng.choice([-1, 1])
            _, _, rc = diagnose(Lc)
            Mc = np.abs(rc).max(0)
            row[f"e2_auc_{delta}"] = auc(Mc, mask)
            row[f"e2_M_{delta}"] = Mc  # thresholded after the loop
            row[f"e2_mask_{delta}"] = mask
        # E4: nuisances in the 2x view only
        edits = {"occlusion": lambda zi: occlude(zi, np.random.default_rng(i)),
                 "blur": lambda zi: zoom.blur_control(zi, 2.0), "jpeg": lambda zi: zoom.jpeg(zi, 20)}
        for kind, edit in edits.items():
            Ln = L.copy()
            Ln[3] = view(model, img, K, hw, 2.0, edit)  # only the 2x view changes
            hn, _, rn = diagnose(Ln)
            row[f"e4_{kind}_dh2"] = hn[3] - h[3]
            row[f"e4_{kind}_spread2"] = float(np.nanmedian(np.abs(rn[3])))
            row[f"e4_{kind}_mean_dh2"] = float(np.nanmean(Ln[3] - Ln[0])) - float(np.nanmean(L[3] - L[0]))
        rows.append(row)
        print(f"{k + 1}/{args.n} {time.time() - t0:.0f}s", flush=True)

    thr = float(np.quantile(np.concatenate(clean_M), 0.95))
    for row in rows:
        for delta in (0.1, 0.2, 0.4):
            Mc, mask = row.pop(f"e2_M_{delta}"), row.pop(f"e2_mask_{delta}")
            ok = np.isfinite(Mc)
            row[f"e2_recall_{delta}"] = float((Mc[mask & ok] > thr).mean())
            row[f"e2_fpr_{delta}"] = float((Mc[~mask & ok] > thr).mean())
    t = pd.DataFrame(rows)
    t["model"], t["dataset"], t["thr_M"] = args.model, args.dataset, thr
    out = out_dir / f"stress__{args.model}__{args.dataset}.csv"
    t.to_csv(out, index=False)

    # selective correction of the 3x view: threshold on view-3 spread from clean images (cross-fitted halves)
    tau = float(np.quantile(clean_spread3, 0.9))
    print(f"\n{args.model} {args.dataset}, {len(t)} images -> {out}")
    print("H3 AbsRel on the 3x region: unzoomed %.3f | mean of raw views %.3f | consensus %.3f" % (
        t.ar_unzoomed.mean(), t.ar_mean_views.mean(), t.ar_consensus.mean()))
    print("3x view: AbsRel raw %.3f | corrected by h_3 %.3f | unzoomed on same px %.3f" % (
        t.ar_zoom3.mean(), t.ar_zoom3_corrected.mean(), t.ar_unzoomed.mean()))
    print("E2 pixel AUC / recall / FPR at the clean 5%% threshold:")
    for d in (0.1, 0.2, 0.4):
        print(f"  delta {d}: AUC {t[f'e2_auc_{d}'].median():.3f}  recall {t[f'e2_recall_{d}'].median():.3f}"
              f"  FPR {t[f'e2_fpr_{d}'].median():.3f}")
    gated = np.where(t["spread3.0"] <= tau, t.ar_zoom3_corrected, t.ar_zoom3)
    print(f"selective (correct only if view-3 spread <= {tau:.3f}, the clean 90% quantile): AbsRel {np.nanmean(gated):.3f}, "
          f"corrected {(t['spread3.0'] <= tau).mean():.2f} of images")
    tau2 = float(t["spread2.0"].quantile(0.9))
    print(f"E4 (2x view only), median |change of h_2| (median-based / mean-based), share flagged (spread > tau2={tau2:.3f}):")
    for kind in ("occlusion", "blur", "jpeg"):
        print(f"  {kind:9s} {t[f'e4_{kind}_dh2'].abs().median():.3f} / {t[f'e4_{kind}_mean_dh2'].abs().median():.3f}"
              f"  flagged {(t[f'e4_{kind}_spread2'] > tau2).mean():.2f}  (clean: 0.10 by construction)")


if __name__ == "__main__":
    main()
