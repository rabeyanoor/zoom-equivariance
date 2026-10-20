"""Independent checks of the audit CSVs before any number goes into the paper.

    python scripts/verify_results.py            # exit code 1 if a check fails

Checks, per CSV:
  - every image has all zoom rows (1, 1.25, 1.5, 2, 3) for each crop centre, and one blur row;
  - no duplicated (image, centre, kind, zoom) rows (a resumed run must not double-count);
  - all models of the same run size saw exactly the same images (fair comparison);
  - the exact zoom factor s = fx'/fx is within 1 % of the nominal one;
  - the 1x self-consistency is exactly zero (the prediction is compared with itself);
  - the share of rows without ground-truth errors (too few valid GT pixels) stays small.
Writes outputs/tables/results_manifest.csv with one line per file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402

ZOOMS = (1.0, 1.25, 1.5, 2.0, 3.0)
problems, manifest = [], []


def check(cond, msg):
    if not cond:
        problems.append(msg)
    return cond


images_by_set = {}
for f in sorted(A.RESULTS.glob("*__*.csv")):
    df = A.repair_cx(pd.read_csv(f))
    tag = f.stem
    if "__cues" in tag:
        # cue split: the four interventions of audit.cue_rows for every image
        kinds = df.groupby("name").kind.apply(lambda v: tuple(sorted(v)))
        expected = ("context", "magnify", "shrink", "zoom")
        check((kinds == expected).all(), f"{tag}: {int((kinds != expected).sum())} images without all four cue rows")
        # the zoom row is the sum of context and magnify in log depth (same pixels, same medians)
        w = df.pivot_table(index="name", columns="kind", values="log_ratio_self")
        check(df.duplicated(["name", "kind"]).sum() == 0, f"{tag}: duplicated cue rows")
        manifest.append(dict(file=f.name, model=tag.split("__")[0], dataset=tag.split("__")[1],
                             images=df.name.nunique(), centres=1, rows=len(df),
                             cue_sum_gap=round(float((w.zoom - w.context - w.magnify).abs().median()), 4)))
        continue
    if "__ctrl" in tag:
        # control runs: one row per control kind and image
        kinds = df.groupby("name").kind.apply(lambda v: tuple(sorted(v)))
        expected = ("bilinear", "digital_lr", "jpeg", "optical")
        check((kinds == expected).all(), f"{tag}: {int((kinds != expected).sum())} images without all four controls")
        check(df.duplicated(["name", "kind"]).sum() == 0, f"{tag}: duplicated control rows")
        manifest.append(dict(file=f.name, model=tag.split("__")[0], dataset=tag.split("__")[1],
                             images=df.name.nunique(), centres=1, rows=len(df)))
        continue
    z = df[df.kind == "zoom"]
    n_img = df.name.nunique()
    centres = z.groupby("name")[["cx", "cy"]].apply(lambda g: len(g.drop_duplicates()))
    per = z.groupby(["name", "cx", "cy"]).zoom.apply(lambda v: tuple(sorted(np.round(v, 4))))
    check((per == ZOOMS).all(), f"{tag}: {int((per != ZOOMS).sum())} (image, centre) without all zooms")
    dup = df.duplicated(["name", "cx", "cy", "kind", "zoom"]).sum()
    check(dup == 0, f"{tag}: {dup} duplicated rows")
    check((df[df.kind == "blur"].groupby("name").size() == 1).all(), f"{tag}: blur rows missing or repeated")
    s_nom = z.zoom.values
    s_true = z.fx_gt.values / z.groupby(["name", "cx", "cy"]).fx_gt.transform("min").values
    check(np.nanmax(np.abs(s_true / s_nom - 1)) < 0.01, f"{tag}: exact zoom factor off by more than 1 %")
    one = z[z.zoom == 1.0].log_ratio_self
    check(np.nanmax(np.abs(one)) < 1e-6, f"{tag}: 1x self-consistency is not zero")
    no_gt = float(z.abs_rel.isna().mean())
    check(no_gt < 0.05, f"{tag}: {100 * no_gt:.1f} % of rows have no GT error")
    model, dataset = tag.split("__")[:2]
    if "__offc" not in tag:  # off-centre runs use a smaller sample
        images_by_set.setdefault((dataset, n_img), {})[model] = frozenset(df.name)
    manifest.append(dict(file=f.name, model=model, dataset=dataset, images=n_img,
                         centres=int(centres.max()), rows=len(df), no_gt_share=round(no_gt, 4),
                         has_base=bool("abs_rel_base" in df and df.abs_rel_base.notna().any()),
                         has_fix="abs_rel_fix1.0" in df))

for (dataset, n), sets in images_by_set.items():
    ref = next(iter(sets.values()))
    for model, names in sets.items():
        check(names == ref, f"{dataset} ({n} images): {model} saw different images than the others")

out = ROOT / "outputs" / "tables"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(manifest).to_csv(out / "results_manifest.csv", index=False)
print(f"{len(manifest)} files checked")
for p in problems:
    print("FAIL", p)
print("all checks passed" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
