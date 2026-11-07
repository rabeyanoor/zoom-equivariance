"""Summary table of the ZoomCal pixel-level pilot: one row per outputs/zoomcal/stress__<model>__<dataset>.csv.

    python scripts/zoomcal_summary.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((ROOT / "outputs" / "zoomcal").glob("stress__*.csv")):
    t = pd.read_csv(f)
    tau = t["spread3.0"].quantile(0.9)  # as in zoomcal_stress.py: the clean 90% quantile, same images
    gate = t["spread3.0"] <= tau
    rows.append(dict(
        model=t.model.iloc[0], dataset=t.dataset.iloc[0], n=len(t),
        ar_1x=t.ar_unzoomed.mean(), ar_mean_views=t.ar_mean_views.mean(), ar_consensus=t.ar_consensus.mean(),
        ar_3x=t.ar_zoom3.mean(), ar_3x_fixed=t.ar_zoom3_corrected.mean(),
        consensus_better=(t.ar_consensus < t.ar_unzoomed).mean(),
        ar_3x_selective=t.ar_zoom3_corrected.where(gate, t.ar_zoom3).mean(),
        removed=(t.ar_zoom3.mean() - t.ar_zoom3_corrected.mean()) / (t.ar_zoom3.mean() - t.ar_unzoomed.mean()),
        **{f"auc{d}": t[f"e2_auc_{d}"].median() for d in (0.1, 0.2, 0.4)},
        **{f"rec{d}": t[f"e2_recall_{d}"].median() for d in (0.1, 0.2, 0.4)},
        **{f"dh2_{k}": t[f"e4_{k}_dh2"].abs().median() for k in ("blur", "occlusion", "jpeg")},
        **{f"flag_{k}": (t[f"e4_{k}_spread2"] > t["spread2.0"].quantile(0.9)).mean()
           for k in ("blur", "occlusion", "jpeg")},
    ))
s = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(s.round(3).to_string(index=False))
s.to_csv(ROOT / "outputs" / "zoomcal" / "summary_stress.csv", index=False)
