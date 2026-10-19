"""repair_cx restores the crop centre that the first audit-a2 run overwrote."""
import numpy as np
import pandas as pd

import analysis
import audit
from test_audit import SAMPLE, Equivariant


def buggy_rows():
    rows = []
    for k, c in enumerate(analysis.CENTRES):
        for r in audit.zoom_rows(Equivariant(), SAMPLE, 80, center=c, blur=(2.0,) if k == 0 else ()):
            # what the buggy version wrote
            if r["kind"] == "zoom":
                r["cx"], r["cx_base"] = r.pop("pp_x"), r.pop("pp_x_base")
                r["cx_gt"], r["cx_gt_base"] = r.pop("pp_x_gt"), r.pop("pp_x_gt_base")
            rows.append(r)
    return pd.DataFrame(rows)


def test_repair_restores_centres():
    fixed = analysis.repair_cx(buggy_rows())
    zoomed = fixed[fixed.kind == "zoom"]
    got = list(zip(zoomed.cx, zoomed.cy))
    want = [c for c in analysis.CENTRES for _ in analysis.ZOOMS]
    assert got == want
    assert np.allclose(zoomed.pp_x, zoomed.pp_x_gt)
