"""
Export raw XGBoost and River predictions on the calibration and test sets
(for ensemble analysis without retraining the models).

Run (from backend/, with trained models):
    python tools/export_predictions.py

Writes ml/preds_calibration.npz and ml/preds_test.npz with fields:
    xgb_margins (N x K)  raw XGBoost margins
    river_z     (N x K)  raw River OvR logits, in XGB class order
    river_base  (K,)     River logits for an empty input
    y           (N,)     true class index (XGB class order)
    classes     (K,)     class names
"""
from __future__ import annotations

import logging
import os
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.dirname(BACKEND), BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import TEST_CSV, XGB_CALIBRATION_CSV  # noqa: E402
from models.river_manager import RiverManager  # noqa: E402
from models.xgb_manager import XGBManager  # noqa: E402
from utils import fmt_duration  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("export_predictions")


def export(xgb: XGBManager, river: RiverManager, path: str, out: str, label: str) -> None:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().lstrip("﻿") for c in df.columns]
    df["diseases"] = df["diseases"].astype(str).str.strip().str.lower()
    classes = [str(c).lower() for c in xgb.le.classes_]
    cidx = {c: i for i, c in enumerate(classes)}
    df = df[df["diseases"].isin(cidx)].reset_index(drop=True)
    for f in xgb.features:
        if f not in df.columns:
            df[f] = 0
    y = df["diseases"].map(cidx).to_numpy(dtype=np.int32)

    margins = xgb._predict_margins_batch(df[xgb.features].to_numpy(dtype=np.float32), f"{label}: XGB")
    with river.lock:
        Z, _, r_classes = river._scores_for_frame(df, f"{label}: River")
        z_base = river._base_scores(r_classes)
    # align River classes to XGB order (classes missing in River get a very low logit)
    Zx = np.full((len(df), len(classes)), -30.0, dtype=np.float32)
    bx = np.full(len(classes), -30.0, dtype=np.float32)
    r_idx = {str(c).lower(): j for j, c in enumerate(r_classes)}
    for c, i in cidx.items():
        j = r_idx.get(c)
        if j is not None:
            Zx[:, i] = Z[:, j]
            bx[i] = z_base[j]
    np.savez_compressed(out, xgb_margins=np.asarray(margins, dtype=np.float32), river_z=Zx,
                        river_base=bx, y=y, classes=np.array(classes))
    log.info("Zapisano %s (N=%d, K=%d, %.1f MB)", out, len(df), len(classes), os.path.getsize(out) / 1e6)


def main() -> None:
    t0 = time.time()
    xgb = XGBManager()
    xgb.train()  # with existing artifacts this only loads the model
    river = RiverManager()
    river.set_features(xgb.features)
    river.train()
    export(xgb, river, XGB_CALIBRATION_CSV, "ml/preds_calibration.npz", "kalibracja")
    export(xgb, river, TEST_CSV, "ml/preds_test.npz", "test")
    log.info("Gotowe w %s.", fmt_duration(time.time() - t0))


if __name__ == "__main__":
    main()
