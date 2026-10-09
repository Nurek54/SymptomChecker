"""
Builds cleaned datasets in backend/data/v2/ (the original files are left untouched).

Problems found in the previous files:
  1. Final_Testset_for_Eval.csv repeats 100% of its rows (symptom vector + disease)
     from train_remainder.csv (77%) and calibration.csv (23%) -> inflated test metrics.
  2. The merged Dataset_2 (Kaggle) added duplicates of existing diseases under other
     names (bronchial_asthma/asthma, osteoarthristis/osteoarthritis, dengue/dengue_fever,
     gerd/..., chicken_pox/chickenpox, aids/hiv, ...) -> the model splits probability
     between two labels of the same disease.
  3. Hundreds of classes have only a few rows (min. 1) and class weights reached ~209 ->
     the model overrates diseases it knows almost nothing about.

What the script does:
  - pools train_remainder + calibration + test and removes duplicates,
  - merges duplicate labels (MERGE_LABELS),
  - drops classes with fewer than --min-rows rows,
  - splits stratified into train/calibration/test so that the same symptom
    vector never lands in two sets,
  - computes class weights sqrt(N / (K * n_c)) clipped to [--w-min, --w-max],
  - writes the build_report.json report.

Run (from backend/):
    python data/build_datasets.py
Then delete the old model artifacts in backend/ml (see README) to retrain the models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SOURCES = ["train_remainder.csv", "calibration.csv", "Final_Testset_for_Eval.csv"]
SPECIAL = {"diseases", "weight", "xgb_label"}

# Dataset_2 labels that denote the same disease as a Dataset_1 label
MERGE_LABELS = {
    "bronchial_asthma": "asthma",
    "osteoarthristis": "osteoarthritis",
    "gerd": "gastroesophageal_reflux_disease_gerd",
    "peptic_ulcer_diseae": "gastroduodenal_ulcer",
    "dengue": "dengue_fever",
    "chicken_pox": "chickenpox",
    "aids": "human_immunodeficiency_virus_infection_hiv",
    "dimorphic_hemmorhoids_piles": "hemorrhoids",
    "paralysis_brain_hemorrhage": "intracerebral_hemorrhage",
    "cervical_spondylosis": "spondylosis",
    "gastroenteritis": "infectious_gastroenteritis",
    "hepatitis_a": "viral_hepatitis",
    "hepatitis_b": "viral_hepatitis",
    "hepatitis_c": "viral_hepatitis",
    "hepatitis_d": "viral_hepatitis",
    "hepatitis_e": "viral_hepatitis",
}


def _read(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {c: np.uint8 for c in header}
    dtypes[header[0]] = str
    for c in header:
        if c.strip().lower().lstrip("﻿") in ("weight", "xgb_label"):
            dtypes[c] = np.float32
    df = pd.read_csv(path, dtype=dtypes)
    df.columns = [c.strip().lower().lstrip("﻿") for c in df.columns]
    return df


def _vector_keys(X: np.ndarray) -> np.ndarray:
    packed = np.packbits(X, axis=1)
    return np.array([hashlib.md5(r.tobytes()).hexdigest() for r in packed], dtype=object)


def _split(df: pd.DataFrame, frac_cal: float, frac_test: float, seed: int) -> pd.Series:
    """Stratified per-class split; then each symptom vector is moved entirely into one set."""
    rng = np.random.default_rng(seed)
    part = pd.Series("train", index=df.index)
    for _, idx in df.groupby("diseases").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * frac_test)))
        n_cal = max(1, int(round(n * frac_cal)))
        if n - n_test - n_cal < 1:  # very small classes: keep at least 1 row for training
            n_cal = max(0, n - n_test - 1)
        part.loc[idx[:n_test]] = "test"
        part.loc[idx[n_test:n_test + n_cal]] = "calibration"
    # same vector (different diseases) -> one set; train wins, then calibration
    order = {"train": 0, "calibration": 1, "test": 2}
    best = part.map(order).groupby(df["vk"]).transform("min")
    inv = {v: k for k, v in order.items()}
    return best.map(inv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-rows", type=int, default=10, help="minimalna liczba przykładów na chorobę")
    ap.add_argument("--cal", type=float, default=0.15, help="udział zbioru kalibracyjnego")
    ap.add_argument("--test", type=float, default=0.15, help="udział zbioru testowego")
    ap.add_argument("--w-min", type=float, default=0.5)
    ap.add_argument("--w-max", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DATA_DIR / "v2"))
    args = ap.parse_args()

    frames = []
    for name in SOURCES:
        df = _read(DATA_DIR / name)
        df["source"] = name
        frames.append(df)
        print(f"[read] {name}: {len(df)} wierszy")
    pool = pd.concat(frames, ignore_index=True)
    feats = [c for c in pool.columns if c not in SPECIAL and c != "source"]

    report: dict = {"sources": SOURCES, "features": len(feats), "rows_in": int(len(pool))}

    pool["diseases"] = pool["diseases"].astype(str).str.strip().str.lower().replace(MERGE_LABELS)
    pool["vk"] = _vector_keys(pool[feats].to_numpy(np.uint8))

    # leakage in the old files: share of test rows also present in train/calibration
    old_test = pool["source"] == "Final_Testset_for_Eval.csv"
    seen = set(zip(pool.loc[~old_test, "vk"], pool.loc[~old_test, "diseases"]))
    report["old_test_rows_also_in_train_or_calib_pct"] = round(
        100.0 * float(np.mean([(k, d) in seen for k, d in zip(pool.loc[old_test, "vk"], pool.loc[old_test, "diseases"])])), 2
    )

    pool = pool.drop_duplicates(["vk", "diseases"]).reset_index(drop=True)
    report["rows_after_dedup"] = int(len(pool))
    report["merged_labels"] = MERGE_LABELS

    vc = pool["diseases"].value_counts()
    dropped = vc[vc < args.min_rows]
    pool = pool[~pool["diseases"].isin(dropped.index)].reset_index(drop=True)
    report["min_rows"] = args.min_rows
    report["dropped_classes"] = {k: int(v) for k, v in dropped.items()}
    report["rows_after_min_rows"] = int(len(pool))

    pool["part"] = _split(pool, args.cal, args.test, args.seed)

    train = pool[pool["part"] == "train"]
    n, k = len(train), train["diseases"].nunique()
    counts = train["diseases"].value_counts()
    w = np.sqrt(n / (k * counts)).clip(args.w_min, args.w_max)
    weights = pd.DataFrame({"disease": counts.index, "count": counts.values, "class_weight": w.values})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["diseases"] + feats
    tr = train[cols].copy()
    tr.insert(1, "weight", tr["diseases"].map(dict(zip(weights.disease, weights.class_weight))).astype(float).round(6))
    tr.to_csv(out / "train.csv", index=False)
    pool.loc[pool["part"] == "calibration", cols].to_csv(out / "calibration.csv", index=False)
    pool.loc[pool["part"] == "test", cols].to_csv(out / "test.csv", index=False)
    weights.to_csv(out / "class_weight.csv", index=False)

    # sanity check: no symptom vector appears in two sets
    sets = {p: set(pool.loc[pool["part"] == p, "vk"]) for p in ("train", "calibration", "test")}
    report["vector_overlap"] = {
        "train_calibration": len(sets["train"] & sets["calibration"]),
        "train_test": len(sets["train"] & sets["test"]),
        "calibration_test": len(sets["calibration"] & sets["test"]),
    }
    for p in ("train", "calibration", "test"):
        sub = pool[pool["part"] == p]
        report[p] = {"rows": int(len(sub)), "classes": int(sub["diseases"].nunique())}
    report["class_weight"] = {"min": float(w.min()), "max": float(w.max())}
    report["classes_total"] = int(pool["diseases"].nunique())

    # symptom dictionary vs data: entries that map only to features with no training data
    fcount = train[feats].sum()
    report["features_without_train_data"] = [f for f in feats if int(fcount[f]) == 0]
    try:
        sym = json.load(open(DATA_DIR / "symptoms_pl_en.json", encoding="utf-8"))
        unreachable, covered = [], set()
        for r in sym:
            labs = r.get("xgb_label")
            labs = [labs] if isinstance(labs, str) else list(labs or [])
            alive = [l for l in labs if l in fcount.index and int(fcount[l]) > 0]
            if alive:
                covered.add(max(alive, key=lambda l: int(fcount[l])))
            else:
                unreachable.append(r.get("pl"))
        report["dictionary_entries_without_data"] = unreachable
        report["features_with_data_but_no_dictionary_entry"] = [
            f for f in feats if int(fcount[f]) > 0 and f not in covered
        ]
    except Exception as e:  # the dictionary is optional for building the data
        report["dictionary_check_error"] = str(e)

    with open(out / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    skip = ("dropped_classes", "merged_labels", "features_without_train_data",
            "dictionary_entries_without_data", "features_with_data_but_no_dictionary_entry")
    print(json.dumps({k: v for k, v in report.items() if k not in skip}, ensure_ascii=False, indent=2))
    for k in skip[2:]:
        if report.get(k):
            print(f"[uwaga] {k}: {len(report[k])} (szczegóły w build_report.json)")
    assert all(v == 0 for v in report["vector_overlap"].values()), "przeciek między zbiorami!"
    print(f"[ok] zapisano w {out}")


if __name__ == "__main__":
    os.chdir(DATA_DIR.parent)
    main()
