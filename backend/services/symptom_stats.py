"""
P(symptom | disease) statistics computed once from the training data.

Used to pick follow-up questions: ask about the symptom whose yes/no answer most
reduces uncertainty over the currently likely diseases (expected information gain).
Cached in an .npz file, so the CSV is read only on first start or after data/feature changes.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np

from config import DATA_CSV, MODEL_PATH, rel_to_backend

log = logging.getLogger("symptomchecker.symptom_stats")

STATS_CACHE_PATH = os.environ.get(
    "SYMPTOM_STATS_PATH",
    os.path.join(os.path.dirname(MODEL_PATH), "symptom_given_disease.npz"),
)


def _norm(s: str) -> str:
    return str(s or "").strip().lower().lstrip("﻿")


class SymptomStats:
    def __init__(self, features: List[str], classes: List[str], q: np.ndarray,
                 feature_counts: Optional[np.ndarray] = None):
        self.features = [_norm(f) for f in features]
        self.classes = [_norm(c) for c in classes]
        self.q = q.astype(np.float32, copy=False)               # (n_classes, n_features)
        self.feat_index: Dict[str, int] = {f: i for i, f in enumerate(self.features)}
        self.class_index: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        norms = np.linalg.norm(self.q, axis=0)
        norms[norms == 0] = 1.0
        self.q_unit = (self.q / norms).astype(np.float32)       # unit symptom profiles for near-duplicate detection
        # training rows containing each symptom; 0 = "dead" feature the model knows nothing about
        fc = np.zeros(len(self.features)) if feature_counts is None else np.asarray(feature_counts, dtype=np.float64)
        self.feature_counts: Dict[str, int] = {f: int(c) for f, c in zip(self.features, fc)}

    @staticmethod
    def _source_signature(csv_path: str) -> str:
        st = os.stat(csv_path)
        return f"{rel_to_backend(csv_path)}|{st.st_size}|{int(st.st_mtime)}"

    @classmethod
    def load_or_build(cls, features: List[str], csv_path: str = DATA_CSV,
                      cache_path: str = STATS_CACHE_PATH) -> Optional["SymptomStats"]:
        feats = [_norm(f) for f in (features or [])]
        if not feats:
            return None

        sig = None
        if csv_path and os.path.exists(csv_path):
            try:
                sig = cls._source_signature(csv_path)
            except OSError:
                sig = None

        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                c_feats = [str(x) for x in z["features"].tolist()]
                c_sig = str(z["signature"].tolist())
                if c_feats == feats and (sig is None or c_sig == sig) and "feature_counts" in z.files:
                    log.info("[Stats] Wczytano P(objaw|choroba) z cache (%s).", cache_path)
                    return cls(c_feats, [str(x) for x in z["classes"].tolist()], z["q"], z["feature_counts"])
                log.info("[Stats] Cache nieaktualny (zmiana cech lub danych) - przeliczam.")
            except Exception as e:
                log.warning("[Stats] Nie udało się wczytać cache %s: %s", cache_path, e)

        if sig is None:
            log.warning("[Stats] Brak pliku z danymi (%s) - wybór pytań użyje rankingu ważności cech.", csv_path)
            return None
        try:
            import pandas as pd
            header = pd.read_csv(csv_path, nrows=0).columns.tolist()
            by_norm = {_norm(c): c for c in header}
            if "diseases" not in by_norm:
                raise ValueError("brak kolumny 'diseases'")
            present = [f for f in feats if f in by_norm]
            usecols = [by_norm["diseases"]] + [by_norm[f] for f in present]
            dtypes = {by_norm[f]: np.uint8 for f in present}
            dtypes[by_norm["diseases"]] = str
            log.info("[Stats] Liczę P(objaw|choroba) z %s (jednorazowo)...", csv_path)
            df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
            df.columns = [_norm(c) for c in df.columns]
            df = df.dropna(subset=["diseases"])
            df["diseases"] = df["diseases"].map(_norm)

            for f in feats:  # features missing from the CSV count as 0
                if f not in df.columns:
                    df[f] = np.uint8(0)

            grp = df.groupby("diseases", sort=True)
            counts = grp.size().to_numpy(dtype=np.float64)
            sums = grp[feats].sum().to_numpy(dtype=np.float64)
            # additive smoothing keeps q strictly inside (0, 1)
            q = (sums + 0.5) / (counts[:, None] + 1.0)
            classes = grp.size().index.tolist()

            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    features=np.array(feats), classes=np.array(classes),
                    q=q.astype(np.float32), signature=np.array(sig),
                    feature_counts=sums.sum(axis=0),
                )
            except Exception as e:
                log.warning("[Stats] Nie zapisano cache: %s", e)

            log.info("[Stats] Gotowe: klasy=%d, cechy=%d.", len(classes), len(feats))
            return cls(feats, classes, q, sums.sum(axis=0))
        except Exception as e:
            log.warning("[Stats] Nie udało się policzyć statystyk: %s", e)
            return None

    def _neg_log_factor(self, negated: Iterable[str]) -> Optional[np.ndarray]:
        idx = sorted({self.feat_index[s] for s in (negated or []) if s in self.feat_index})
        if not idx:
            return None
        return np.log1p(-self.q[:, idx].astype(np.float64)).sum(axis=1)   # (n_classes,)

    def apply_negatives(self, dist: Dict[str, float], negated: Iterable[str]) -> Dict[str, float]:
        """
        Account for denied symptoms: p(d) * prod(1 - P(symptom | d)), then renormalize.

        XGB/River only see present symptoms; "I don't have X" and "nothing known about X" are the
        same zero to them. Without this, a "no" answer to a follow-up would not change the top-3,
        even though it rules out diseases where that symptom is nearly always present.
        """
        lf = self._neg_log_factor(negated)
        if lf is None or not dist:
            return dist
        f = np.exp(lf - lf.max())
        out = {c: float(p) * (float(f[self.class_index[c]]) if c in self.class_index else 1.0)
               for c, p in dist.items()}
        z = sum(out.values())
        return {c: v / z for c, v in out.items()} if z > 0 else dist

    def nb_posterior(self, present: Iterable[str], negated: Iterable[str]) -> Dict[str, float]:
        """
        Disease distribution from the stats alone (naive Bayes, uniform prior) using present and
        denied symptoms, assuming nothing about the rest. Better for choosing the next question
        than the model distribution, which treats missing information as an absent symptom.
        """
        logp = np.zeros(len(self.classes), dtype=np.float64)
        pos = sorted({self.feat_index[s] for s in (present or []) if s in self.feat_index})
        if pos:
            logp += np.log(self.q[:, pos].astype(np.float64)).sum(axis=1)
        lf = self._neg_log_factor(negated)
        if lf is not None:
            logp += lf
        p = np.exp(logp - logp.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}

    def best_question(
        self,
        posterior: Dict[str, float],
        candidates: List[str],
        reference: List[str],
        top_n_diseases: int = 40,
        dup_threshold: float = 0.95,
    ) -> Optional[str]:
        """
        Return the candidate with the highest expected information gain under `posterior`.
        Skips candidates whose disease profile nearly duplicates an already known/asked symptom
        in `reference` (e.g. "krwioplucie" after asking about "krew w plwocinie").
        """
        cand = [c for c in candidates if c in self.feat_index]
        if not cand:
            return None

        # near-duplicate filter
        ref_idx = [self.feat_index[r] for r in reference if r in self.feat_index]
        if ref_idx:
            cand_idx = np.array([self.feat_index[c] for c in cand])
            sims = self.q_unit[:, cand_idx].T @ self.q_unit[:, ref_idx]   # (C, R)
            keep = sims.max(axis=1) < dup_threshold
            cand = [c for c, k in zip(cand, keep) if k]
            if not cand:
                return None

        items = [(self.class_index[d], float(p)) for d, p in (posterior or {}).items()
                 if d in self.class_index and p > 0]
        if not items:
            return None
        items.sort(key=lambda x: -x[1])
        items = items[:top_n_diseases]
        cls_idx = np.array([i for i, _ in items])
        p = np.array([pp for _, pp in items], dtype=np.float64)
        p = p / p.sum()

        cand_idx = np.array([self.feat_index[c] for c in cand])
        Q = self.q[np.ix_(cls_idx, cand_idx)].astype(np.float64)        # (N, C)

        def _h(m: np.ndarray) -> np.ndarray:
            m = np.clip(m, 1e-12, 1.0)
            return -(m * np.log(m)).sum(axis=0)

        h_prior = float(_h(p[:, None])[0])
        p1 = p @ Q                                                    # P(yes) per candidate
        post1 = (p[:, None] * Q) / np.clip(p1, 1e-12, None)
        post0 = (p[:, None] * (1.0 - Q)) / np.clip(1.0 - p1, 1e-12, None)
        ig = h_prior - (p1 * _h(post1) + (1.0 - p1) * _h(post0))

        best = int(np.argmax(ig))
        if not np.isfinite(ig[best]) or ig[best] <= 1e-6:
            return None
        return cand[best]
