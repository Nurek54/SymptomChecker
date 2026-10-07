# backend/services/symptom_stats.py
"""
Statystyki P(objaw | choroba) liczone raz z danych treningowych.

Służą do wyboru pytań uzupełniających: pytamy o objaw, którego odpowiedź
(tak/nie) najbardziej zmniejszy niepewność co do aktualnie prawdopodobnych
chorób (expected information gain). Wynik jest cache'owany w pliku .npz,
więc CSV czytamy tylko przy pierwszym starcie albo po zmianie danych/cech.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional

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
        self.q_unit = (self.q / norms).astype(np.float32)       # profile objawów do wykrywania duplikatów
        # ile wierszy treningowych ma dany objaw; 0 = cecha "martwa" (model nic o niej nie wie)
        fc = np.zeros(len(self.features)) if feature_counts is None else np.asarray(feature_counts, dtype=np.float64)
        self.feature_counts: Dict[str, int] = {f: int(c) for f, c in zip(self.features, fc)}

    # ───────────────────────── budowa / cache ─────────────────────────
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

        # 1) cache
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

        # 2) budowa z CSV
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

            for f in feats:  # cechy nieobecne w CSV = 0
                if f not in df.columns:
                    df[f] = np.uint8(0)

            grp = df.groupby("diseases", sort=True)
            counts = grp.size().to_numpy(dtype=np.float64)
            sums = grp[feats].sum().to_numpy(dtype=np.float64)
            # wygładzanie Laplace'a: brak zer i jedynek
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

    # ───────────────────────── wybór pytania ─────────────────────────
    def best_question(
        self,
        posterior: Dict[str, float],
        candidates: List[str],
        reference: List[str],
        top_n_diseases: int = 40,
        dup_threshold: float = 0.95,
    ) -> Optional[str]:
        """
        Zwraca objaw z `candidates` o największym oczekiwanym przyroście informacji
        względem rozkładu `posterior`. Pomija objawy niemal identyczne (profil po
        chorobach) z objawami już znanymi/zadanymi (`reference`), np. "krwioplucie"
        po pytaniu o "krew w plwocinie".
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
        p1 = p @ Q                                                    # P(tak) dla każdego kandydata
        post1 = (p[:, None] * Q) / np.clip(p1, 1e-12, None)
        post0 = (p[:, None] * (1.0 - Q)) / np.clip(1.0 - p1, 1e-12, None)
        ig = h_prior - (p1 * _h(post1) + (1.0 - p1) * _h(post0))

        best = int(np.argmax(ig))
        if not np.isfinite(ig[best]) or ig[best] <= 1e-6:
            return None
        return cand[best]
