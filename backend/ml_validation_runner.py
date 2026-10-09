"""
Standalone ML model validator (engineering thesis).

Run from the backend directory:

    python ml_validation_runner.py

Assumes that:
- config.TEST_CSV points to the frozen test set,
- XGB and River models are already trained and saved at MODEL_PATH / RIVER_MODEL_PATH,
- EnsembleTopK uses the same XGB/River instances as the backend.
"""

from __future__ import annotations

import json
import os
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

# headless backend; must be set before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from config import (  # noqa: E402
    TEST_CSV,
    MODEL_PATH,
    LE_PATH,
    RIVER_MODEL_PATH,
    ENSEMBLE_W_XGB,
    ENSEMBLE_W_RIVER,
    ENSEMBLE_K_FUSE,
    ENSEMBLE_MIN_XGB_WEIGHT,
    ENSEMBLE_MIN_RIVER_WEIGHT,
    ENSEMBLE_CONF_GAP,
)

from models.xgb_manager import XGBManager
from models.river_manager import RiverManager
from services.ensemble import EnsembleTopK
from utils import seed_everywhere

logger = logging.getLogger(__name__)

# Slightly larger plot defaults so figures stay readable in the PDF/TeX thesis
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.autolayout": False,
    }
)


@dataclass
class _DummyState:
    """Minimal app state required by EnsembleTopK."""

    xgb: XGBManager
    river: RiverManager
    ensemble_T: float = 1.0  # fusion temperature from services/ensemble_calibration.py
    ensemble_mode: str = "heuristic"  # fusion mode chosen on the calibration set
    ensemble_wx: float | None = None


class MLValidationRunner:
    """
    Implements the procedure from the thesis chapter on ML prediction quality validation.

    Steps:
      - load the frozen TEST_CSV,
      - load the trained XGB + River models and the ensemble,
      - compute all metrics (Top-1, Top-3, bootstrap, ECE, baseline),
      - generate artifacts W1-W6,
      - check criteria K1-K4 and the pass condition:
          K1/K2: ensemble significantly better than the best single model in Top-1/Top-3
                 (exact McNemar test, p<0.05; paired comparison on the same cases),
          K3: ensemble makes no more errors than either model,
          K4: ensemble ECE no worse than the worse model,
          pass: at least 2 of K1-K3 and no significant regression in any class
          (drop >10 pp, per-class McNemar test with Holm correction),
      - write the summary to JSON.
    """

    def __init__(
        self,
        out_dir: str | Path = "reports/ml_validation",
        n_bootstrap: int = 1000,
        n_bins_calibration: int = 10,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.n_bootstrap = int(n_bootstrap)
        self.n_bins_calibration = int(n_bins_calibration)

        self.df: pd.DataFrame | None = None
        self.y_true: np.ndarray | None = None

        self.xgb: XGBManager | None = None
        self.river: RiverManager | None = None
        self.ens: EnsembleTopK | None = None

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        # try relative to backend/ (next to config.py), then the repo root
        backend_dir = Path(__file__).resolve().parent
        cand1 = backend_dir / path
        if cand1.exists():
            return cand1
        cand2 = backend_dir.parent / path
        if cand2.exists():
            return cand2
        # return as-is; a missing file fails later with a clear error
        return path

    def load_testset(self) -> None:
        test_path = self._resolve_path(TEST_CSV)
        logger.info(f"[1/7] Wczytywanie zbioru testowego z: {test_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Brak zbioru testowego: {test_path}")
        df = pd.read_csv(test_path)
        if "diseases" not in df.columns:
            raise ValueError("TEST_CSV musi mieć kolumnę 'diseases' z etykietą klasy.")
        self.df = df
        self.y_true = df["diseases"].astype(str).to_numpy()
        logger.info(f"Zbiór testowy: {len(self.df)} rekordów, {len(self.df.columns)} kolumn.")

    def load_models(self) -> None:
        logger.info("[2/7] Ładowanie gotowych modeli XGB i River z dysku.")
        if not (self._resolve_path(MODEL_PATH).exists() and self._resolve_path(LE_PATH).exists()):
            raise FileNotFoundError(f"Brak artefaktów XGB: {MODEL_PATH} / {LE_PATH}")
        if not self._resolve_path(RIVER_MODEL_PATH).exists():
            raise FileNotFoundError(f"Brak artefaktów River: {RIVER_MODEL_PATH}")

        logger.info("  • Ładowanie XGBManager (artefakty istnieją – bez ponownego trenowania).")
        xgb = XGBManager()
        xgb.train()  # with existing artifacts: load + warm-up only

        logger.info("  • Ładowanie RiverManager (artefakty istnieją – bez bootstrapu od zera).")
        river = RiverManager()
        river.set_features(xgb.features)
        river.train()  # with existing artifacts: load + calibration if needed

        st = _DummyState(xgb=xgb, river=river)
        # same ensemble temperature as the backend (cached in ml/ensemble_calibration.json)
        try:
            from services.ensemble_calibration import ensure_ensemble_calibration
            ensure_ensemble_calibration(st)
            logger.info(f"  • Ensemble: tryb={st.ensemble_mode}, waga XGB={st.ensemble_wx}, T_ens={st.ensemble_T:.4f}.")
        except Exception as e:  # validation must also work without ensemble calibration
            logger.warning(f"  • Brak kalibracji ensemble (T_ens=1): {e}")
        ens = EnsembleTopK(st)

        self.xgb = xgb
        self.river = river
        self.ens = ens
        self._state = st

        logger.info(f"  • Załadowano modele. Liczba cech XGB/River: {len(self.xgb.features)}.")

    @staticmethod
    def _topk_hits(y_true: np.ndarray, preds_topk: List[List[str]], k: int) -> np.ndarray:
        hits = []
        for yt, top in zip(y_true, preds_topk):
            hits.append(1.0 if top and yt in top[:k] else 0.0)
        return np.asarray(hits, dtype=float)

    @staticmethod
    def _bootstrap_ci(
        hits: np.ndarray,
        n_boot: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Bootstrap CI for the hit rate (Top-1 or Top-3)."""
        rng = np.random.default_rng(seed)
        n = len(hits)
        if n == 0:
            return 0.0, 0.0, 0.0
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[i] = hits[idx].mean()
        lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(hits.mean()), float(lower), float(upper)

    @staticmethod
    def _per_class_acc(
        y_true: np.ndarray,
        preds_top1: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Per-class Top-1 accuracy and class counts."""
        acc: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for yt, yp in zip(y_true, preds_top1):
            counts[yt] = counts.get(yt, 0) + 1
            if yt == yp:
                acc[yt] = acc.get(yt, 0.0) + 1.0
        for cls, n in counts.items():
            acc[cls] = acc.get(cls, 0.0) / float(n)
        return acc, counts

    @staticmethod
    def _calibration_bins(
        conf: np.ndarray,
        hits: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin confidences into equal-width calibration bins (empty bins are skipped).

        Returns (mean_conf, acc, counts) per non-empty bin.
        """
        if len(conf) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        conf = np.asarray(conf, dtype=float)
        hits = np.asarray(hits, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        mean_conf = []
        acc = []
        counts = []
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            mean_conf.append(conf[mask].mean())
            acc.append(hits[mask].mean())
            counts.append(int(mask.sum()))
        return np.asarray(mean_conf), np.asarray(acc), np.asarray(counts, dtype=int)

    @staticmethod
    def _ece(conf: np.ndarray, hits: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE) in [0, 1]."""
        mconf, acc, counts = MLValidationRunner._calibration_bins(conf, hits, n_bins=n_bins)
        n = counts.sum()
        if n == 0:
            return 0.0
        return float(np.sum(counts * np.abs(acc - mconf)) / n)

    @staticmethod
    def _features_hash(features: List[str]) -> str:
        s = "|".join(features)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

    def _predict_all(self) -> Dict[str, Dict[str, np.ndarray] | List[List[str]]]:
        """
        Batch XGB, River and ensemble predictions for the whole test set.

        Uses exactly the same transforms as the backend (XGB and River calibrators,
        EnsembleTopK fusion with T_ens; parity is covered by regression tests), but computes
        whole matrices instead of 3 calls per row: minutes instead of hours.
        """
        assert self.df is not None and self.y_true is not None
        assert self.xgb is not None and self.river is not None and self.ens is not None
        from services.ensemble_calibration import model_distributions_df, ensemble_distribution

        feats = self.xgb.features
        if not feats:
            raise RuntimeError("XGBManager.features jest puste – sprawdź trening.")
        missing = [f for f in feats if f not in self.df.columns]
        if missing:
            raise RuntimeError(f"Zbiór testowy nie zawiera cech XGB: {missing[:5]}...")

        logger.info(f"[3/7] Generowanie predykcji dla wszystkich próbek testowych (n={len(self.df)}).")
        st = self._state
        Px, Pr, y_idx, classes = model_distributions_df(st, self.df, "Walidacja")
        Pe = ensemble_distribution(Px, Pr, self.ens, st)
        cls = np.asarray(classes, dtype=object)
        # model_distributions_df drops rows with classes unknown to the model, so realign labels
        y_true = cls[y_idx]
        if len(y_true) != len(self.y_true):
            logger.warning(f"  • Pominięto {len(self.y_true) - len(y_true)} wierszy z klasami spoza modelu.")
        self.y_true = y_true
        self._y_idx = y_idx
        self._probs = {"xgb": Px, "river": Pr, "ens": Pe}

        def top3(P: np.ndarray) -> Tuple[List[List[str]], np.ndarray]:
            idx = np.argsort(-P, axis=1)[:, :3]
            return [list(cls[r]) for r in idx], P[np.arange(len(P)), idx[:, 0]]

        preds_top3_xgb, conf_xgb = top3(Px)
        preds_top3_river, conf_river = top3(Pr)
        preds_top3_ens, conf_ens = top3(Pe)

        examples_rows: List[Dict[str, str]] = []
        for i in range(min(10, len(y_true))):
            yt, lx, lr, le = y_true[i], preds_top3_xgb[i], preds_top3_river[i], preds_top3_ens[i]
            ok_x, ok_r, ok_e = yt in lx[:1], yt in lr[:1], yt in le[:1]
            comment = ""
            if ok_e and (not ok_x or not ok_r):
                comment = "Ensemble poprawił błąd modelu pojedynczego."
            elif ok_x and ok_r and not ok_e:
                comment = "Ensemble zepsuł zgodną poprawną predykcję."
            elif ok_x and not ok_r and not ok_e:
                comment = "Tylko XGB trafił Top-1."
            elif ok_r and not ok_x and not ok_e:
                comment = "Tylko River trafił Top-1."
            examples_rows.append({"test_id": str(i), "true": yt, "xgb_top3": ", ".join(lx),
                                  "river_top3": ", ".join(lr), "ens_top3": ", ".join(le), "comment": comment})
        logger.info("  • Predykcje zakończone, zapisywanie examples.csv (pierwsze 10 rekordów).")
        pd.DataFrame(examples_rows).to_csv(self.out_dir / "examples.csv", index=False)

        return {
            "preds_top3_xgb": preds_top3_xgb,
            "preds_top3_river": preds_top3_river,
            "preds_top3_ens": preds_top3_ens,
            "conf_xgb": np.asarray(conf_xgb, dtype=float),
            "conf_river": np.asarray(conf_river, dtype=float),
            "conf_ens": np.asarray(conf_ens, dtype=float),
        }

    def _plot_confusion(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str,
        path: Path,
        normalize_rows: bool = False,
    ) -> None:
        logger.info(f"  • Rysowanie macierzy pomyłek: {path.name}")

        if normalize_rows:
            with np.errstate(divide="ignore", invalid="ignore"):
                row_sums = cm.sum(axis=1, keepdims=True)
                m = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        else:
            m = cm

        # size scales with the number of classes, clamped to [6, 14] inches
        n_labels = len(labels)
        size = max(6.0, min(14.0, 0.18 * n_labels))
        fig, ax = plt.subplots(figsize=(size, size))

        # higher-contrast colormap than the pale "Blues"
        im = ax.imshow(m, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel("Przewidziana klasa")
        ax.set_ylabel("Prawdziwa klasa")
        ax.set_title(title)

        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        ax.set_xticklabels(labels, rotation=90, ha="right")
        ax.set_yticklabels(labels)

        ax.grid(False)
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def _plot_bars_topk(self, metrics: Dict[str, Dict[str, float]], path: Path) -> None:
        """
        (W2) Top-1/Top-3 bars with bootstrap error bars.
        metrics[name] = {top1, top1_lo, top1_hi, top3, top3_lo, top3_hi}
        """
        logger.info(f"[5/7] Rysowanie słupków Top-1/Top-3 z przedziałami bootstrap: {path.name}")
        models = ["xgb", "river", "ens"]
        idx = np.arange(len(models))

        top1 = [metrics[m]["top1"] for m in models]
        top3 = [metrics[m]["top3"] for m in models]

        top1_err_lo = [metrics[m]["top1"] - metrics[m]["top1_lo"] for m in models]
        top1_err_hi = [metrics[m]["top1_hi"] - metrics[m]["top1"] for m in models]
        top3_err_lo = [metrics[m]["top3"] - metrics[m]["top3_lo"] for m in models]
        top3_err_hi = [metrics[m]["top3_hi"] - metrics[m]["top3"] for m in models]

        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(idx - width / 2, top1, width, label="Top-1")
        ax.errorbar(
            idx - width / 2,
            top1,
            yerr=[top1_err_lo, top1_err_hi],
            fmt="none",
            capsize=4,
        )

        ax.bar(idx + width / 2, top3, width, label="Top-3")
        ax.errorbar(
            idx + width / 2,
            top3,
            yerr=[top3_err_lo, top3_err_hi],
            fmt="none",
            capsize=4,
        )

        ax.set_xticks(idx)
        ax.set_xticklabels(["XGB", "River", "Ens"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Odsetek trafień")
        ax.set_title("Top-1 / Top-3 z przedziałami bootstrap (XGB / River / Ensemble)")
        ax.legend()
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def _plot_per_class(
            self,
            acc: Dict[str, float],
            counts: Dict[str, int],
            title: str,
            path: Path,
            top_k: int | None = None,
            labels_subset: List[str] | None = None,
            sort_desc: bool = True,
    ) -> None:
        """
        (W3) Per-class accuracy bars with class counts on a secondary axis.
        top_k keeps the top_k classes by count; labels_subset restricts to those classes.
        sort_desc=True sorts from most to least frequent, False the other way.
        """
        logger.info(f"  • Rysowanie trafności per klasa: {path.name}")

        items = list(acc.items())

        if labels_subset is not None:
            labels_set = set(labels_subset)
            items = [kv for kv in items if kv[0] in labels_set]

        items = sorted(items, key=lambda kv: counts.get(kv[0], 0), reverse=sort_desc)
        if top_k is not None:
            items = items[:top_k]

        if not items:
            logger.warning(f"  • Brak klas do narysowania dla: {title}")
            return

        labels = [k for k, _ in items]
        values = [v for _, v in items]
        ns = [counts.get(k, 0) for k in labels]

        x = np.arange(len(labels))
        # width scales with the number of classes, clamped to [8, 18] inches
        width_fig = max(8.0, min(18.0, 0.12 * len(labels)))
        fig, ax1 = plt.subplots(figsize=(width_fig, 5))

        ax1.bar(x, values, alpha=0.8)
        ax1.set_ylabel("Top-1 per klasa")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=90, fontsize=7)

        ax2 = ax1.twinx()
        ax2.plot(x, ns, linestyle="--", marker=".", linewidth=1)
        ax2.set_ylabel("Liczność klasy")

        ax1.set_title(title)
        ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def _plot_calibration(
        self,
        conf: np.ndarray,
        hits: np.ndarray,
        name: str,
        path: Path,
    ) -> float:
        """(W4) Reliability diagram; returns ECE."""
        logger.info(f"  • Rysowanie wykresu kalibracji dla {name}: {path.name}")
        mconf, acc, counts = self._calibration_bins(conf, hits, n_bins=self.n_bins_calibration)
        ece = self._ece(conf, hits, n_bins=self.n_bins_calibration)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="idealna kalibracja")
        if len(mconf) > 0:
            ax.plot(mconf, acc, marker="o", linewidth=1.5, label=name)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Deklarowana pewność")
        ax.set_ylabel("Rzeczywista trafność")
        ax.set_title(f"Wykres kalibracji – {name}")
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)
        return ece

    def _plot_coverage_accuracy(
        self,
        conf_xgb: np.ndarray,
        hits_xgb: np.ndarray,
        conf_river: np.ndarray,
        hits_river: np.ndarray,
        conf_ens: np.ndarray,
        hits_ens: np.ndarray,
        path: Path,
    ) -> None:
        """(W5) Coverage vs accuracy curve (sorted by confidence), one curve per model."""
        logger.info(f"[6/7] Rysowanie krzywej pokrycie–dokładność: {path.name}")

        def _curve(conf: np.ndarray, hits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            order = np.argsort(-conf)
            conf_s = conf[order]
            hits_s = hits[order]
            n = len(conf_s)
            if n == 0:
                return np.zeros(0), np.zeros(0)
            coverages = np.arange(1, n + 1, dtype=float) / float(n)
            accuracies = np.cumsum(hits_s) / np.arange(1, n + 1, dtype=float)
            return coverages, accuracies

        cov_x, acc_x = _curve(conf_xgb, hits_xgb)
        cov_r, acc_r = _curve(conf_river, hits_river)
        cov_e, acc_e = _curve(conf_ens, hits_ens)

        fig, ax = plt.subplots(figsize=(7, 5))
        if len(cov_x) > 0:
            ax.plot(cov_x, acc_x, label="XGB")
        if len(cov_r) > 0:
            ax.plot(cov_r, acc_r, label="River")
        if len(cov_e) > 0:
            ax.plot(cov_e, acc_e, label="Ensemble")

        ax.set_xlabel("Pokrycie (odsetek przypadków pokazanych)")
        ax.set_ylabel("Top-1 na pokazanych")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Krzywa pokrycie–dokładność\n(XGB / River / Ensemble)")
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def run(self) -> bool:
        """Run the full validation; returns True if the pass condition is met."""
        logger.info("=== Start walidacji modeli ML (XGB / River / Ensemble) ===")
        seed_everywhere(42)

        self.load_testset()
        assert self.df is not None and self.y_true is not None

        # Full test set by default. VALIDATION_MAX_N enables a quick run on a RANDOM
        # sample with a fixed seed (stratification is unnecessary at large N).
        max_n = int(os.environ.get("VALIDATION_MAX_N", "0") or 0)
        if max_n and len(self.df) > max_n:
            logger.info(f"Tryb szybki: losowa próbka {max_n} z {len(self.df)} rekordów (seed=42).")
            self.df = self.df.sample(n=max_n, random_state=42).reset_index(drop=True)
            self.y_true = self.df["diseases"].astype(str).to_numpy()

        self.load_models()
        assert self.xgb is not None and self.river is not None and self.ens is not None

        preds = self._predict_all()
        y_true = self.y_true
        preds_top3_xgb: List[List[str]] = preds["preds_top3_xgb"]  # type: ignore
        preds_top3_river: List[List[str]] = preds["preds_top3_river"]  # type: ignore
        preds_top3_ens: List[List[str]] = preds["preds_top3_ens"]  # type: ignore
        conf_xgb: np.ndarray = preds["conf_xgb"]  # type: ignore
        conf_river: np.ndarray = preds["conf_river"]  # type: ignore
        conf_ens: np.ndarray = preds["conf_ens"]  # type: ignore

        logger.info("[4/7] Liczenie metryk globalnych (Top-1 / Top-3) i bootstrapu.")

        # global Top-1 / Top-3 metrics with bootstrap CIs (W2)
        hits_xgb_top1 = self._topk_hits(y_true, preds_top3_xgb, k=1)
        hits_xgb_top3 = self._topk_hits(y_true, preds_top3_xgb, k=3)
        hits_river_top1 = self._topk_hits(y_true, preds_top3_river, k=1)
        hits_river_top3 = self._topk_hits(y_true, preds_top3_river, k=3)
        hits_ens_top1 = self._topk_hits(y_true, preds_top3_ens, k=1)
        hits_ens_top3 = self._topk_hits(y_true, preds_top3_ens, k=3)

        metrics: Dict[str, Dict[str, float]] = {}
        for name, h1, h3 in [
            ("xgb", hits_xgb_top1, hits_xgb_top3),
            ("river", hits_river_top1, hits_river_top3),
            ("ens", hits_ens_top1, hits_ens_top3),
        ]:
            logger.info(f"  • Bootstrap ({self.n_bootstrap} prób) dla {name.upper()} (Top-1/Top-3).")
            top1, t1_lo, t1_hi = self._bootstrap_ci(h1, n_boot=self.n_bootstrap)
            top3, t3_lo, t3_hi = self._bootstrap_ci(h3, n_boot=self.n_bootstrap)
            metrics[name] = {
                "top1": top1,
                "top1_lo": t1_lo,
                "top1_hi": t1_hi,
                "top3": top3,
                "top3_lo": t3_lo,
                "top3_hi": t3_hi,
            }

        # naive baseline: always predict the most frequent class
        (unique, counts_arr) = np.unique(y_true, return_counts=True)
        most_idx = int(np.argmax(counts_arr))
        most_class = str(unique[most_idx])
        baseline_top1 = float(counts_arr[most_idx] / len(y_true))
        baseline_top3 = baseline_top1
        logger.info(
            f"[4/7] Baseline (najczęstsza klasa: {most_class}) – Top-1/Top-3={baseline_top1:.3f}."
        )

        # confusion matrices (W1)
        logger.info("[5/7] Liczenie i rysowanie macierzy pomyłek.")

        def top1_from_top3(p: List[List[str]]) -> np.ndarray:
            return np.asarray([row[0] if row else '__none__' for row in p], dtype=object)

        y_pred_xgb = top1_from_top3(preds_top3_xgb)
        y_pred_river = top1_from_top3(preds_top3_river)
        y_pred_ens = top1_from_top3(preds_top3_ens)

        # union of true and predicted labels so every model's matrix has the same axes
        labels_all = sorted(
            set(y_true.tolist())
            | set(y_pred_xgb.tolist())
            | set(y_pred_river.tolist())
            | set(y_pred_ens.tolist())
        )

        cm_xgb_all = confusion_matrix(y_true, y_pred_xgb, labels=labels_all)
        cm_river_all = confusion_matrix(y_true, y_pred_river, labels=labels_all)
        cm_ens_all = confusion_matrix(y_true, y_pred_ens, labels=labels_all)

        # all classes, raw and row-normalized
        self._plot_confusion(
            cm_xgb_all,
            labels_all,
            "Macierz pomyłek – XGB (wszystkie klasy)",
            self.out_dir / "cm_xgb_all.svg",
            normalize_rows=False,
        )
        self._plot_confusion(
            cm_river_all,
            labels_all,
            "Macierz pomyłek – River (wszystkie klasy)",
            self.out_dir / "cm_river_all.svg",
            normalize_rows=False,
        )
        self._plot_confusion(
            cm_ens_all,
            labels_all,
            "Macierz pomyłek – Ensemble (wszystkie klasy)",
            self.out_dir / "cm_ens_all.svg",
            normalize_rows=False,
        )

        self._plot_confusion(
            cm_xgb_all,
            labels_all,
            "Macierz pomyłek – XGB (wszystkie klasy)",
            self.out_dir / "cm_xgb_all_norm.svg",
            normalize_rows=True,
        )
        self._plot_confusion(
            cm_river_all,
            labels_all,
            "Macierz pomyłek – River (wszystkie klasy)",
            self.out_dir / "cm_river_all_norm.svg",
            normalize_rows=True,
        )
        self._plot_confusion(
            cm_ens_all,
            labels_all,
            "Macierz pomyłek – Ensemble (wszystkie klasy)",
            self.out_dir / "cm_ens_all_norm.svg",
            normalize_rows=True,
        )

        # 50 most / least frequent classes by y_true
        label_counts = {str(lbl): int(cnt) for lbl, cnt in zip(unique, counts_arr)}

        sorted_by_freq_desc = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
        labels_top50_freq = [lbl for lbl, _ in sorted_by_freq_desc[:50]]

        sorted_by_freq_asc = sorted(label_counts.items(), key=lambda kv: kv[1])
        labels_top50_rare = [lbl for lbl, _ in sorted_by_freq_asc[:50]]

        if len(labels_top50_freq) > 0:
            cm_xgb_top_freq = confusion_matrix(y_true, y_pred_xgb, labels=labels_top50_freq)
            cm_river_top_freq = confusion_matrix(y_true, y_pred_river, labels=labels_top50_freq)
            cm_ens_top_freq = confusion_matrix(y_true, y_pred_ens, labels=labels_top50_freq)

            self._plot_confusion(
                cm_xgb_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – XGB (50 najczęstszych klas)",
                self.out_dir / "cm_xgb_top50freq.svg",
                normalize_rows=False,
            )
            self._plot_confusion(
                cm_river_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – River (50 najczęstszych klas)",
                self.out_dir / "cm_river_top50freq.svg",
                normalize_rows=False,
            )
            self._plot_confusion(
                cm_ens_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – Ensemble (50 najczęstszych klas)",
                self.out_dir / "cm_ens_top50freq.svg",
                normalize_rows=False,
            )

            self._plot_confusion(
                cm_xgb_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – XGB (50 najczęstszych klas)",
                self.out_dir / "cm_xgb_top50freq_norm.svg",
                normalize_rows=True,
            )
            self._plot_confusion(
                cm_river_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – River (50 najczęstszych klas)",
                self.out_dir / "cm_river_top50freq_norm.svg",
                normalize_rows=True,
            )
            self._plot_confusion(
                cm_ens_top_freq,
                labels_top50_freq,
                "Macierz pomyłek – Ensemble (50 najczęstszych klas)",
                self.out_dir / "cm_ens_top50freq_norm.svg",
                normalize_rows=True,
            )

        if len(labels_top50_rare) > 0:
            cm_xgb_top_rare = confusion_matrix(y_true, y_pred_xgb, labels=labels_top50_rare)
            cm_river_top_rare = confusion_matrix(y_true, y_pred_river, labels=labels_top50_rare)
            cm_ens_top_rare = confusion_matrix(y_true, y_pred_ens, labels=labels_top50_rare)

            self._plot_confusion(
                cm_xgb_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – XGB (50 najrzadszych klas)",
                self.out_dir / "cm_xgb_top50rare.svg",
                normalize_rows=False,
            )
            self._plot_confusion(
                cm_river_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – River (50 najrzadszych klas)",
                self.out_dir / "cm_river_top50rare.svg",
                normalize_rows=False,
            )
            self._plot_confusion(
                cm_ens_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – Ensemble (50 najrzadszych klas)",
                self.out_dir / "cm_ens_top50rare.svg",
                normalize_rows=False,
            )

            self._plot_confusion(
                cm_xgb_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – XGB (50 najrzadszych klas)",
                self.out_dir / "cm_xgb_top50rare_norm.svg",
                normalize_rows=True,
            )
            self._plot_confusion(
                cm_river_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – River (50 najrzadszych klas)",
                self.out_dir / "cm_river_top50rare_norm.svg",
                normalize_rows=True,
            )
            self._plot_confusion(
                cm_ens_top_rare,
                labels_top50_rare,
                "Macierz pomyłek – Ensemble (50 najrzadszych klas)",
                self.out_dir / "cm_ens_top50rare_norm.svg",
                normalize_rows=True,
            )

        self._plot_bars_topk(metrics, self.out_dir / "bars_topk.svg")

        # per-class accuracy for the ensemble (W3)
        logger.info("[5/7] Liczenie trafności per klasa dla Ensemble.")
        acc_ens, counts_ens = self._per_class_acc(y_true, y_pred_ens)

        self._plot_per_class(
            acc_ens,
            counts_ens,
            "Trafność Top-1 per klasa – Ensemble (wszystkie klasy)",
            self.out_dir / "perclass_ens.svg",
            top_k=None,
        )

        self._plot_per_class(
            acc_ens,
            counts_ens,
            "Trafność Top-1 per klasa – Ensemble (50 najczęstszych klas)",
            self.out_dir / "perclass_ens_top50freq.svg",
            top_k=None,
            labels_subset=labels_top50_freq,
            sort_desc=True,
        )

        self._plot_per_class(
            acc_ens,
            counts_ens,
            "Trafność Top-1 per klasa – Ensemble (50 najrzadszych klas)",
            self.out_dir / "perclass_ens_top50rare.svg",
            top_k=None,
            labels_subset=labels_top50_rare,
            sort_desc=False,
        )

        # reliability diagrams and ECE (W4)
        logger.info("[6/7] Liczenie kalibracji (ECE) i rysowanie wykresów.")
        ece_xgb = self._plot_calibration(
            conf_xgb,
            hits_xgb_top1,
            "XGB",
            self.out_dir / "calibration_xgb.svg",
        )
        ece_river = self._plot_calibration(
            conf_river,
            hits_river_top1,
            "River",
            self.out_dir / "calibration_river.svg",
        )
        ece_ens = self._plot_calibration(
            conf_ens,
            hits_ens_top1,
            "Ensemble",
            self.out_dir / "calibration_ens.svg",
        )

        self._plot_coverage_accuracy(
            conf_xgb,
            hits_xgb_top1,
            conf_river,
            hits_river_top1,
            conf_ens,
            hits_ens_top1,
            self.out_dir / "coverage_accuracy_all.svg",
        )

        logger.info("[7/7] Sprawdzanie kryteriów akceptacji K1–K4 i regresji lokalnych.")
        # K1/K2: ensemble significantly better than the BEST single model. Both are scored on
        # the same cases, so the comparison is paired: exact McNemar test (p < 0.05, two-sided)
        # plus direction of improvement. The "95% CIs do not overlap" check is kept for
        # information only (K1_ci / K2_ci); it is known to be too conservative for paired data.
        from scipy.stats import binomtest

        def _mcnemar(h_ens: np.ndarray, h_best: np.ndarray) -> Dict[str, float]:
            e, bst = h_ens.astype(bool), h_best.astype(bool)
            b = int((e & ~bst).sum())   # ensemble correct, best model wrong
            c = int((~e & bst).sum())   # the reverse
            p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            return {"ens_only": b, "best_only": c, "p_value": p}

        best1 = "xgb" if metrics["xgb"]["top1"] >= metrics["river"]["top1"] else "river"
        best3 = "xgb" if metrics["xgb"]["top3"] >= metrics["river"]["top3"] else "river"
        h1 = {"xgb": hits_xgb_top1, "river": hits_river_top1}
        h3 = {"xgb": hits_xgb_top3, "river": hits_river_top3}
        mc1 = _mcnemar(hits_ens_top1, h1[best1])
        mc3 = _mcnemar(hits_ens_top3, h3[best3])
        K1 = bool(metrics["ens"]["top1"] > metrics[best1]["top1"] and mc1["p_value"] < 0.05)
        K2 = bool(metrics["ens"]["top3"] > metrics[best3]["top3"] and mc3["p_value"] < 0.05)
        K1_ci = bool(metrics["ens"]["top1_lo"] > max(metrics["xgb"]["top1_hi"], metrics["river"]["top1_hi"]))
        K2_ci = bool(metrics["ens"]["top3_lo"] > max(metrics["xgb"]["top3_hi"], metrics["river"]["top3_hi"]))

        # off-diagonal errors from the all-classes confusion matrix
        off_xgb = int(cm_xgb_all.sum() - np.trace(cm_xgb_all))
        off_river = int(cm_river_all.sum() - np.trace(cm_river_all))
        off_ens = int(cm_ens_all.sum() - np.trace(cm_ens_all))
        K3 = bool(off_ens <= min(off_xgb, off_river))

        K4 = bool(ece_ens <= max(ece_xgb, ece_river))

        # Local regression: a class where the ensemble loses >10 pp vs the better of the two
        # models AND the drop is significant (one-sided exact McNemar within the class, Holm
        # correction over classes). Without correction, ~570 classes always yield chance
        # "drops". Raw drops >10 pp are still reported in summary.json.
        acc_xgb, _ = self._per_class_acc(y_true, y_pred_xgb)
        acc_river, _ = self._per_class_acc(y_true, y_pred_river)
        hx_c, hr_c, he_c = (y_pred_xgb == y_true), (y_pred_river == y_true), (y_pred_ens == y_true)
        cls_rows: List[Dict[str, Any]] = []
        for cls in sorted(acc_ens):
            m = y_true == cls
            best_h = hx_c[m] if acc_xgb.get(cls, 0.0) >= acc_river.get(cls, 0.0) else hr_c[m]
            drop = float(best_h.mean() - he_c[m].mean())
            lose = int((~he_c[m] & best_h).sum())
            gain = int((he_c[m] & ~best_h).sum())
            p = float(binomtest(lose, lose + gain, 0.5, alternative="greater").pvalue) if lose + gain else 1.0
            cls_rows.append({"class": cls, "n": int(m.sum()), "drop": drop, "p_value": p})
        # Holm step-down correction (monotone adjusted p-values)
        order = sorted(range(len(cls_rows)), key=lambda i: cls_rows[i]["p_value"])
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (len(cls_rows) - rank) * cls_rows[i]["p_value"])
            cls_rows[i]["p_holm"] = min(1.0, running)
        bad_classes = [r for r in cls_rows if r["drop"] > 0.10 and r["p_holm"] < 0.05]
        raw_drops = [r for r in cls_rows if r["drop"] > 0.10]
        no_local_regression = len(bad_classes) == 0

        # ensemble minus best model, paired bootstrap on the same cases
        def _paired_delta(h_a: np.ndarray, h_b: np.ndarray) -> Dict[str, float]:
            rng = np.random.default_rng(42)
            d = h_a - h_b
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(self.n_bootstrap)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}

        paired = {
            f"top1_ens_minus_{best1}": _paired_delta(hits_ens_top1, h1[best1]),
            f"top3_ens_minus_{best3}": _paired_delta(hits_ens_top3, h3[best3]),
        }

        # proper scoring rules on full distributions (NLL, multiclass Brier)
        prob_metrics: Dict[str, Dict[str, float]] = {}
        yi = getattr(self, "_y_idx", None)
        for name, P in (getattr(self, "_probs", None) or {}).items():
            py = np.clip(P[np.arange(len(yi)), yi], 1e-12, 1.0)
            prob_metrics[name] = {
                "nll": float(-np.log(py).mean()),
                "brier": float(((P ** 2).sum(1) - 2 * py + 1).mean()),
            }

        satisfied_main = sum(1 for flag in (K1, K2, K3) if flag)
        passed = bool(satisfied_main >= 2 and no_local_regression)

        # reproducibility metadata
        feats_hash = self._features_hash(self.xgb.features)  # type: ignore
        ensemble_version = {
            "w_xgb": ENSEMBLE_W_XGB,
            "w_river": ENSEMBLE_W_RIVER,
            "k_fuse": ENSEMBLE_K_FUSE,
            "min_w_xgb": ENSEMBLE_MIN_XGB_WEIGHT,
            "min_w_river": ENSEMBLE_MIN_RIVER_WEIGHT,
            "conf_gap": ENSEMBLE_CONF_GAP,
        }
        model_id_xgb = f"{os.path.basename(MODEL_PATH)}@{int(os.path.getmtime(self._resolve_path(MODEL_PATH)))}"
        model_id_river = f"{os.path.basename(RIVER_MODEL_PATH)}@{int(os.path.getmtime(self._resolve_path(RIVER_MODEL_PATH)))}"

        summary = {
            "n_test": int(len(y_true)),
            "baseline": {
                "most_frequent_class": most_class,
                "top1": baseline_top1,
                "top3": baseline_top3,
            },
            "metrics": metrics,
            "ece": {
                "xgb": ece_xgb,
                "river": ece_river,
                "ens": ece_ens,
            },
            "probabilistic": prob_metrics,
            "paired_bootstrap": paired,
            "off_diagonal_errors": {
                "xgb": off_xgb,
                "river": off_river,
                "ens": off_ens,
            },
            "criteria": {
                "K1_top1_ensemble_better": K1,
                "K2_top3_ensemble_better": K2,
                "K1_mcnemar": {"vs": best1, **mc1},
                "K2_mcnemar": {"vs": best3, **mc3},
                "K1_ci_nonoverlap_info": K1_ci,
                "K2_ci_nonoverlap_info": K2_ci,
                "K3_offdiag_reduced": K3,
                "K4_calibration_not_worse": K4,
                "satisfied_main": satisfied_main,
                "no_local_regression_gt_10pp": no_local_regression,
                "bad_classes": bad_classes,
                "raw_drops_gt_10pp": sorted(raw_drops, key=lambda r: r["p_value"])[:50],
                "passed": passed,
            },
            "meta": {
                "test_csv": str(self._resolve_path(TEST_CSV)),
                "model_path_xgb": str(self._resolve_path(MODEL_PATH)),
                "model_path_river": str(self._resolve_path(RIVER_MODEL_PATH)),
                "model_id_xgb": model_id_xgb,
                "model_id_river": model_id_river,
                "features_hash": feats_hash,
                "ensemble_version": ensemble_version,
            },
        }

        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=== ML validation summary ===")
        print(f"Test samples: {len(y_true)}")
        print(
            f"Baseline (most frequent: {most_class}): "
            f"Top-1={baseline_top1:.3f}, Top-3={baseline_top3:.3f}"
        )
        for name in ["xgb", "river", "ens"]:
            m = metrics[name]
            print(
                f"{name.upper():6s} Top-1={m['top1']:.3f} [{m['top1_lo']:.3f}, {m['top1_hi']:.3f}] "
                f"Top-3={m['top3']:.3f} [{m['top3_lo']:.3f}, {m['top3_hi']:.3f}]"
                + (f"  NLL={prob_metrics[name]['nll']:.4f} Brier={prob_metrics[name]['brier']:.4f}"
                   if name in prob_metrics else "")
                + f"  ECE={ {'xgb': ece_xgb, 'river': ece_river, 'ens': ece_ens}[name]:.4f}"
            )
        for key, d in paired.items():
            print(f"{key}: {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] (sparowany bootstrap 95%)")
        print(f"K1: McNemar vs {best1.upper()}: ensemble lepszy w {mc1['ens_only']}, gorszy w {mc1['best_only']} przypadkach, p={mc1['p_value']:.4g}")
        print(f"K2: McNemar vs {best3.upper()}: ensemble lepszy w {mc3['ens_only']}, gorszy w {mc3['best_only']} przypadkach, p={mc3['p_value']:.4g}")
        print(f"(informacyjnie: nienakładające się przedziały ufności K1={K1_ci}, K2={K2_ci})")
        print(
            f"Klasy z istotnym spadkiem >10 p.p. (Holm, p<0.05): {len(bad_classes)}; "
            f"surowe spadki >10 p.p. bez testu (informacyjnie): {len(raw_drops)}"
        )
        print(
            f"K1={K1}, K2={K2}, K3={K3}, K4={K4}, "
            f"main_satisfied={satisfied_main}, no_local_regression={no_local_regression}"
        )
        print(f"Passed = {passed}")
        print(f"Artefakty zapisane w: {self.out_dir}")

        logger.info(
            "=== Koniec walidacji. Wynik: %s ===",
            "ZALICZONE" if passed else "NIEZALICZONE",
        )
        return passed

if __name__ == "__main__":
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        )
        runner = MLValidationRunner(
            n_bootstrap=1000,  # predictions are batched, so a full bootstrap is cheap
            out_dir="reports/ml_validation",
        )
        ok = runner.run()
        raise SystemExit(0 if ok else 1)
