from __future__ import annotations
import re
import random
from typing import List, Dict, Any
from config import STOP_PHRASES

STOP_PATTERNS = [
    r"\bto wszystko\b",
    r"\bto już wszystko\b",
    r"\bkoniec\b",
    r"\bnie mam (już )?więcej objawów\b",
    r"\bstop\b",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PESEL_RE = re.compile(r"\b\d{11}\b")


def mask_sensitive(msg: str) -> str:
    """Mask e-mail addresses and 11-digit numbers (e.g. PESEL) with ***. Used for logs and R3 tests."""
    masked = EMAIL_RE.sub("***", msg or "")
    masked = PESEL_RE.sub("***", masked)
    return masked


def contains_stop_phrase(text: str) -> bool:
    """
    Whether the patient is done listing symptoms ("to wszystko", "koniec", ...).
    Whole-word match: "stop" must not match "39 stopni", nor "end" match "trend".
    """
    t = (text or "").strip().lower()
    for p in (STOP_PHRASES or []):
        p = (p or "").strip().lower().rstrip(".!")
        if p and re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", t):
            return True
    return any(re.search(p, t) for p in STOP_PATTERNS)


_BAD_INTROS = [
    r"^assistant\s*:?",
    r"^system\s*:?",
    r"^user\s*:?",
    r"^as an ai\b",
    r"^jako model (językowy|sztucznej inteligencji)\b",
]


def sanitize_llm_output(text: str) -> str:
    t = (text or "").strip()
    t = "\n".join(
        ln for ln in t.splitlines()
        if not any(re.search(p, (ln or "").lower()) for p in _BAD_INTROS)
    ).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    # app style: no long dashes in patient-facing text, replace U+2014 with a comma
    t = re.sub(r"^[ \t]*—[ \t]*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]*—[ \t]*", ", ", t)
    t = re.sub(r",\s*([,.;:!?])", r"\1", t)
    return t


def ensure_summary_only(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*#{1,6}\s*.*$", "", t, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_llm_summary(
    text: str,
    diag_names_allowed: List[str],
    symptoms_for_display: List[str]
) -> bool:
    t = _norm(text)
    if any(pat in t for pat in ("### system", "### kontekst", "### instrukcja", "assistant:", "user:")):
        return False
    if diag_names_allowed:
        ok = False
        for name in diag_names_allowed[:3]:
            n = _norm(name)
            if n and n in t:
                ok = True
                break
        if not ok:
            return False
    if symptoms_for_display:
        any_sym = any(_norm(s) and _norm(s) in t for s in symptoms_for_display)
        if not any_sym:
            return False
    return True

def _is_pl(lang: str | None) -> bool:
    """True for Polish language codes: 'pl', 'PL', 'pl-PL', 'pl_PL'; None/'' -> False."""
    if not lang:
        return False
    code = str(lang).strip().lower().replace("_", "-")
    return code == "pl" or code.startswith("pl-")


def join_with_conjunction(items: List[str], conj_pl: str = "i") -> str:
    items = [x for x in (items or []) if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj_pl} " + items[-1]


def seed_everywhere(seed: int = 42) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)


# Progress logging for long operations (training, calibration)
def fmt_duration(seconds: float) -> str:
    """12.3 -> '12s', 125 -> '2m05s', 3725 -> '1h02m05s'."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "?"
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    """
    Log progress of a long loop: percent, steps, rate, elapsed time and ETA.
    Logs every `every_pct` percent but at most once per `min_interval` seconds
    (the final step is always logged).

        p = Progress("River/train", total=len(df), log=log, unit="rows")
        for i, row in enumerate(...):
            ...
            p.update(i + 1)
        p.finish()
    """

    def __init__(self, label: str, total: int, log, every_pct: float = 2.0,
                 min_interval: float = 5.0, unit: str = "") -> None:
        import time as _time
        self._time = _time
        self.label = label
        self.total = max(int(total), 1)
        self.log = log
        self.every = max(float(every_pct), 0.1)
        self.min_interval = float(min_interval)
        self.unit = unit
        self.start = _time.time()
        self._last_pct = -1e9
        self._last_t = 0.0
        self.log.info("[%s] START (%d %s)", self.label, self.total, self.unit or "kroków")

    def update(self, done: int, extra: str = "") -> None:
        now = self._time.time()
        pct = 100.0 * done / self.total
        if done < self.total and (pct - self._last_pct < self.every or now - self._last_t < self.min_interval):
            return
        self._last_pct, self._last_t = pct, now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        self.log.info(
            "[%s] %5.1f%% (%d/%d) %.1f %s/s | minęło %s | zostało ~%s%s",
            self.label, pct, done, self.total, rate, self.unit or "kroków",
            fmt_duration(elapsed), fmt_duration(eta), (" | " + extra) if extra else "",
        )

    def finish(self, extra: str = "") -> float:
        elapsed = self._time.time() - self.start
        self.log.info("[%s] KONIEC w %s%s", self.label, fmt_duration(elapsed), (" | " + extra) if extra else "")
        return elapsed


def calibration_metrics(P, y) -> Dict[str, float]:
    """
    Probability quality for distributions P (N x K) and true class indices y.
    top1/top3: accuracy; nll: log-loss; ece: Expected Calibration Error (15 bins over
    top-1 confidence); brier: multiclass Brier score; mean_conf: mean top-1 confidence.
    """
    import numpy as np
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1.0)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n == 0:
        return {}
    py = P[np.arange(n), y]
    k = min(3, P.shape[1])
    top = np.argpartition(-P, kth=k - 1, axis=1)[:, :k]
    top1 = P.argmax(1)
    conf = P.max(1)
    hit = top1 == y
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return {
        "n": int(n),
        "top1": round(float(hit.mean()), 4),
        "top3": round(float((top == y[:, None]).any(1).mean()), 4),
        "nll": round(float(-np.log(py).mean()), 4),
        "ece": round(float(ece), 4),
        "brier": round(float(((P ** 2).sum(1) - 2 * py + 1).mean()), 4),
        "mean_conf": round(float(conf.mean()), 4),
    }


def _training_report_path() -> str:
    import os
    from config import MODEL_PATH
    return os.path.join(os.path.dirname(MODEL_PATH), "training_report.json")


def update_training_report(section: str, data: Dict[str, Any]) -> None:
    """Write or replace one section (xgb / river / ensemble) in ml/training_report.json."""
    import json, os, datetime
    path = _training_report_path()
    report: Dict[str, Any] = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f) or {}
    except Exception:
        report = {}
    report[section] = {**data, "updated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def log_metrics_table(log, title: str, results: Dict[str, Any]) -> None:
    """Log a metrics table, one row per variant (e.g. raw vs calibrated)."""
    rows = [(k, v) for k, v in results.items() if isinstance(v, dict) and "top1" in v]
    if not rows:
        return
    log.info("[%s] %-22s %7s %7s %7s %7s %7s %9s", title, "wariant", "top1", "top3", "NLL", "ECE", "Brier", "śr.pewn.")
    for name, m in rows:
        log.info("[%s] %-22s %7.4f %7.4f %7.4f %7.4f %7.4f %9.4f", title, name,
                 m["top1"], m["top3"], m["nll"], m["ece"], m["brier"], m["mean_conf"])
