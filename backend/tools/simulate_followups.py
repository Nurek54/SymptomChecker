"""
Symulacja rozmów z pytaniami uzupełniającymi na zbiorze testowym.

Każdy wiersz testowy to "pacjent" z kompletem objawów. Pacjent zaczyna od kilku losowych
objawów (--start), a na każde pytanie "Czy ...?" odpowiada zgodnie z prawdą ("tak"/"nie").
Rozmowa idzie przez prawdziwy ChatService (ta sama logika co w API), bez LLM
(SUMMARY_MODE=template), więc wynik mierzy dokładnie wybór pytań i moment zakończenia.

Uruchomienie (w backend/, przy wytrenowanych modelach):
    python tools/simulate_followups.py --n 200 --start 1 2 3

Raport dla każdej liczby objawów startowych:
    pytania   średnia liczba zadanych pytań
    trafione  odsetek pytań z odpowiedzią "tak"
    top1/top3 trafność końcowej oceny
    bez pytań trafność Top-1/Top-3 z samych objawów startowych (punkt odniesienia)
    komplet   trafność, gdyby model znał wszystkie objawy pacjenta (górna granica)
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.dirname(BACKEND), BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BACKEND)
os.environ["SUMMARY_MODE"] = "template"   # bez LLM: liczy się wybór pytań, nie tekst

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import TEST_CSV  # noqa: E402
from core.state import AppState  # noqa: E402
from models.river_manager import RiverManager  # noqa: E402
from models.xgb_manager import XGBManager  # noqa: E402
from schemas import ChatRequest  # noqa: E402
from services.chat_service import ChatService  # noqa: E402
from services.ensemble_calibration import ensure_ensemble_calibration  # noqa: E402
from services.session_store import SessionStore  # noqa: E402
from services.symptom_matcher import SymptomMatcher  # noqa: E402
from services.symptom_stats import SymptomStats  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("simulate_followups")


def _quiet() -> None:
    # moduły aplikacji ustawiają INFO (log każdej predykcji); w symulacji to tysiące linii
    logging.getLogger().setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("symptomchecker"):
            logging.getLogger(name).setLevel(logging.WARNING)


def build_state() -> AppState:
    xgb = XGBManager()
    xgb.train()
    stats = SymptomStats.load_or_build(xgb.features)
    counts = stats.feature_counts if stats is not None and any(stats.feature_counts.values()) else None
    matcher = SymptomMatcher(xgb.features, device="cpu", feature_counts=counts)
    river = RiverManager()
    river.set_features(xgb.features)
    river.train()
    st = AppState(device="cpu", xgb=xgb, llm=None, matcher=matcher, sessions=SessionStore(),
                  river=river, symptom_stats=stats)
    ensure_ensemble_calibration(st)
    return st


def top3(cs: ChatService, symptoms) -> list:
    dist = cs.ens.predict_distribution([cs._build_feature_vector(sorted(symptoms))])
    return [d for d, _ in sorted(dist.items(), key=lambda x: -x[1])[:3]]


def simulate(cs: ChatService, rows, n_start: int, seed: int, max_turns: int = 15) -> pd.DataFrame:
    rng = random.Random(seed)
    out = []
    for i, (disease, symptoms) in enumerate(rows):
        start = rng.sample(symptoms, n_start)
        sid = f"sim-{n_start}-{i}"
        r = cs.handle_chat(ChatRequest(session_id=sid, user_input=".", client_symptoms=start))
        asked = yes = 0
        while not r.top_diagnoses and asked < max_turns:
            pending = cs.st.sessions.get_pending_symptom(sid)
            if not pending:
                break
            hit = pending in symptoms
            asked += 1
            yes += hit
            r = cs.handle_chat(ChatRequest(session_id=sid, user_input="tak" if hit else "nie"))
        final = [d.diagnosis_en for d in (r.top_diagnoses or [])]
        base, full = top3(cs, start), top3(cs, symptoms)
        out.append(dict(
            asked=asked, yes=yes,
            top1=final[:1] == [disease], top3=disease in final,
            base_top1=base[:1] == [disease], base_top3=disease in base,
            full_top1=full[:1] == [disease], full_top3=disease in full,
        ))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="liczba pacjentów")
    ap.add_argument("--start", type=int, nargs="+", default=[1, 2, 3], help="liczba objawów na start")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    _quiet()
    st = build_state()
    _quiet()
    cs = ChatService(st)
    feats = set(st.xgb.features)

    df = pd.read_csv(TEST_CSV)
    df.columns = [str(c).strip().lower().lstrip("﻿") for c in df.columns]
    fcols = [c for c in df.columns if c != "diseases"]
    df = df.sample(n=min(len(df), args.n * 3), random_state=args.seed)
    X = df[fcols].to_numpy()
    rows = []
    for d, x in zip(df["diseases"], X):
        syms = [fcols[j] for j in np.flatnonzero(x) if fcols[j] in feats]
        if len(syms) >= max(args.start) + 1:
            rows.append((str(d).strip().lower(), syms))
    rows = rows[: args.n]

    print(f"Pacjenci: {len(rows)} (zbiór {TEST_CSV})")
    print(f"{'start':>5} | {'pytania':>7} | {'trafione':>8} | {'top1':>5} | {'top3':>5} | "
          f"{'bez pytań':>11} | {'komplet':>11}")
    for n_start in args.start:
        t = time.time()
        res = simulate(cs, rows, n_start, seed=args.seed + n_start)
        hit_rate = res.yes.sum() / max(1, res.asked.sum())
        print(f"{n_start:>5} | {res.asked.mean():>7.1f} | {hit_rate:>8.0%} | {res.top1.mean():>5.0%} | "
              f"{res.top3.mean():>5.0%} | {res.base_top1.mean():>4.0%} / {res.base_top3.mean():>4.0%} | "
              f"{res.full_top1.mean():>4.0%} / {res.full_top3.mean():>4.0%}   ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
