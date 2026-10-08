from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Set, Optional, Iterable, Any, Tuple

from config import SESSION_STORE_TTL_SECONDS, SESSION_STORE_MAX_SESSIONS

log = logging.getLogger("symptomchecker.session_store")


class _SessionData:
    __slots__ = (
        "history",
        "symptoms",
        "stopped",
        "last_assistant",
        "pending_symptom",
        "negated",
        "followups",
        "last_top3",
        "last_seen",
        "asked",
        "followup_count",
        "stop_warned",
        "last_pred",
        "owner",
        "extra",
    )

    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []
        self.symptoms: Set[str] = set()
        self.stopped: bool = False
        self.last_assistant: str = ""
        self.pending_symptom: Optional[str] = None
        self.negated: Set[str] = set()
        self.followups: List[str] = []
        self.last_top3: List[str] = []
        self.last_seen: float = time.monotonic()
        # wszystkie zadane pytania uzupełniające (żeby nie pytać drugi raz)
        self.asked: Set[str] = set()
        # ile pytań uzupełniających zadaliśmy w tej sesji
        self.followup_count: int = 0
        # użytkownik raz napisał "to wszystko" przy zbyt małej liczbie objawów
        self.stop_warned: bool = False
        # cache predykcji: (wektor cech, k, wynik)
        self.last_pred: Optional[Tuple[Tuple[int, ...], int, List[Dict[str, Any]]]] = None
        # id zalogowanego użytkownika, który pierwszy użył sesji
        self.owner: Optional[int] = None
        # objawy spoza modelu (np. światłowstręt), etykiety PL: do opisu i oceny pilności
        self.extra: List[str] = []


class SessionStore:
    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_sessions: int | None = None,
    ) -> None:
        self._ttl: int = int(ttl_seconds if ttl_seconds is not None else SESSION_STORE_TTL_SECONDS)
        self._max: int = int(max_sessions if max_sessions is not None else SESSION_STORE_MAX_SESSIONS)
        # OrderedDict, żeby tanio odsuwać najstarsze przy LRU
        self._data: "OrderedDict[str, _SessionData]" = OrderedDict()
        self._lock = threading.RLock()
        self._janitor_thread: threading.Thread | None = None
        self._janitor_stop = threading.Event()

    # ===== Janitor =====
    def start_janitor(self, interval_seconds: int = 300) -> None:
        if self._janitor_thread and self._janitor_thread.is_alive():
            return

        def _loop() -> None:
            while not self._janitor_stop.is_set():
                try:
                    self._sweep_expired()
                except Exception:
                    log.exception("Session janitor sweep error")
                # czekaj z możliwością wczesnego wybudzenia (na shutdown)
                self._janitor_stop.wait(timeout=interval_seconds)

        t = threading.Thread(target=_loop, name="session-store-janitor", daemon=True)
        t.start()
        self._janitor_thread = t
        log.info("SessionStore janitor started (ttl=%ss, max=%s).", self._ttl, self._max)

    def stop_janitor(self) -> None:
        self._janitor_stop.set()
        t = self._janitor_thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._janitor_thread = None

    def _sweep_expired(self) -> int:
        now = time.monotonic()
        removed = 0
        with self._lock:
            stale = [k for k, v in self._data.items() if now - v.last_seen > self._ttl]
            for k in stale:
                self._data.pop(k, None)
                removed += 1
        if removed:
            log.info("SessionStore swept %d expired sessions.", removed)
        return removed

    def _evict_if_needed(self) -> None:
        # zakładamy, że trzymamy lock z zewnątrz
        while len(self._data) > self._max:
            sid, _ = self._data.popitem(last=False)  # FIFO/LRU najstarszy
            log.debug("SessionStore LRU evict: %s", sid)

    # ===== Internal =====
    def _get_or_create(self, session_id: str) -> _SessionData:
        # pod lockiem
        s = self._data.get(session_id)
        if s is None:
            s = _SessionData()
            self._data[session_id] = s
            self._evict_if_needed()
        else:
            # bumpnij na koniec OrderedDict (LRU)
            self._data.move_to_end(session_id)
        s.last_seen = time.monotonic()
        return s

    # ===== API publiczne =====
    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._data

    def init_session(self, session_id: str) -> None:
        with self._lock:
            self._data[session_id] = _SessionData()
            self._evict_if_needed()

    def clear_all(self) -> None:
        with self._lock:
            self._data.clear()

    # === Historia ===
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            s = self._get_or_create(session_id)
            # zwracamy referencję — dla kompatybilności z istniejącym kodem,
            # ale konsumenci nie powinni jej mutować bez locka.
            return s.history

    def get_recent_history(self, session_id: str, max_messages: int = 8) -> List[Dict[str, str]]:
        with self._lock:
            s = self._get_or_create(session_id)
            return list(s.history[-max_messages:]) if max_messages > 0 else list(s.history)

    # === Objawy ===
    def get_symptoms(self, session_id: str) -> Set[str]:
        """Zwraca KOPIĘ zbioru objawów (zmiany tylko przez metody store'a)."""
        with self._lock:
            s = self._get_or_create(session_id)
            return set(s.symptoms)

    def get_symptoms_list(self, session_id: str) -> List[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return sorted(s.symptoms)

    def add_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        if not items:
            return
        with self._lock:
            s = self._get_or_create(session_id)
            for x in items:
                if x:
                    key = str(x).strip().lower()
                    s.symptoms.add(key)
                    s.negated.discard(key)

    def remove_negated_symptom(self, session_id: str, feat_en: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.negated.discard((feat_en or "").strip().lower())

    def remove_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        if not items:
            return
        with self._lock:
            s = self._get_or_create(session_id)
            for x in items:
                key = (x or "").strip().lower()
                if key:
                    s.symptoms.discard(key)

    def set_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.symptoms = {str(x).strip().lower() for x in (items or []) if x}
            s.negated -= s.symptoms  # potwierdzony objaw przestaje być zaprzeczony

    def replace_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        self.set_symptoms(session_id, items)

    def clear_symptoms(self, session_id: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.symptoms.clear()

    # === Wiadomości ===
    def append_user(self, session_id: str, content: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.history.append({"role": "user", "content": content})

    def append_assistant(self, session_id: str, content: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.history.append({"role": "assistant", "content": content})
            s.last_assistant = (content or "").strip()

    def last_assistant(self, session_id: str) -> str:
        with self._lock:
            s = self._get_or_create(session_id)
            return s.last_assistant

    # === Stop ===
    def set_stopped(self, session_id: str, value: bool = True) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.stopped = bool(value)

    def is_stopped(self, session_id: str) -> bool:
        with self._lock:
            s = self._get_or_create(session_id)
            return s.stopped

    # === Follow-up Tak/Nie ===
    def set_pending_symptom(self, session_id: str, feat_en: Optional[str]) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.pending_symptom = feat_en

    def clear_pending_symptom(self, session_id: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.pending_symptom = None

    def get_pending_symptom(self, session_id: str) -> Optional[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return s.pending_symptom

    # === Negacje ===
    def add_negated_symptom(self, session_id: str, feat_en: str) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            key = (feat_en or "").strip().lower()
            if key:
                s.negated.add(key)
                s.symptoms.discard(key)  # zaprzeczony objaw nie może jednocześnie być potwierdzony

    def get_negated(self, session_id: str) -> Set[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return set(s.negated)  # kopia, nie pozwalamy mutować z zewnątrz

    # === Anti-loop (ostatnie follow-upy) ===
    def mark_followup_asked(self, session_id: str, feat_en: str, keep_last: int = 5) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            f = (feat_en or "").strip().lower()
            if not f:
                return
            s.asked.add(f)
            s.followup_count += 1
            arr = s.followups
            arr.append(f)
            # deduplikacja z zachowaniem kolejności + przycięcie
            seen: Set[str] = set()
            dedup: List[str] = []
            for x in reversed(arr):
                if x not in seen:
                    seen.add(x)
                    dedup.append(x)
            s.followups = list(reversed(dedup))[-keep_last:]

    def get_recent_followups(self, session_id: str, n: int = 3) -> List[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return list(s.followups[-n:])

    def get_asked(self, session_id: str) -> Set[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return set(s.asked)

    def get_followup_count(self, session_id: str) -> int:
        with self._lock:
            s = self._get_or_create(session_id)
            return int(s.followup_count)

    def set_stop_warned(self, session_id: str, value: bool = True) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.stop_warned = bool(value)

    def reset_flow(self, session_id: str) -> None:
        """Pacjent zaczyna od nowa (np. wyczyścił objawy): zerujemy stan dopytywania."""
        with self._lock:
            s = self._get_or_create(session_id)
            s.stopped = False
            s.stop_warned = False
            s.followup_count = 0
            s.asked.clear()
            s.followups.clear()
            s.negated.clear()
            s.pending_symptom = None
            s.last_pred = None
            s.extra.clear()

    # === Objawy spoza modelu (services/unmodeled_symptoms.py) ===
    def update_extra(self, session_id: str, present: Iterable[str], negated: Iterable[str] = ()) -> List[str]:
        """Dodaje/usuwa etykiety; zwraca te, które są nowe w tej sesji."""
        with self._lock:
            s = self._get_or_create(session_id)
            new = [x for x in (present or []) if x and x not in s.extra]
            s.extra.extend(new)
            for x in negated or []:
                if x in s.extra:
                    s.extra.remove(x)
            return new

    def get_extra(self, session_id: str) -> List[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return list(s.extra)

    def is_stop_warned(self, session_id: str) -> bool:
        with self._lock:
            s = self._get_or_create(session_id)
            return bool(s.stop_warned)

    # === Cache predykcji ===
    def get_last_pred(self, session_id: str) -> Optional[Tuple[Tuple[int, ...], int, List[Dict[str, Any]]]]:
        with self._lock:
            s = self._get_or_create(session_id)
            return s.last_pred

    def set_last_pred(self, session_id: str, vec: Tuple[int, ...], k: int, top: List[Dict[str, Any]]) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.last_pred = (tuple(vec), int(k), [dict(d) for d in top])

    # === Właściciel sesji ===
    def claim_owner(self, session_id: str, user_id: Optional[int]) -> bool:
        """
        Pierwszy zalogowany użytkownik "przejmuje" sesję. Zwraca False, jeśli
        sesja należy już do INNEGO zalogowanego użytkownika.
        Anonimowe żądania (user_id=None) nie zmieniają właściciela.
        """
        with self._lock:
            s = self._get_or_create(session_id)
            if user_id is None:
                return True
            if s.owner is None:
                s.owner = int(user_id)
                return True
            return s.owner == int(user_id)

    # === Top-3 ===
    def get_last_top3(self, session_id: str) -> List[str]:
        with self._lock:
            s = self._get_or_create(session_id)
            return list(s.last_top3)

    def set_last_top3(self, session_id: str, names_en: List[str]) -> None:
        with self._lock:
            s = self._get_or_create(session_id)
            s.last_top3 = list(names_en or [])

    # === Diagnostyka ===
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sessions": len(self._data),
                "max_sessions": self._max,
                "ttl_seconds": self._ttl,
            }
