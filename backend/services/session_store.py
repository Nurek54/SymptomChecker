from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Set, Optional, Iterable, Any

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
        with self._lock:
            s = self._get_or_create(session_id)
            return s.symptoms

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
                    s.symptoms.add(str(x).strip().lower())

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
