from __future__ import annotations
from typing import Dict, List, Set, Optional, Iterable


class SessionStore:
    """
    Przechowuje w pamięci:
    - historię rozmowy,
    - zebrane objawy (EN, kanon),
    - flagę stop,
    - ostatnią wiadomość asystenta,
    - oczekujący follow-up (Tak/Nie) na konkretny objaw,
    - negowane objawy,
    - historię ostatnich follow-upów (anti-loop),
    - poprzednie Top-3 (dla notki o aktualizacji).
    """

    def __init__(self) -> None:
        # core
        self.histories: Dict[str, List[Dict[str, str]]] = {}
        self.symptoms: Dict[str, Set[str]] = {}
        self.stopped: Dict[str, bool] = {}
        self._last_assistant: Dict[str, str] = {}

        # follow-up / negacje / top3
        self._pending_symptom: Dict[str, Optional[str]] = {}
        self._negated: Dict[str, Set[str]] = {}
        self._followups: Dict[str, List[str]] = {}  # session_id -> lista ostatnich pytań
        self.last_top3: Dict[str, List[str]] = {}

    # === Sesja ===
    def exists(self, session_id: str) -> bool:
        return session_id in self.histories

    def init_session(self, session_id: str) -> None:
        self.histories[session_id] = []
        self.symptoms[session_id] = set()
        self.stopped[session_id] = False
        self._last_assistant[session_id] = ""
        self._pending_symptom[session_id] = None
        self._negated[session_id] = set()
        self._followups[session_id] = []
        self.last_top3[session_id] = []

    def _ensure(self, session_id: str) -> None:
        if not self.exists(session_id):
            self.init_session(session_id)

    def clear_all(self) -> None:
        self.histories.clear()
        self.symptoms.clear()
        self.stopped.clear()
        self._last_assistant.clear()
        self._pending_symptom.clear()
        self._negated.clear()
        self._followups.clear()
        self.last_top3.clear()

    # === Historia ===
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        self._ensure(session_id)
        return self.histories[session_id]

    def get_recent_history(self, session_id: str, max_messages: int = 8) -> List[Dict[str, str]]:
        hist = self.get_history(session_id)
        return hist[-max_messages:]

    # === Objawy ===
    def get_symptoms(self, session_id: str) -> Set[str]:
        """Zwraca *żywy* zbiór objawów (mutacje na nim są widoczne w store)."""
        self._ensure(session_id)
        return self.symptoms[session_id]

    def get_symptoms_list(self, session_id: str) -> List[str]:
        """Posortowana lista (kopii) objawów."""
        return sorted(list(self.get_symptoms(session_id)))

    def add_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        """Dodaj objawy do istniejącego zbioru."""
        self._ensure(session_id)
        if not items:
            return
        norm = [str(s).strip().lower() for s in items if s]
        self.symptoms[session_id].update(norm)

    def remove_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        """Usuń wskazane objawy ze zbioru."""
        self._ensure(session_id)
        if not items:
            return
        cur = self.symptoms[session_id]
        for s in items:
            s = (s or "").strip().lower()
            if s and s in cur:
                cur.remove(s)

    def set_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        """Ustaw dokładnie przekazany zestaw objawów (zastąp obecny)."""
        self._ensure(session_id)
        norm = [str(s).strip().lower() for s in (items or []) if s]
        self.symptoms[session_id] = set(norm)

    def replace_symptoms(self, session_id: str, items: Iterable[str] | None) -> None:
        """Alias do set_symptoms — dla czytelności przy 'replace'."""
        self.set_symptoms(session_id, items)

    def clear_symptoms(self, session_id: str) -> None:
        """Wyczyść wszystkie objawy w sesji."""
        self._ensure(session_id)
        self.symptoms[session_id].clear()

    # === Wiadomości ===
    def append_user(self, session_id: str, content: str) -> None:
        self._ensure(session_id)
        self.histories[session_id].append({"role": "user", "content": content})

    def append_assistant(self, session_id: str, content: str) -> None:
        self._ensure(session_id)
        self.histories[session_id].append({"role": "assistant", "content": content})
        self._last_assistant[session_id] = (content or "").strip()

    def last_assistant(self, session_id: str) -> str:
        self._ensure(session_id)
        return self._last_assistant.get(session_id, "")

    # === Stop ===
    def set_stopped(self, session_id: str, value: bool = True) -> None:
        self._ensure(session_id)
        self.stopped[session_id] = value

    def is_stopped(self, session_id: str) -> bool:
        self._ensure(session_id)
        return self.stopped.get(session_id, False)

    # === Follow-up Tak/Nie ===
    def set_pending_symptom(self, session_id: str, feat_en: Optional[str]) -> None:
        self._ensure(session_id)
        self._pending_symptom[session_id] = feat_en

    def clear_pending_symptom(self, session_id: str) -> None:
        self._ensure(session_id)
        self._pending_symptom[session_id] = None

    def get_pending_symptom(self, session_id: str) -> Optional[str]:
        self._ensure(session_id)
        return self._pending_symptom.get(session_id)

    # === Negacje ===
    def add_negated_symptom(self, session_id: str, feat_en: str) -> None:
        self._ensure(session_id)
        self._negated[session_id].add((feat_en or "").lower())

    def get_negated(self, session_id: str) -> Set[str]:
        self._ensure(session_id)
        return self._negated[session_id]

    # === Anti-loop (ostatnie follow-upy) ===
    def mark_followup_asked(self, session_id: str, feat_en: str, keep_last: int = 5) -> None:
        self._ensure(session_id)
        arr = self._followups[session_id]
        f = (feat_en or "").lower()
        if f:
            arr.append(f)
            # deduplikacja z zachowaniem kolejności + przycięcie
            dedup: List[str] = []
            seen = set()
            for x in reversed(arr):
                if x not in seen:
                    seen.add(x)
                    dedup.append(x)
            arr[:] = list(reversed(dedup))[-keep_last:]

    def get_recent_followups(self, session_id: str, n: int = 3) -> List[str]:
        self._ensure(session_id)
        return self._followups.get(session_id, [])[-n:]

    # === Top-3 ===
    def get_last_top3(self, session_id: str) -> List[str]:
        self._ensure(session_id)
        return self.last_top3.get(session_id, [])

    def set_last_top3(self, session_id: str, names_en: List[str]) -> None:
        self._ensure(session_id)
        self.last_top3[session_id] = list(names_en or [])
