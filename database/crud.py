from __future__ import annotations
from sqlalchemy import func, select, delete, and_
from datetime import date, datetime
from typing import Optional, Iterable, List
from sqlalchemy.orm import Session

from .models import User, Conversation, Message, Extraction, TriageResult
from .security import hash_password, verify_password

# Title helpers
_GENERIC_TITLES = {
    "rozmowa", "nowa rozmowa", "czat", "chat", "conversation", "new chat",
}

def _join_pl(items: List[str], limit: int = 3) -> str:
    items = [str(x).strip() for x in (items or []) if str(x or "").strip()]
    items = items[: max(1, int(limit))]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " i " + items[-1]

def _clean_user_text_for_title(text: str, max_words: int = 8) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    import re
    t = re.sub(r"`{1,3}.*?`{1,3}", " ", t, flags=re.DOTALL)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[\r\n]+", " ", t)
    GREET = r"^(?:\s*(?:cześć|hej|dzień dobry|dzien dobry|dobry wieczór|witam|halo|siema|proszę o pomoc|mam pytanie|czy mogę|czy moge|poproszę|poprosze)\W+)+"
    t = re.sub(GREET, "", t, flags=re.IGNORECASE)
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    else:
        t = " ".join(words)
    t = t.strip(" .,:;!?-–—")
    if not t:
        return ""
    return t[0].upper() + t[1:]

def _looks_generic(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    if t in _GENERIC_TITLES:
        return True
    return len(t) < 4

def _normalize_title(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    if not s:
        return "Rozmowa"
    if s.endswith("."):
        s = s[:-1]
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s

# Users
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == (email or "").lower().strip()).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)

def get_user_by_provider(db: Session, provider: str, sub: str) -> Optional[User]:
    return db.query(User).filter(
        and_(User.auth_provider == provider, User.provider_sub == sub)
    ).first()

def generate_username(db: Session, base: str) -> str:
    import re
    base = (base or "").strip()
    if "@" in base:
        base = base.split("@", 1)[0]
    base = re.sub(r"[^a-zA-Z0-9_.-]", "", base) or "user"
    candidate = base[:50]
    if not db.query(User).filter(User.username == candidate).first():
        return candidate
    i = 1
    while True:
        cand = f"{base[:45]}-{i}"
        if not db.query(User).filter(User.username == cand).first():
            return cand
        i += 1

def create_user(
    db: Session,
    username: str,
    email: str,
    password: Optional[str],
    *,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    date_of_birth: Optional[date] = None,
    auth_provider: str = "local",
    provider_sub: Optional[str] = None,
    **_ignore,  # extra provider fields (e.g. email_verified, avatar_url) are ignored
) -> User:
    """Create a user. For OAuth accounts pass password=None (password_hash stays NULL)."""
    u = User(
        username=username.strip(),
        email=email.lower().strip(),
        password_hash=(hash_password(password) if password else None),
        first_name=(first_name or None),
        last_name=(last_name or None),
        date_of_birth=(date_of_birth or None),
        auth_provider=auth_provider,
        provider_sub=provider_sub,
        last_login_at=datetime.utcnow(),
    )
    db.add(u); db.flush()
    return u

def check_password(u: User, raw: str) -> bool:
    if not u.password_hash:
        return False
    return verify_password(raw, u.password_hash)

def touch_login(db: Session, user: User) -> None:
    user.last_login_at = datetime.utcnow()
    db.add(user); db.flush()

def update_user_profile(
    db: Session,
    user: User,
    *,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    date_of_birth: Optional[date] = None,
) -> User:
    if username is not None and username.strip() and username.strip() != user.username:
        exists = db.query(User).filter(User.username == username.strip()).first()
        if exists and exists.id != user.id:
            raise ValueError("Nazwa użytkownika jest już zajęta.")
        user.username = username.strip()

    if first_name is not None:
        user.first_name = first_name.strip() or None
    if last_name is not None:
        user.last_name = last_name.strip() or None
    if date_of_birth is not None:
        user.date_of_birth = date_of_birth

    db.add(user); db.flush()
    return user

# Conversations and messages
def get_or_create_conversation(db: Session, user_id: int, session_id: str, title_hint: str) -> Conversation:
    c = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.session_id == session_id
    ).first()
    if c:
        return c
    c = Conversation(user_id=user_id, session_id=session_id, title=(title_hint or "Rozmowa"))
    db.add(c); db.flush()
    return c

def add_message(db: Session, conv: Conversation, role: str, content: str) -> Message:
    m = Message(conversation_id=conv.id, role=role, content=content)
    db.add(m)
    conv.last_message_at = func.now()
    conv.updated_at = func.now()
    db.flush()
    return m

def list_conversations_for_user(db: Session, user_id: int, limit: int = 100) -> list[tuple[Conversation, int]]:
    stmt = (
        select(Conversation, func.count(Message.id).label("cnt"))
        .join(Message, Message.conversation_id == Conversation.id, isouter=True)
        .where(Conversation.user_id == user_id)
        .group_by(Conversation.id)
        .order_by(Conversation.last_message_at.desc())
        .limit(limit)
    )
    return [(row[0], int(row[1])) for row in db.execute(stmt).all()]

def get_conversation_with_messages(db: Session, user_id: int, conv_id: int) -> tuple[Conversation | None, list[Message]]:
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user_id).first()
    if not conv:
        return None, []
    msgs = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.created_at.asc()).all()
    return conv, msgs

def delete_conversation(db: Session, user_id: int, conv_id: int) -> bool:
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user_id).first()
    if not conv:
        return False
    db.delete(conv)
    return True

# Smart titling
def suggest_conversation_title(db: Session, conv_id: int) -> str:
    conv = db.get(Conversation, conv_id)
    if not conv:
        return "Rozmowa"

    top1 = (
        db.query(TriageResult)
        .filter(TriageResult.conversation_id == conv_id)
        .order_by(TriageResult.rank.asc().nulls_last(), TriageResult.id.asc())
        .first()
    )

    syms = (
        db.query(Extraction)
        .filter(Extraction.conversation_id == conv_id)
        .order_by(Extraction.id.asc())
        .limit(6)
        .all()
    )
    sym_texts = []
    seen = set()
    for s in syms or []:
        t = (s.text or "").strip()
        if t and t.lower() not in seen:
            sym_texts.append(t)
            seen.add(t.lower())

    first_user = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id, Message.role == "user")
        .order_by(Message.created_at.asc())
        .first()
    )
    first_text = _clean_user_text_for_title(first_user.content if first_user else "")

    if top1 and sym_texts:
        cand = f"{_normalize_title(top1.label)} – {_join_pl(sym_texts, limit=3)}"
        return _normalize_title(cand)
    if top1:
        cand = f"{_normalize_title(top1.label)} – wstępna ocena"
        return _normalize_title(cand)
    if sym_texts:
        cand = f"{_join_pl(sym_texts, limit=3)} – konsultacja objawów"
        return _normalize_title(cand)
    if first_text:
        return _normalize_title(first_text)
    return "Rozmowa"

def smart_update_conversation_title(db: Session, conv_id: int, *, force: bool = False) -> str:
    conv = db.get(Conversation, conv_id)
    if not conv:
        return "Rozmowa"

    current = (conv.title or "").strip()
    candidate = suggest_conversation_title(db, conv_id)
    if not candidate:
        return current or "Rozmowa"

    cur_norm = current.strip().lower()
    cand_norm = candidate.strip().lower()

    should = force or _looks_generic(current) or (
        ("–" in candidate and "–" not in current)
        or (len(current) < 24 and len(candidate) > len(current))
    )

    if should and cur_norm != cand_norm:
        conv.title = _normalize_title(candidate)
        conv.updated_at = func.now()
        db.add(conv); db.flush()
        return conv.title

    return current or "Rozmowa"

# Extracted symptoms and triage
def replace_extractions(db: Session, conv_id: int, items: Iterable[dict]) -> None:
    db.execute(delete(Extraction).where(Extraction.conversation_id == conv_id))
    for it in items or []:
        text_pl_first = (
            it.get("pl") or it.get("text_pl") or it.get("name_pl") or it.get("symptom_pl")
            or it.get("text") or it.get("name") or it.get("label") or it.get("symptom") or ""
        )
        db.add(Extraction(
            conversation_id=conv_id,
            text=str(text_pl_first).strip()[:255],
            canonical=(it.get("canonical") or it.get("lemma") or it.get("en") or it.get("slug") or None),
            lang=(it.get("lang") or "pl"),
            score=(float(it.get("score")) if isinstance(it.get("score"), (int, float)) else None),
        ))

def replace_triage(db: Session, conv_id: int, items: Iterable[dict]) -> None:
    db.execute(delete(TriageResult).where(TriageResult.conversation_id == conv_id))
    rank_auto = 1
    for it in items or []:
        label = None
        score = None
        rank = None

        if isinstance(it, dict):
            label = (
                it.get("label")
                or it.get("diagnosis_pl")
                or it.get("name_pl")
                or it.get("disease_pl")
                or it.get("diagnosis_en")
                or it.get("name")
                or it.get("disease")
            )
            score = it.get("score")
            if score is None:
                score = it.get("probability")
            rank = it.get("rank") or it.get("order")
        else:
            label = str(it)

        label = (str(label).strip() if label is not None else "")
        if not label:
            continue

        db.add(
            TriageResult(
                conversation_id=conv_id,
                label=label[:255],
                score=(float(score) if isinstance(score, (int, float)) else None),
                rank=int(rank) if isinstance(rank, (int, float)) else rank_auto,
            )
        )
        rank_auto += 1

def get_summary_for_conversation(db: Session, conv_id: int) -> tuple[list[Extraction], list[TriageResult]]:
    syms = db.query(Extraction).filter(Extraction.conversation_id == conv_id).order_by(Extraction.id.asc()).all()
    dias = (
        db.query(TriageResult)
        .filter(TriageResult.conversation_id == conv_id)
        .order_by(TriageResult.rank.asc().nulls_last(), TriageResult.id.asc())
        .all()
    )
    return syms, dias
