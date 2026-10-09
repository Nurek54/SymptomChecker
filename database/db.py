# Supabase Postgres only, no fallback database.
from __future__ import annotations
import os
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

# Root .env with override=False: system/CI env vars win over .env (same as config.py).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

DB_URL = os.getenv("DATABASE_URL", "").strip()
if not DB_URL:
    raise RuntimeError(
        "Brak DATABASE_URL w .env. Użyj Supabase pooled URL, np.:\n"
        "postgresql+psycopg://postgres.<ref>:<haslo>@aws-1-<region>.pooler.supabase.com:5432/postgres?sslmode=require"
    )

# Accept only postgresql:// or postgresql+psycopg:// on a Supabase host; force the psycopg
# driver and default sslmode=require.
u = urlparse(DB_URL)

if u.scheme not in ("postgresql", "postgresql+psycopg"):
    raise RuntimeError("DATABASE_URL musi mieć schemat 'postgresql' lub 'postgresql+psycopg' (Supabase).")

host = (u.hostname or "").lower()
if ".supabase." not in host:
    raise RuntimeError("DATABASE_URL nie wskazuje na Supabase (host nie zawiera '.supabase.').")

scheme = "postgresql+psycopg"
qs = dict(parse_qsl(u.query))
qs.setdefault("sslmode", "require")

DB_URL_NORMALIZED = urlunparse((scheme, u.netloc, u.path, "", urlencode(qs), ""))

engine = create_engine(
    DB_URL_NORMALIZED,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,
    pool_timeout=30,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
