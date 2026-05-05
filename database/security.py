from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict

from jose import jwt, JWTError
from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False

# ——— token JWT ———
def create_access_token(
    subject: str,
    secret_key: str,
    expires_minutes: int = 60 * 24 * 7,  # 7 dni
    algorithm: str = "HS256",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {"sub": subject, "iat": int(now.timestamp())}
    if expires_minutes:
        payload["exp"] = int((now + timedelta(minutes=int(expires_minutes))).timestamp())
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret_key, algorithm=algorithm)

def decode_token(token: str, secret_key: str, algorithms: list[str] = ["HS256"]) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, secret_key, algorithms=algorithms)
    except JWTError:
        return None
