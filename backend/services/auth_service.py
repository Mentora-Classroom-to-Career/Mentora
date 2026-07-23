"""
Auth service: password hashing (passlib/bcrypt), JWT issuing/verification
(python-jose), and the `get_current_user` FastAPI dependency every
protected route uses.

Per the master plan §7.2: this is intentionally per-route dependency
injection, not blanket ASGI middleware, since some routes (login,
register) must stay open.
"""
import random
import string
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import settings
from database.session import get_db
from database.models import User

bearer_scheme = HTTPBearer(auto_error=False)

# Using the `bcrypt` library directly rather than passlib's CryptContext:
# passlib's bcrypt backend detection breaks against bcrypt>=4.1 (a known,
# still-open compatibility issue — passlib probes `bcrypt.__about__`, which
# newer bcrypt releases removed). Calling bcrypt directly sidesteps it.
_BCRYPT_MAX_BYTES = 72  # bcrypt's own hard input limit


def hash_password(plain_password: str) -> str:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.checkpw(pw_bytes, password_hash.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"user_id": user_id, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def generate_reset_code() -> str:
    """6-digit numeric code, matching the frontend's `pattern="[0-9]{6}"` field."""
    return "".join(random.choices(string.digits, k=6))


def _decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload["user_id"]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    The real security gate (per master plan §7.2/§7.3) — every protected
    endpoint depends on this. Next.js middleware.ts is a UX convenience
    layer only, never a substitute for this check.
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = _decode_token(credentials.credentials)
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
