import hashlib
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    prehash = hashlib.sha256(password.encode("utf-8")).digest()
    hashed = bcrypt.hashpw(prehash, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    hashed_bytes = hashed.encode("utf-8")
    digest_prehash = hashlib.sha256(password.encode("utf-8")).digest()
    if bcrypt.checkpw(digest_prehash, hashed_bytes):
        return True

    # Backward compatibility for older accounts hashed using sha256().hexdigest()
    legacy_hex_prehash = hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")
    return bcrypt.checkpw(legacy_hex_prehash, hashed_bytes)


def create_access_token(subject: str, expires_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=expires_minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
