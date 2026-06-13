"""auth.py — JWT creation and verification for the FL gateway.

Design
    * HMAC-SHA256 (HS256) — single shared secret.
    * Token payload: { sub: client_id, role: role, exp: <unix timestamp>, jti: token_uuid }
    * Passwords stored as bcrypt hashes via direct bcrypt library.
    * Secret MUST be set via FL_GATEWAY_SECRET environment variable.
      Missing secret causes RuntimeError at module load — the app will not start.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 3D — Strict secret loading: no hardcoded fallback
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.environ.get("FL_GATEWAY_SECRET")  # type: ignore[assignment]
if not SECRET_KEY:
    raise RuntimeError(
        "FL_GATEWAY_SECRET environment variable is not set. "
        "The application cannot start without this secret. "
        "Set it in your .env file or deployment environment."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1-hour tokens

import bcrypt

# Password utilities
def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

# JWT utilities
def create_access_token(
    client_id: str,
    role: str = "fl_client",
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
    token_uuid: str = None,
) -> str:
    """Create a signed JWT for the given client_id and role.
    
    The 'jti' (JWT ID) claim is stored in the tokens table and checked on every
    authenticated request to support immediate token revocation.
    """
    jti = token_uuid or uuid.uuid4().hex
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": client_id, "role": role, "exp": expire, "jti": jti}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("JWT issued for client=%s role=%s jti=%s exp=%s", client_id, role, jti, expire.isoformat())
    return token


def decode_token(token: str) -> dict:
    """Verify token signature/expiry and return the payload claims dict.

    Returns: { "sub": client_id, "role": role, "jti": token_id }
    Raises: JWTError if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        client_id: str = payload.get("sub")
        role: str = payload.get("role", "fl_client")
        jti: str = payload.get("jti", "")
        if not client_id:
            raise JWTError("Missing 'sub' claim")
        return {"sub": client_id, "role": role, "jti": jti}
    except JWTError as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise
