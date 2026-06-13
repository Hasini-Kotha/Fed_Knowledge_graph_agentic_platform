"""encryption.py — Two-layer protection for model weights in transit.

Layer 1 — Fernet Encryption (AES-128-CBC + HMAC-SHA256):
    Prevents anyone intercepting the network from reading model weights.
    The encryption key is deterministically derived from FL_GATEWAY_SECRET
    using PBKDF2-SHA256, so it is stable across server restarts.

Layer 2 — HMAC-SHA256 Payload Signature:
    Covers (client_id + round_num + encrypted_payload).
    Prevents a man-in-the-middle from:
      - Substituting another client's weights
      - Replaying an old round's weights
      - Injecting poisoned weights into the aggregation

Usage:
    # Server → Client (send global weights)
    payload  = encrypt_weights(list_of_ndarrays)

    # Client → Server (submit update)
    payload   = encrypt_weights(trained_weights)
    signature = sign_payload(client_id, round_num, payload)
    # POST { encrypted_weights: payload, signature: signature, ... }

    # Server on receipt
    ok = verify_signature(client_id, round_num, payload, signature)
    weights = decrypt_weights(payload)
"""

import base64
import hashlib
import hmac as _hmac
import logging
import os
import pickle
from typing import List

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 3D — Strict secret loading: no hardcoded fallback
# ---------------------------------------------------------------------------
_SECRET: str = os.getenv("FL_GATEWAY_SECRET")  # type: ignore[assignment]
if not _SECRET:
    raise RuntimeError(
        "FL_GATEWAY_SECRET environment variable is not set. "
        "The application cannot start without this secret."
    )

# Step 3F — Static salt documentation
# SECURITY NOTE: This salt is static and intentional for cross-session key
# consistency in the current FL demo. In production, use a per-client or
# per-round salt stored in the database, and derive separate keys per session.
# A static salt means key derivation security depends entirely on the strength
# of FL_GATEWAY_SECRET.
_PBKDF2_SALT = b"fl-gateway-enc-salt-v1"
logger.warning(
    "SECURITY: PBKDF2 salt is static. Key security relies entirely on "
    "FL_GATEWAY_SECRET strength. Use per-session salts in production."
)

_fernet_instance: Fernet | None = None


def _get_fernet() -> Fernet:
    """Return a cached Fernet instance derived from the gateway secret."""
    global _fernet_instance
    if _fernet_instance is None:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_PBKDF2_SALT,
            iterations=100_000,
        )
        raw_key = kdf.derive(_SECRET.encode())
        fernet_key = base64.urlsafe_b64encode(raw_key)
        _fernet_instance = Fernet(fernet_key)
        logger.info("Fernet encryption engine initialised.")
    return _fernet_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encrypt_weights(weights: List[np.ndarray]) -> str:
    """Serialize and encrypt model weight arrays.

    Args:
        weights: List of numpy float32 arrays (model parameters).

    Returns:
        Base64url-encoded Fernet ciphertext string safe for JSON transport.
    """
    serialized = pickle.dumps([w.astype(np.float32) for w in weights])
    ciphertext = _get_fernet().encrypt(serialized)
    payload = base64.urlsafe_b64encode(ciphertext).decode()
    logger.debug("Encrypted %d weight tensors → %d chars", len(weights), len(payload))
    return payload


def decrypt_weights(payload: str) -> List[np.ndarray]:
    """Decrypt and deserialize model weight arrays.

    Args:
        payload: Base64url-encoded Fernet ciphertext string.

    Returns:
        List of numpy float32 arrays.

    Raises:
        ValueError: if decryption fails (wrong key, tampered payload).
    """
    try:
        ciphertext = base64.urlsafe_b64decode(payload.encode())
        serialized = _get_fernet().decrypt(ciphertext)
        weights = pickle.loads(serialized)
        logger.debug("Decrypted %d weight tensors.", len(weights))
        return weights
    except InvalidToken:
        raise ValueError(
            "Weight decryption failed: invalid token. "
            "Possible causes: wrong secret key, tampered payload, or expired token."
        )
    except Exception as exc:
        raise ValueError(f"Weight decryption error: {exc}")


def sign_payload(client_id: str, round_num: int, encrypted_payload: str) -> str:
    """Generate an HMAC-SHA256 signature that binds the encrypted payload
    to a specific client and round number.

    This prevents:
    - Impersonation (client_a submitting as client_b)
    - Replay attacks (reusing a previous round's valid submission)
    - Weight substitution (replacing payload with poisoned weights)

    Args:
        client_id:         Registered client identifier.
        round_num:         Current FL round number.
        encrypted_payload: The Fernet-encrypted weight string.

    Returns:
        Hex-encoded HMAC-SHA256 digest (64 characters).
    """
    message = f"{client_id}:{round_num}:{encrypted_payload}".encode()
    digest = _hmac.new(_SECRET.encode(), message, hashlib.sha256).hexdigest()
    return digest


def verify_signature(
    client_id: str, round_num: int, encrypted_payload: str, signature: str
) -> bool:
    """Verify an HMAC-SHA256 signature using constant-time comparison.

    Returns:
        True if signature is valid, False otherwise.
    """
    expected = sign_payload(client_id, round_num, encrypted_payload)
    return _hmac.compare_digest(expected, signature)
