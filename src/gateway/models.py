"""models.py — Pydantic request/response schemas for the FL Gateway API."""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=64, example="bank_alpha")
    bank_name: str = Field(..., min_length=3, max_length=128, example="Alpha National Bank")
    password:  str = Field(..., min_length=8, example="SecurePass123!")


class LoginRequest(BaseModel):
    client_id: str = Field(..., example="bank_alpha")
    password:  str = Field(..., example="SecurePass123!")


class TokenResponse(BaseModel):
    access_token:       str
    token_type:         str = "bearer"
    expires_in_minutes: int


# ---------------------------------------------------------------------------
# FL Training — Server → Client (global model download)
# ---------------------------------------------------------------------------

class GlobalWeightsResponse(BaseModel):
    """Encrypted global model weights sent from server to client.

    The client:
        1. Decrypts encrypted_weights using the shared Fernet key.
        2. Loads decrypted weights into their local LiteFraudNet.
        3. Trains locally on their private data.
        4. Re-encrypts updated weights and signs the payload.
        5. POSTs to /fl/submit-update.
    """
    round_num:          int
    model_version:      str
    encrypted_weights:  str = Field(..., description="Fernet-encrypted base64 weight payload")
    message:            str


# ---------------------------------------------------------------------------
# FL Training — Client → Server (weight submission)
# ---------------------------------------------------------------------------

class EncryptedWeightUpdate(BaseModel):
    """Authenticated + encrypted weight update from a bank client.

    Security properties:
        encrypted_weights: Fernet ciphertext — prevents reading weights in transit.
        signature:         HMAC-SHA256 over (client_id + round_num + encrypted_weights)
                           — prevents weight poisoning and replay attacks.
    """
    round_num:          int = Field(..., ge=1, description="Current FL round number")
    model_version:      str = Field(..., example="LiteFraudNet-v1")
    n_samples:          int = Field(..., ge=1, description="Number of local training samples")
    encrypted_weights:  str = Field(..., description="Fernet-encrypted base64 weight payload")
    signature:          str = Field(..., description="HMAC-SHA256 signature for integrity")


class SubmitResponse(BaseModel):
    status:    str
    message:   str
    round_num: int


# ---------------------------------------------------------------------------
# Round tracking
# ---------------------------------------------------------------------------

class RoundStatus(BaseModel):
    current_round:    int
    expected_clients: int
    received_updates: int
    submitted_by:     List[str]
    round_open:       bool
    message:          str


# ---------------------------------------------------------------------------
# Prediction (Testing Phase)
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Risk assessment for a single transaction."""
    fraud_risk_score: float
    predicted_label:  int          # 0 = legitimate, 1 = fraud
    risk_level:       str          # LOW | MEDIUM | HIGH
    threshold_used:   float
    model_version:    str
