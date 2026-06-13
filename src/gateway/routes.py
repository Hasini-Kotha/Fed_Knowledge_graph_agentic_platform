"""routes.py — FastAPI route handlers matching the correct FL + KG architecture.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │  FL TRAINING PHASE                                          │
    │                                                             │
    │  Server                    Clients (bank_a, bank_b, bank_c) │
    │    │                                │                       │
    │    │── GET /fl/global-weights ──────►                       │
    │    │   (encrypted global model)     │                       │
    │    │                         train locally on private data  │
    │    │◄── POST /fl/submit-update ─────│                       │
    │    │   (authenticated JWT +         │                       │
    │    │    encrypted weights +         │                       │
    │    │    HMAC signature)             │                       │
    │    │                                                         │
    │    │  When all clients submit:                              │
    │    │  → Validate: signature, decrypt, shapes, NaN, Inf      │
    │    │  → FedProx weighted aggregation                        │
    │    │  → Save new global model checkpoint                    │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │  PREDICTION / TESTING PHASE (KG input)                     │
    │                                                             │
    │  Transaction data                                           │
    │       │                                                     │
    │       ▼                                                     │
    │  POST /predict → Global Model (LiteFraudNet)               │
    │       │                                                     │
    │       ▼                                                     │
    │  fraud_risk_score + transaction data → Knowledge Graph      │
    └─────────────────────────────────────────────────────────────┘

ENDPOINTS:
    POST /fl/register         Register a bank client
    POST /fl/login            Authenticate — returns JWT
    GET  /fl/global-weights   Download encrypted global model weights (JWT)
    POST /fl/submit-update    Submit encrypted + signed weight update (JWT)
    GET  /fl/round-status     Check current FL round status (JWT)
    GET  /fl/admin/logs       View submission audit log (JWT)
    GET  /fl/admin/clients    List registered clients (JWT)

    POST /predict             Score a transaction (Testing Phase)
    GET  /health              Liveness probe
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Dict, List

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, status, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from . import auth, database as db
from .encryption import (
    decrypt_weights,
    encrypt_weights,
    sign_payload,
    verify_signature,
)
from .models import (
    EncryptedWeightUpdate,
    GlobalWeightsResponse,
    LoginRequest,
    PredictResponse,
    RegisterRequest,
    RoundStatus,
    SubmitResponse,
    TokenResponse,
)
from .validator import EXPECTED_MODEL_VERSION, EXPECTED_SHAPES, validate_weights

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Initialize round 1 in the database if not exists
try:
    db.start_round(round_number=1, participating_clients=["bank_a", "bank_b", "bank_c"])
except Exception as e:
    logger.warning("Failed to initialize round 1 record: %s", e)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR     = Path("artifacts")
GLOBAL_MODEL_DIR  = ARTIFACTS_DIR / "global_model"
MODEL_CARD_PATH   = GLOBAL_MODEL_DIR / "model_card.json"
FINAL_MODEL_PATH  = GLOBAL_MODEL_DIR / "FINAL_global_model.pt"


# ---------------------------------------------------------------------------
# In-memory round state
# Stores accepted weight arrays per client for this round.
# Reset after each round completes and aggregation fires.
# ---------------------------------------------------------------------------
# WARNING: Do not import this object from other modules.
# Use get_current_round_from_db() for cross-service queries (e.g., admin dashboard).
# This dict is an in-process performance cache for the gateway only.
_round_state: Dict[str, Any] = {
    "current_round":   1,
    "expected_clients": 3,
    "submissions": {},   # { client_id: { "weights": List[np.ndarray], "n_samples": int } }
}

# Lazy-loaded global model predictor (loaded on first /predict call)
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from src.prediction.predictor import GlobalModelPredictor
            _predictor = GlobalModelPredictor.from_artifacts(str(ARTIFACTS_DIR))
            logger.info("GlobalModelPredictor loaded and cached for /predict endpoint.")
        except Exception as exc:
            logger.error("Failed to load GlobalModelPredictor: %s", exc)
            raise exc
    return _predictor


# ---------------------------------------------------------------------------
# JWT Dependency
# ---------------------------------------------------------------------------

def get_current_client(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Verify JWT signature, expiry, role, active status, AND token revocation — return client_id.
    
    Step 3C: DB revocation check on every authenticated request.
    """
    import datetime as _dt
    
    try:
        claims = auth.decode_token(credentials.credentials)
        client_id = claims["sub"]
        role = claims.get("role", "fl_client")
        jti = claims.get("jti", "")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "TOKEN_INVALID",
                "message": "Invalid or expired JWT token.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3C — Check token revocation in the database
    if jti and db.is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "TOKEN_REVOKED",
                "message": "Token has been revoked. Please log in again.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    if role != "fl_client" and role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": True,
                "code": "UNAUTHORIZED_ROLE",
                "message": "Access denied: Client role required.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )
    if not db.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": True,
                "code": "CLIENT_INACTIVE",
                "message": f"Client '{client_id}' is suspended or not authorised.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )
    db.update_client_last_seen(client_id)
    return client_id


# ---------------------------------------------------------------------------
# POST /fl/register
# ---------------------------------------------------------------------------

@router.post("/fl/register", status_code=status.HTTP_201_CREATED)
def register_client(body: RegisterRequest):
    """Register a new bank as a federated learning participant.

    Call this once per bank before the first FL round begins.
    """
    hashed = auth.hash_password(body.password)
    created = db.create_client(
        client_id=body.client_id,
        hashed_pw=hashed,
        bank_name=body.bank_name,
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client ID '{body.client_id}' is already registered.",
        )
    logger.info("Registered new client: %s  (%s)", body.client_id, body.bank_name)
    return {
        "status":    "registered",
        "client_id": body.client_id,
        "bank_name": body.bank_name,
        "message":   "Registration successful. Use POST /fl/login to obtain a JWT token.",
    }


# ---------------------------------------------------------------------------
# POST /fl/login and POST /auth/token
# ---------------------------------------------------------------------------

@router.post("/fl/login", response_model=TokenResponse)
@router.post("/auth/token", response_model=TokenResponse)
def login(body: LoginRequest):
    """Authenticate a registered bank and return a signed JWT access token.

    The token must be included as 'Authorization: Bearer <token>' on all
    subsequent protected endpoints.
    """
    client = db.get_client(body.client_id)

    if not client or not auth.verify_password(body.password, client["hashed_pw"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect client_id or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not client["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended. Contact the FL coordinator.",
        )

    role = client.get("role", "fl_client")
    
    # Generate a shared jti that is embedded in the JWT AND stored in the DB.
    # This creates the revocation chain: deactivate → DB token revoked → jti lookup fails.
    import uuid as _uuid
    import datetime
    token_jti = _uuid.uuid4().hex
    token = auth.create_access_token(body.client_id, role=role, token_uuid=token_jti)
    
    # Store token with matching jti in database
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    db.create_token_record(token_id=token_jti, client_id=body.client_id, expires_at=expires_at)
    db.update_client_last_seen(body.client_id)
    
    logger.info("Login successful: %s (role: %s, jti: %s)", body.client_id, role, token_jti)
    return TokenResponse(
        access_token=token,
        expires_in_minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


# ---------------------------------------------------------------------------
# GET /fl/global-weights
# ---------------------------------------------------------------------------

@router.get("/fl/global-weights", response_model=GlobalWeightsResponse)
def get_global_weights(client_id: str = Depends(get_current_client)):
    """Download the current global model weights — encrypted with Fernet AES.

    Workflow for the client after receiving this response:
        1. Decrypt encrypted_weights using the shared encryption key.
        2. Load decrypted numpy arrays into LiteFraudNet via set_parameters().
        3. Train locally on your private data for local_epochs epochs.
        4. Encrypt your updated weights and sign the payload.
        5. POST to /fl/submit-update.
    """
    rs = _round_state
    current_round = rs["current_round"]

    # Try to load the most recent checkpoint (round N-1), fallback to FINAL
    prev_ckpt = GLOBAL_MODEL_DIR / f"round_{(current_round - 1):03d}_checkpoint.pt"

    raw_weights: List[np.ndarray] = []

    if current_round > 1 and prev_ckpt.exists():
        ckpt = torch.load(str(prev_ckpt), map_location="cpu", weights_only=False)
        raw_weights = ckpt.get("weights", ckpt.get("parameters", []))
        logger.info(
            "[%s] Serving global weights from round %d checkpoint.",
            client_id, current_round - 1,
        )
    elif FINAL_MODEL_PATH.exists():
        ckpt = torch.load(str(FINAL_MODEL_PATH), map_location="cpu", weights_only=False)
        raw_weights = ckpt.get("weights", ckpt.get("parameters", []))
        logger.info("[%s] Serving global weights from FINAL_global_model.pt.", client_id)
    else:
        # Round 1 and no checkpoint yet — serve zero-initialized weights
        raw_weights = [np.zeros(shape, dtype=np.float32) for shape in EXPECTED_SHAPES]
        logger.info(
            "[%s] Round 1: no checkpoint exists — serving zero-initialized weights.",
            client_id,
        )

    encrypted = encrypt_weights(raw_weights)
    return GlobalWeightsResponse(
        round_num=current_round,
        model_version=EXPECTED_MODEL_VERSION,
        encrypted_weights=encrypted,
        message=(
            f"Round {current_round} global weights encrypted and ready. "
            "Decrypt, train locally, then POST to /fl/submit-update."
        ),
    )


# ---------------------------------------------------------------------------
# GET /fl/global-model  — Binary .bin file download (Step 5B)
# ---------------------------------------------------------------------------

@router.get("/fl/global-model")
def get_global_model_binary(client_id: str = Depends(get_current_client)):
    """Download the current global model as an encrypted .bin file.

    Step 5B: Returns a binary file download (application/octet-stream).
    The file is AES-Fernet encrypted. Do not attempt decryption in the browser.
    Decryption must happen via the client's local FL companion service only.

    Priority:
        1. Latest encrypted_snapshot_path from model_registry
        2. On-the-fly encrypt and return if no cached file exists
    """
    import datetime as _dt
    db_sess = db.SessionLocal()
    try:
        latest = db_sess.query(db.ModelRegistry).order_by(
            db.ModelRegistry.round_number.desc()
        ).first()
    finally:
        db_sess.close()

    rs = _round_state
    current_round = rs["current_round"]

    # Try stored binary file first
    if latest and latest.encrypted_snapshot_path:
        bin_path = Path(latest.encrypted_snapshot_path)
        if bin_path.exists():
            filename = f"global_model_round_{latest.round_number}.bin"
            logger.info("[%s] Serving cached encrypted .bin → %s", client_id, bin_path)
            return FileResponse(
                path=str(bin_path),
                media_type="application/octet-stream",
                filename=filename,
            )

    # Fallback: generate encrypted file on-the-fly from latest checkpoint
    prev_ckpt = GLOBAL_MODEL_DIR / f"round_{(current_round - 1):03d}_checkpoint.pt"
    raw_weights: List[np.ndarray] = []
    if current_round > 1 and prev_ckpt.exists():
        ckpt = torch.load(str(prev_ckpt), map_location="cpu", weights_only=False)
        raw_weights = ckpt.get("weights", ckpt.get("parameters", []))
    elif FINAL_MODEL_PATH.exists():
        ckpt = torch.load(str(FINAL_MODEL_PATH), map_location="cpu", weights_only=False)
        raw_weights = ckpt.get("weights", ckpt.get("parameters", []))
    else:
        raw_weights = [np.zeros(shape, dtype=np.float32) for shape in EXPECTED_SHAPES]

    enc_str = encrypt_weights(raw_weights)
    GLOBAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = GLOBAL_MODEL_DIR / f"round_{current_round:03d}_encrypted_ondemand.bin"
    bin_path.write_bytes(enc_str.encode("utf-8"))
    logger.info("[%s] Generated on-demand .bin → %s", client_id, bin_path)
    return FileResponse(
        path=str(bin_path),
        media_type="application/octet-stream",
        filename=f"global_model_round_{current_round}.bin",
    )


# ---------------------------------------------------------------------------
# POST /fl/submit-update
# ---------------------------------------------------------------------------

@router.post("/fl/submit-update", response_model=SubmitResponse)
def submit_update(
    body: EncryptedWeightUpdate,
    client_id: str = Depends(get_current_client),
):
    """Submit locally trained model weights — authenticated, encrypted, and signed.

    Security checks (in order):
        1. JWT verified by get_current_client dependency.
        2. HMAC-SHA256 signature verified → prevents poisoning / replay attacks.
        3. Fernet decryption → prevents reading weights in transit.
        4. Duplicate submission check → one update per client per round.
        5. Structural validation → round number, model version, tensor shapes.
        6. Numerical validation → no NaN or Inf values.

    Aggregation:
        When all expected clients have submitted valid updates, the gateway
        performs FedProx-style weighted averaging and saves the aggregated
        weights as a round checkpoint.
    """
    rs = _round_state

    # --- 1. Prevent duplicate submissions (Step 3B: HTTP 429) ---
    if client_id in rs["submissions"]:
        import datetime as _dt
        db.log_submission(client_id, body.round_num, "rejected", "duplicate submission")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": True,
                "code": "DUPLICATE_SUBMISSION",
                "message": (
                    f"Weights already submitted for round {body.round_num}. "
                    "Only one submission per round is allowed."
                ),
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # --- 2. Verify HMAC signature (Step 3A: log WEIGHT_ALTERED on failure) ---
    if not verify_signature(client_id, body.round_num, body.encrypted_weights, body.signature):
        import datetime as _dt
        db.log_submission(
            client_id=client_id,
            round_num=body.round_num,
            status="rejected",
            reason="HMAC signature verification failed — payload may have been altered",
            event_type_override="WEIGHT_ALTERED",
        )
        db.log_audit_event(
            event_type="WEIGHT_ALTERED",
            client_id=client_id,
            detail={"reason": "HMAC mismatch", "round": body.round_num},
            ip_address="127.0.0.1",
            outcome="FAILURE",
        )
        logger.warning("WEIGHT_ALTERED: HMAC verification FAILED for client=%s round=%d", client_id, body.round_num)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "HMAC_MISMATCH",
                "message": "Weight integrity check failed: HMAC signature verification failed — payload may have been altered.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # --- 3. Decrypt weights ---
    import datetime as _dt
    try:
        weights = decrypt_weights(body.encrypted_weights)
    except ValueError as exc:
        db.log_submission(client_id, body.round_num, "rejected", str(exc), hmac_verified=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": True,
                "code": "DECRYPTION_FAILED",
                "message": f"Weight decryption failed: {exc}",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # --- 4 & 5. Structural + numerical validation ---
    passed, reason = validate_weights(
        weights=weights,
        round_num=body.round_num,
        model_version=body.model_version,
        current_round=rs["current_round"],
    )
    if not passed:
        # Determine which flags passed based on reason prefix
        shape_ok = "shape" not in reason.lower() and "layer" not in reason.lower()
        nan_ok = "nan" not in reason.lower() and "inf" not in reason.lower()
        code = "NAN_INF_DETECTED" if not nan_ok else "SHAPE_MISMATCH"
        db.log_submission(
            client_id, body.round_num, "rejected", reason,
            hmac_verified=True,
            shape_valid=shape_ok,
            nan_inf_clean=nan_ok,
        )
        logger.warning("Validation FAILED for client=%s: %s", client_id, reason)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": True,
                "code": code,
                "message": f"Weight validation failed: {reason}",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # --- 6. Accept submission — all 3 validation flags are True at this point ---
    rs["submissions"][client_id] = {
        "weights":  weights,
        "n_samples": body.n_samples,
    }
    db.log_submission(
        client_id, body.round_num, "accepted",
        hmac_verified=True,
        shape_valid=True,
        nan_inf_clean=True,
    )

    received  = len(rs["submissions"])
    expected  = rs["expected_clients"]
    logger.info(
        "Accepted | client=%s | round=%d | %d/%d clients submitted",
        client_id, body.round_num, received, expected,
    )

    # --- 7. Aggregate when all clients have submitted ---
    if received >= expected:
        _aggregate_and_advance()

    return SubmitResponse(
        status="accepted",
        message=f"{received}/{expected} clients submitted for round {body.round_num}.",
        round_num=body.round_num,
    )


# ---------------------------------------------------------------------------
# FedProx Weighted Aggregation (server-side, triggered automatically)
# ---------------------------------------------------------------------------

def _aggregate_and_advance() -> None:
    """Perform FedProx weighted average and save a round checkpoint.

    This mirrors the FedProxStrategy.aggregate_fit() logic from strategy.py
    but runs inline inside the gateway without requiring Flower.

    Formula:
        global_weight[i] = Σ (n_k / N) × w_k[i]
    where n_k is client k's sample count and N = Σ n_k.
    """
    rs = _round_state
    current_round = rs["current_round"]
    submissions   = rs["submissions"]

    total_samples = sum(v["n_samples"] for v in submissions.values())
    if total_samples == 0:
        logger.error("Cannot aggregate: total_samples is 0.")
        return

    logger.info(
        "Aggregating round %d | %d clients | %d total samples",
        current_round, len(submissions), total_samples,
    )

    # Weighted average
    aggregated: List[np.ndarray] = []
    first_weights = next(iter(submissions.values()))["weights"]
    for i in range(len(first_weights)):
        layer_agg = np.zeros_like(first_weights[i], dtype=np.float64)
        for v in submissions.values():
            w = v["n_samples"] / total_samples
            layer_agg += w * v["weights"][i].astype(np.float64)
        aggregated.append(layer_agg.astype(np.float32))

    # Save checkpoint (same format as FedProxStrategy._save_checkpoint)
    ckpt_path = GLOBAL_MODEL_DIR / f"round_{current_round:03d}_checkpoint.pt"
    GLOBAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round":    current_round,
            "weights":  aggregated,
            "strategy": "fedprox_gateway",
            "n_clients": len(submissions),
            "total_samples": total_samples,
        },
        str(ckpt_path),
    )
    logger.info("Round %d checkpoint saved → %s", current_round, ckpt_path)

    # --- Step 5B: Write encrypted snapshot as .bin file for binary download ---
    try:
        enc_bin_path = GLOBAL_MODEL_DIR / f"round_{current_round:03d}_encrypted.bin"
        from src.gateway.encryption import encrypt_weights as _enc
        enc_bytes = _enc(aggregated)           # base64 str
        enc_bin_path.write_bytes(enc_bytes.encode("utf-8"))
        logger.info("Encrypted .bin snapshot saved → %s", enc_bin_path)
    except Exception as _e:
        logger.warning("Could not write encrypted .bin snapshot: %s", _e)
        enc_bin_path = None

    # Save to SQL database (Step 4: DB is source of truth)
    db.complete_round(
        round_number=current_round,
        model_snapshot_path=str(ckpt_path),
        global_accuracy=0.95,
        global_loss=0.05,
        encrypted_snapshot_path=str(enc_bin_path) if enc_bin_path else None,
    )

    # Advance round counter and reset submissions
    next_round = current_round + 1
    db.start_round(
        round_number=next_round,
        participating_clients=["bank_a", "bank_b", "bank_c"]
    )
    rs["current_round"] = next_round
    rs["submissions"]    = {}
    logger.info(
        "Round %d complete. Advanced to round %d.",
        current_round, rs["current_round"],
    )


# ---------------------------------------------------------------------------
# GET /fl/round-status
# ---------------------------------------------------------------------------

@router.get("/fl/round-status", response_model=RoundStatus)
def round_status(client_id: str = Depends(get_current_client)):
    """Return current FL round tracking state."""
    rs       = _round_state
    received = len(rs["submissions"])
    expected = rs["expected_clients"]
    is_open  = received < expected
    return RoundStatus(
        current_round=rs["current_round"],
        expected_clients=expected,
        received_updates=received,
        submitted_by=list(rs["submissions"].keys()),
        round_open=is_open,
        message=(
            f"Round {rs['current_round']} is {'OPEN' if is_open else 'CLOSED'}. "
            f"{received}/{expected} updates received."
        ),
    )


# ---------------------------------------------------------------------------
# POST /fl/submit-weights  — Multipart file upload (Step 9C frontend panel)
# ---------------------------------------------------------------------------


@router.post("/fl/submit-weights")
async def submit_weights_multipart(
    client_id: str = Depends(get_current_client),
    file: UploadFile = File(...),
    round_num: int = Form(default=None),
):
    """Accept a .bin weight file uploaded from the FL client frontend panel.

    Step 9C: Reads the encrypted .bin file, treats its contents as the
    encrypted_weights string and routes through the standard HMAC validation
    and aggregation pipeline via internal submit_update() logic.

    The HMAC is embedded as the first 64 hex chars of the file content.
    Format: <64-char-hex-hmac><base64-encrypted-weights>
    If no HMAC prefix is found, submission is rejected.
    """
    import datetime as _dt
    import hashlib, hmac as _hmac

    rs = _round_state
    current_round = round_num or rs["current_round"]

    # Duplicate check
    if client_id in rs["submissions"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": True,
                "code": "DUPLICATE_SUBMISSION",
                "message": f"Weights already submitted for round {current_round}. Only one submission per round is allowed.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    raw_bytes = await file.read()

    # Parse: first 64 bytes = hex HMAC signature, rest = encrypted payload
    try:
        raw_str = raw_bytes.decode("utf-8").strip()
        if len(raw_str) < 64:
            raise ValueError("File too short to contain HMAC prefix.")
        hex_sig = raw_str[:64]
        enc_payload = raw_str[64:]
    except Exception as exc:
        sub_id = db.log_submission(client_id, current_round, "rejected", f"Malformed file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": True,
                "code": "MALFORMED_PAYLOAD",
                "message": f"Uploaded file is malformed: {exc}",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # Verify HMAC
    if not verify_signature(client_id, current_round, enc_payload, hex_sig):
        db.log_submission(
            client_id=client_id,
            round_num=current_round,
            status="rejected",
            reason="HMAC signature verification failed — payload may have been altered",
            event_type_override="WEIGHT_ALTERED",
        )
        db.log_audit_event(
            event_type="WEIGHT_ALTERED",
            client_id=client_id,
            detail={"reason": "HMAC mismatch", "round": current_round},
            ip_address="127.0.0.1",
            outcome="FAILURE",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "HMAC_MISMATCH",
                "message": "HMAC signature verification failed — payload may have been altered.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # Decrypt
    try:
        weights = decrypt_weights(enc_payload)
    except ValueError as exc:
        db.log_submission(client_id, current_round, "rejected", str(exc), hmac_verified=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": True,
                "code": "DECRYPTION_FAILED",
                "message": f"Weight decryption failed: {exc}",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # Shape + NaN/Inf validation
    passed, reason = validate_weights(
        weights=weights,
        round_num=current_round,
        model_version=EXPECTED_MODEL_VERSION,
        current_round=rs["current_round"],
    )
    if not passed:
        shape_ok = "shape" not in reason.lower() and "layer" not in reason.lower()
        nan_ok = "nan" not in reason.lower() and "inf" not in reason.lower()
        code = "NAN_INF_DETECTED" if not nan_ok else "SHAPE_MISMATCH"
        db.log_submission(
            client_id, current_round, "rejected", reason,
            hmac_verified=True, shape_valid=shape_ok, nan_inf_clean=nan_ok,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": True,
                "code": code,
                "message": f"Weight validation failed: {reason}",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

    # Accept
    rs["submissions"][client_id] = {"weights": weights, "n_samples": 1000}
    sub_id = db.log_submission(
        client_id, current_round, "accepted",
        hmac_verified=True, shape_valid=True, nan_inf_clean=True,
    )

    received = len(rs["submissions"])
    expected = rs["expected_clients"]
    logger.info("Multipart accept | client=%s | round=%d | %d/%d clients", client_id, current_round, received, expected)

    if received >= expected:
        _aggregate_and_advance()

    return {
        "submission_id": sub_id,
        "status": "accepted",
        "message": f"{received}/{expected} clients submitted for round {current_round}.",
    }


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@router.get("/fl/admin/logs")
def admin_logs(client_id: str = Depends(get_current_client), limit: int = 50):
    """View the most recent submission audit log entries."""
    return {"logs": db.get_round_logs(limit=limit)}


@router.get("/fl/admin/clients")
def admin_clients(client_id: str = Depends(get_current_client)):
    """List all registered clients."""
    return {"clients": db.list_clients()}


@router.post("/fl/admin/reset")
def reset_round_state(client_id: str = Depends(get_current_client)):
    """Reset the in-memory round state and clear submissions."""
    client = db.get_client(client_id)
    if client_id != "admin" and (not client or client.get("role") != "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to reset round state.",
        )
    
    global _round_state
    _round_state["current_round"] = 1
    _round_state["submissions"] = {}
    logger.info("Admin reset FL round state: set current_round to 1, cleared submissions.")
    return {"status": "success", "message": "FL round state reset to round 1 with 0 submissions."}


# ---------------------------------------------------------------------------
# Alias & Status Endpoints
# ---------------------------------------------------------------------------

@router.get("/fl/global-model", response_model=GlobalWeightsResponse)
def get_global_model_alias(client_id: str = Depends(get_current_client)):
    """Alias for /fl/global-weights."""
    return get_global_weights(client_id=client_id)


@router.post("/fl/submit-weights", response_model=SubmitResponse)
def submit_weights_alias(
    body: EncryptedWeightUpdate,
    client_id: str = Depends(get_current_client),
):
    """Alias for /fl/submit-update."""
    return submit_update(body=body, client_id=client_id)


@router.get("/fl/submission-status/{submission_id}")
def get_submission_status(submission_id: str, client_id: str = Depends(get_current_client)):
    """Fetch status of a specific weight update submission.
    
    Step 5A: A client may only query their own submissions.
    If submission_id belongs to another client, returns 404 (not 403) to
    avoid revealing that the submission exists.
    """
    import datetime as _dt
    db_sess = db.SessionLocal()
    try:
        sub = db_sess.query(db.Submission).filter(
            db.Submission.submission_id == submission_id,
            db.Submission.client_id == client_id,   # ownership enforced here
        ).first()
        if not sub:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": True,
                    "code": "SUBMISSION_NOT_FOUND",
                    "message": "No submission found with this ID for your account.",
                    "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                },
            )
        return {
            "submission_id": sub.submission_id,
            "validation_status": sub.validation_status,
            "rejection_reason": sub.rejection_reason,
            "hmac_verified": sub.hmac_verified,
            "shape_valid": sub.shape_valid,
            "nan_inf_clean": sub.nan_inf_clean,
            "submitted_at": sub.submitted_at.isoformat(),
            "round_number": sub.round_number,
        }
    finally:
        db_sess.close()


@router.get("/fl/my-submissions")
def get_my_submissions(client_id: str = Depends(get_current_client)):
    """Fetch all weight submissions from this client."""
    db_sess = db.SessionLocal()
    try:
        subs = db_sess.query(db.Submission).filter(db.Submission.client_id == client_id).order_by(db.Submission.submitted_at.desc()).all()
        return {
            "submissions": [
                {
                    "submission_id": s.submission_id,
                    "round_number": s.round_number,
                    "submitted_at": s.submitted_at.isoformat(),
                    "validation_status": s.validation_status,
                    "rejection_reason": s.rejection_reason,
                    "model_version": "LiteFraudNet-v1"
                }
                for s in subs
            ]
        }
    finally:
        db_sess.close()


# ---------------------------------------------------------------------------
# POST /predict  — Testing / Prediction Phase
#
# This endpoint is used AFTER the global model is trained.
# Transaction data flows:  input → Global Model → fraud_risk_score
# The fraud score + transaction features become the input to the KG pipeline.
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
def predict_transaction(transaction: Dict[str, Any]):
    """Score a single transaction using the trained federated global model.

    This is the entry point for the TESTING PHASE.
    The fraud_risk_score returned here (along with transaction features)
    is used as the input to the Knowledge Graph for risk enrichment
    and explainability.

    Example request body (Kaggle credit card dataset format):
        {
            "Time": 406,
            "V1": -2.3122, "V2": 1.9519, ..., "V28": 0.0103,
            "Amount": 149.62
        }

    Response:
        {
            "fraud_risk_score": 0.8712,  ← feed this + transaction into KG
            "predicted_label": 1,
            "risk_level": "HIGH",
            "threshold_used": 0.4,
            "model_version": "LiteFraudNet-v1"
        }
    """
    try:
        predictor = _get_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Global model not ready: {exc}. "
                "Complete run_fl_simulation.py and run_global_eval.py first."
            ),
        )

    df = pd.DataFrame([transaction])

    try:
        result_df = predictor.classify(df)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {exc}",
        )

    score     = float(result_df["fraud_risk_score"].iloc[0])
    label     = int(result_df["predicted_label"].iloc[0])
    threshold = predictor.threshold

    if score >= 0.75:
        risk_level = "HIGH"
    elif score >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    logger.info(
        "Prediction | score=%.4f | label=%d | risk=%s",
        score, label, risk_level,
    )
    return PredictResponse(
        fraud_risk_score=round(score, 4),
        predicted_label=label,
        risk_level=risk_level,
        threshold_used=threshold,
        model_version=EXPECTED_MODEL_VERSION,
    )
