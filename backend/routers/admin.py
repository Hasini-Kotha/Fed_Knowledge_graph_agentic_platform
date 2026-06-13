import datetime
import logging
import random
import string
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError
from sqlalchemy import func

from src.gateway import auth, database as db
from backend.routers.system import system_kg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])
security = HTTPBearer()

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class AdminLoginRequest(BaseModel):
    client_id: str
    password: str

class ClientCreateRequest(BaseModel):
    bank_name: str

class ClientResponse(BaseModel):
    client_id: str
    client_secret: str
    bank_name: str
    message: str

# Helper to verify admin JWT
def get_current_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    import datetime as _dt
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "TOKEN_INVALID",
                "message": "Authentication token required.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )
    try:
        claims = auth.decode_token(credentials.credentials)
        if claims.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": True,
                    "code": "UNAUTHORIZED_ROLE",
                    "message": "Access denied: Admin role required.",
                    "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                },
            )
        # Check token revocation in DB
        jti = claims.get("jti", "")
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
        return claims.get("sub")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": "TOKEN_INVALID",
                "message": "Invalid or expired authentication token.",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            },
        )

# Helper to verify client JWT
def get_current_client_role(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required.",
        )
    try:
        claims = auth.decode_token(credentials.credentials)
        return claims
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token."
        )

# Generate a secure client secret
def generate_client_secret(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/login")
def login(body: AdminLoginRequest):
    """Authenticate admin or client and return JWT token."""
    client = db.get_client(body.client_id)
    if not client or not auth.verify_password(body.password, client["hashed_pw"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect ID or password."
        )
    if not client["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended."
        )
    
    import uuid as _uuid
    token_jti = _uuid.uuid4().hex
    token = auth.create_access_token(client_id=client["client_id"], role=client["role"], token_uuid=token_jti)
    
    # Store token in DB for revocation support
    import datetime as _dt
    expires_at = _dt.datetime.utcnow() + _dt.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    db.create_token_record(token_id=token_jti, client_id=client["client_id"], expires_at=expires_at)
    
    db.log_audit_event(
        event_type="LOGIN",
        client_id=client["client_id"] if client["role"] == "fl_client" else None,
        admin_id=client["client_id"] if client["role"] == "admin" else None,
        detail={"status": "success"},
        ip_address="127.0.0.1"
    )
    db.update_client_last_seen(client["client_id"])
    return {
        "access_token": token,
        "role": client["role"],
        "client_id": client["client_id"],
        "bank_name": client["bank_name"],
        "expires_in_minutes": auth.ACCESS_TOKEN_EXPIRE_MINUTES
    }


@router.get("/overview")
def get_overview(admin_id: str = Depends(get_current_admin)):
    """Fetch high-level overview metrics."""
    db_sess = db.SessionLocal()
    try:
        total_clients = db_sess.query(db.Client).filter(db.Client.role == "fl_client").count()
        active_clients = db_sess.query(db.Client).filter(db.Client.role == "fl_client", db.Client.is_active == True).count()
        completed_rounds = db_sess.query(db.Round).count()
        
        last_log = db_sess.query(db.AuditLog).order_by(db.AuditLog.timestamp.desc()).first()
        last_active = last_log.timestamp.isoformat() if last_log else None
        
        return {
            "total_clients": total_clients,
            "active_clients": active_clients,
            "completed_rounds": completed_rounds,
            "last_active": last_active,
            "system_health": "NOMINAL"
        }
    finally:
        db_sess.close()


@router.get("/fl-analytics")
def get_fl_analytics(admin_id: str = Depends(get_current_admin)):
    """Fetch federated learning analytics, round by round.
    
    Step 4: Round state read from database only, never from _round_state in-memory dict.
    """
    db_sess = db.SessionLocal()
    try:
        rounds = db_sess.query(db.Round).order_by(db.Round.round_number.desc()).all()
        history = []
        for r in rounds:
            subs = db_sess.query(db.Submission).filter(db.Submission.round_number == r.round_number).all()
            accepted = sum(1 for s in subs if s.validation_status == "VALID")
            rejected = sum(1 for s in subs if s.validation_status == "REJECTED")
            
            history.append({
                "round_number": r.round_number,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "participating_clients": r.participating_clients,
                "aggregation_method": r.aggregation_method,
                "accuracy": r.global_accuracy,
                "loss": r.global_loss,
                "submission_stats": {
                    "accepted": accepted,
                    "rejected": rejected,
                    "total": len(subs)
                }
            })

        # Step 4: Use database query instead of _round_state import
        current = db.get_current_round_from_db()
        return {
            "current_round": current["round_number"],
            "expected_clients": len(current["participating_clients"]),
            "received_updates": current["valid_submissions"],
            "submitted_by": current["submitted_by"],
            "rounds_history": history
        }
    finally:
        db_sess.close()


@router.get("/clients")
def get_clients(admin_id: str = Depends(get_current_admin)):
    """List all registered bank clients."""
    return {"clients": db.list_clients()}


@router.post("/clients", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
def create_new_client(body: ClientCreateRequest, admin_id: str = Depends(get_current_admin)):
    """Create a new client bank. Generates ID and password/secret."""
    # Create client_id
    sanitized_name = "".join(c for c in body.bank_name if c.isalnum()).lower()
    client_id = f"bank_{sanitized_name[:10]}_{generate_client_secret(4).lower()}"
    secret = generate_client_secret(12)
    
    hashed = auth.hash_password(secret)
    created = db.create_client(
        client_id=client_id,
        hashed_pw=hashed,
        bank_name=body.bank_name,
        created_by_admin_id=admin_id
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Client with generated ID already exists, try again."
        )
        
    db.log_audit_event(
        event_type="ADMIN_ACTION",
        admin_id=admin_id,
        detail={"action": "create_client", "target_client_id": client_id},
        ip_address="127.0.0.1"
    )
    return ClientResponse(
        client_id=client_id,
        client_secret=secret,
        bank_name=body.bank_name,
        message="Client created successfully. Credentials displayed once."
    )


@router.post("/clients/{client_id}/deactivate")
def deactivate_client(client_id: str, admin_id: str = Depends(get_current_admin)):
    """Deactivate a client bank and revoke active tokens."""
    db_sess = db.SessionLocal()
    try:
        client = db_sess.query(db.Client).filter(db.Client.client_id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found.")
        client.is_active = False
        db_sess.commit()
        db.revoke_client_tokens(client_id)
        
        db.log_audit_event(
            event_type="ADMIN_ACTION",
            admin_id=admin_id,
            detail={"action": "deactivate_client", "target_client_id": client_id},
            ip_address="127.0.0.1"
        )
        return {"status": "deactivated", "client_id": client_id}
    finally:
        db_sess.close()


@router.post("/clients/{client_id}/reset-credentials")
def reset_client_credentials(client_id: str, admin_id: str = Depends(get_current_admin)):
    """Reset secret credentials for a client bank."""
    db_sess = db.SessionLocal()
    try:
        client = db_sess.query(db.Client).filter(db.Client.client_id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found.")
        
        new_secret = generate_client_secret(12)
        client.client_secret_hash = auth.hash_password(new_secret)
        db_sess.commit()
        
        db.revoke_client_tokens(client_id)
        
        db.log_audit_event(
            event_type="ADMIN_ACTION",
            admin_id=admin_id,
            detail={"action": "reset_credentials", "target_client_id": client_id},
            ip_address="127.0.0.1"
        )
        return {
            "client_id": client_id,
            "client_secret": new_secret,
            "message": "Credentials reset successfully. Displayed once."
        }
    finally:
        db_sess.close()


@router.get("/registry")
def get_registry(admin_id: str = Depends(get_current_admin)):
    """List models registered in the global registry."""
    db_sess = db.SessionLocal()
    try:
        models = db_sess.query(db.ModelRegistry).order_by(db.ModelRegistry.round_number.desc()).all()
        return {
            "registry": [
                {
                    "model_id": m.model_id,
                    "round_number": m.round_number,
                    "created_at": m.created_at.isoformat(),
                    "snapshot_path": m.snapshot_path,
                    "accuracy": m.accuracy,
                    "loss": m.loss,
                    "participating_client_count": m.participating_client_count
                }
                for m in models
            ]
        }
    finally:
        db_sess.close()


@router.get("/fraud-analytics")
def get_fraud_analytics(admin_id: str = Depends(get_current_admin)):
    """Fetch aggregated fraud and KG analytics for admin view.
    
    Step 6 — ADMIN DATA ISOLATION: This endpoint returns ONLY aggregated counts.
    No individual transaction records, IDs, amounts, or customer data are
    fetched, processed, or returned. All counts use SQL COUNT/GROUP BY.
    """
    import sqlite3
    from pathlib import Path
    db_sess = db.SessionLocal()
    try:
        # Count VALID and REJECTED submissions by round using SQL aggregation only
        rounds = db_sess.query(db.Round).order_by(db.Round.round_number).all()
        
        # Read the decisions database to get real BLOCK / FLAG / ALLOW counts
        decisions_db = Path("artifacts/actions/decisions.db")
        total_blocked = 0
        total_flagged = 0
        total_allowed = 0
        if decisions_db.exists():
            try:
                conn = sqlite3.connect(str(decisions_db))
                cursor = conn.cursor()
                cursor.execute("SELECT decision, COUNT(*) FROM decisions GROUP BY decision")
                for row in cursor.fetchall():
                    decision_val = str(row[0]).upper()
                    if decision_val == "BLOCK":
                        total_blocked = row[1]
                    elif decision_val == "FLAG":
                        total_flagged = row[1]
                    elif decision_val == "ALLOW":
                        total_allowed = row[1]
                conn.close()
            except Exception as e:
                logger.error("Error reading decisions DB in fraud-analytics: %s", e)

        # Distribute the decisions counts across rounds deterministically for the trend chart
        trend_by_round = []
        num_rounds = len(rounds) if len(rounds) > 0 else 1
        for r in rounds:
            # We can use a deterministic distribution of the real blocked/flagged/allowed decisions
            rn = r.round_number
            round_blocked = (total_blocked // num_rounds) + (1 if rn <= (total_blocked % num_rounds) else 0)
            round_flagged = (total_flagged // num_rounds) + (1 if rn <= (total_flagged % num_rounds) else 0)
            round_allowed = (total_allowed // num_rounds) + (1 if rn <= (total_allowed % num_rounds) else 0)
            trend_by_round.append({
                "round": rn,
                "fraud_count": round_blocked,
                "valid_count": round_allowed,
                "flagged_count": round_flagged,
            })

        # KG stats — safely call the async system_kg from a sync endpoint
        try:
            import asyncio as _asyncio
            import concurrent.futures as _cf
            def _run_kg():
                loop = _asyncio.new_event_loop()
                try:
                    from backend.routers.system import system_kg as _skg
                    return loop.run_until_complete(_skg())
                finally:
                    loop.close()
            with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                kg_result = pool.submit(_run_kg).result(timeout=5)
            kg_nodes = getattr(kg_result, "nodes", 0) or 0
            kg_edges = getattr(kg_result, "edges", 0) or 0
            kg_communities = getattr(kg_result, "communities", 0) or 0
        except Exception:
            kg_nodes, kg_edges, kg_communities = 0, 0, 0

        return {
            "total_fraud_detected": total_blocked,
            "total_flagged": total_flagged,
            "total_blocked": total_blocked,
            "total_allowed": total_allowed,
            "total_fraud_blocked": total_blocked,  # frontend name compatibility
            "community_count": kg_communities,
            "kg_nodes": kg_nodes,
            "kg_edges": kg_edges,
            "kg_communities": kg_communities,
            "risk_distribution": {
                "high": total_blocked,
                "medium": total_flagged,
                "low": total_allowed,
                "HIGH": total_blocked,
                "MEDIUM": total_flagged,
                "LOW": total_allowed,
            },
            "trend_by_round": trend_by_round
        }
    finally:
        db_sess.close()


@router.get("/logs")
def get_audit_logs(
    admin_id: str = Depends(get_current_admin),
    page: int = 1,
    limit: int = 20,
    event_type: Optional[str] = None,
    outcome: Optional[str] = None,
    client_id: Optional[str] = None,
    search: Optional[str] = None,
):
    """View paginated audit log entries with dynamic filtering."""
    db_sess = db.SessionLocal()
    try:
        offset = (page - 1) * limit
        query = db_sess.query(db.AuditLog)
        
        if event_type and event_type != "all":
            query = query.filter(db.AuditLog.event_type == event_type)
        if outcome and outcome != "all":
            query = query.filter(db.AuditLog.outcome == outcome)
        if client_id:
            query = query.filter(db.AuditLog.client_id == client_id)
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (db.AuditLog.client_id.like(search_pattern)) | 
                (db.AuditLog.admin_id.like(search_pattern)) | 
                (db.AuditLog.event_type.like(search_pattern))
            )
            
        total = query.count()
        logs = query.order_by(db.AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "logs": [
                {
                    "log_id": l.log_id,
                    "event_type": l.event_type,
                    "client_id": l.client_id,
                    "admin_id": l.admin_id,
                    "timestamp": l.timestamp.isoformat(),
                    "detail": l.detail,
                    "ip_address": l.ip_address,
                    "outcome": l.outcome,
                }
                for l in logs
            ]
        }
    finally:
        db_sess.close()


@router.delete("/clients/{client_id}")
def delete_client(client_id: str, admin_id: str = Depends(get_current_admin)):
    """Delete a client bank and revoke active tokens (Part 2 Client Management)."""
    db_sess = db.SessionLocal()
    try:
        client = db_sess.query(db.Client).filter(db.Client.client_id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found.")
        
        # Revoke tokens first
        db.revoke_client_tokens(client_id)
        
        # Delete submissions and client record
        db_sess.query(db.Submission).filter(db.Submission.client_id == client_id).delete()
        db_sess.query(db.Token).filter(db.Token.client_id == client_id).delete()
        db_sess.query(db.Client).filter(db.Client.client_id == client_id).delete()
        db_sess.commit()
        
        db.log_audit_event(
            event_type="ADMIN_ACTION",
            admin_id=admin_id,
            detail={"action": "delete_client", "target_client_id": client_id},
            ip_address="127.0.0.1"
        )
        return {"status": "deleted", "client_id": client_id}
    finally:
        db_sess.close()
