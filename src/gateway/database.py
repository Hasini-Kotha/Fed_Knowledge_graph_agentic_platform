"""database.py — SQLAlchemy ORM configuration and models for FL Gateway database.

Defines the 6 tables:
    1. clients
    2. tokens
    3. rounds
    4. submissions
    5. audit_logs
    6. model_registry
"""

import datetime
import os
import logging
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DB_PATH = Path(os.environ.get("GATEWAY_DB_PATH", "artifacts/gateway/fl_gateway.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# SQLAlchemy Models
# ---------------------------------------------------------------------------

class Client(Base):
    __tablename__ = "clients"
    client_id = Column(String(64), primary_key=True, index=True)
    client_name = Column(String(128), nullable=False)
    client_secret_hash = Column(String(128), nullable=False)
    role = Column(String(32), default="fl_client", nullable=False) # "fl_client" | "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    created_by_admin_id = Column(String(64), nullable=True)
    last_seen = Column(DateTime, nullable=True)


class Token(Base):
    __tablename__ = "tokens"
    token_id = Column(String(128), primary_key=True, index=True)
    client_id = Column(String(64), ForeignKey("clients.client_id"), nullable=False)
    issued_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)


class Round(Base):
    __tablename__ = "rounds"
    round_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    round_number = Column(Integer, unique=True, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    participating_clients = Column(JSON, nullable=False) # JSON array of client IDs
    aggregation_method = Column(String(32), default="fedprox", nullable=False)
    model_snapshot_path = Column(String(256), nullable=True)
    global_accuracy = Column(Float, nullable=True)
    global_loss = Column(Float, nullable=True)


class Submission(Base):
    __tablename__ = "submissions"
    submission_id = Column(String(64), primary_key=True, index=True)
    client_id = Column(String(64), ForeignKey("clients.client_id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    validation_status = Column(String(32), default="PENDING", nullable=False) # PENDING, VALID, REJECTED
    rejection_reason = Column(String(256), nullable=True)
    weight_contribution_score = Column(Float, nullable=True)
    # Security validation flags (Step 1 migration)
    hmac_verified = Column(Boolean, default=False, nullable=False)
    shape_valid = Column(Boolean, default=False, nullable=False)
    nan_inf_clean = Column(Boolean, default=False, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String(64), nullable=False) # LOGIN, UPLOAD, VALIDATION_FAILURE, WEIGHT_ALTERED, ADMIN_ACTION, AGGREGATION_COMPLETE
    client_id = Column(String(64), ForeignKey("clients.client_id"), nullable=True)
    admin_id = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    detail = Column(JSON, nullable=False)
    ip_address = Column(String(64), nullable=False)
    # Outcome field (Step 1 migration)
    outcome = Column(String(16), default="SUCCESS", nullable=False)  # SUCCESS | FAILURE


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    round_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    snapshot_path = Column(String(256), nullable=False)
    encrypted_snapshot_path = Column(String(256), nullable=True)  # Step 1 migration
    accuracy = Column(Float, nullable=False)
    loss = Column(Float, nullable=False)
    participating_client_count = Column(Integer, nullable=False)


# ---------------------------------------------------------------------------
# Database Initialization & Migration (Self-Healing Schema Check)
# ---------------------------------------------------------------------------

def _run_schema_migrations(conn) -> None:
    """Apply ALTER TABLE migrations for columns added after initial schema creation."""
    import sqlite3
    cursor = conn.cursor()
    migrations = [
        # submissions table — security validation flags
        ("submissions", "hmac_verified",   "ALTER TABLE submissions ADD COLUMN hmac_verified INTEGER NOT NULL DEFAULT 0"),
        ("submissions", "shape_valid",     "ALTER TABLE submissions ADD COLUMN shape_valid INTEGER NOT NULL DEFAULT 0"),
        ("submissions", "nan_inf_clean",   "ALTER TABLE submissions ADD COLUMN nan_inf_clean INTEGER NOT NULL DEFAULT 0"),
        # audit_logs table — outcome
        ("audit_logs",  "outcome",         "ALTER TABLE audit_logs ADD COLUMN outcome TEXT NOT NULL DEFAULT 'SUCCESS'"),
        # model_registry — encrypted path
        ("model_registry", "encrypted_snapshot_path", "ALTER TABLE model_registry ADD COLUMN encrypted_snapshot_path TEXT"),
    ]
    for table, col, sql in migrations:
        cursor.execute(f"PRAGMA table_info({table})")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if col not in existing_cols:
            try:
                cursor.execute(sql)
                conn.commit()
                logger.info("Migration applied: %s.%s", table, col)
            except Exception as exc:
                logger.warning("Migration skipped for %s.%s: %s", table, col, exc)
    cursor.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check if database is old format (sqlite3 columns mismatched)
    if DB_PATH.exists():
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT hashed_pw FROM clients LIMIT 1")
            # If "hashed_pw" column is found, it's the old schema. Recreate.
            cursor.close()
            conn.close()
            logger.info("Old DB schema detected. Upgrading database to SQLAlchemy ORM schema.")
            DB_PATH.unlink()
        except Exception:
            cursor.close()
            # Run additive migrations on existing DB
            _run_schema_migrations(conn)
            conn.close()

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised via SQLAlchemy at %s", DB_PATH)
    seed_data()


def seed_data():
    """Seed initial admin and bank client records. Only runs if records don't already exist."""
    # --- 3E: Admin password from environment, never hardcoded ---
    admin_password = os.environ.get("ADMIN_SEED_PASSWORD")
    if not admin_password:
        raise RuntimeError(
            "ADMIN_SEED_PASSWORD environment variable is not set. "
            "Set it before starting the gateway server."
        )

    db_session = SessionLocal()
    try:
        from .auth import hash_password

        # Seed admin (only if not exists)
        admin = db_session.query(Client).filter(Client.client_id == "admin").first()
        if not admin:
            admin_client = Client(
                client_id="admin",
                client_name="System Administrator",
                client_secret_hash=hash_password(admin_password),
                role="admin",
                is_active=True
            )
            db_session.add(admin_client)
            logger.info("Seeded admin user.")

        # Seed bank clients (only if not exists)
        clients_to_seed = [
            {"client_id": "bank_a", "bank_name": "Alpha Bank",  "password": "BankAlpha123!"},
            {"client_id": "bank_b", "bank_name": "Beta Bank",   "password": "BankBeta123!"},
            {"client_id": "bank_c", "bank_name": "Gamma Bank",  "password": "BankGamma123!"},
        ]
        for c in clients_to_seed:
            existing = db_session.query(Client).filter(Client.client_id == c["client_id"]).first()
            if not existing:
                db_client = Client(
                    client_id=c["client_id"],
                    client_name=c["bank_name"],
                    client_secret_hash=hash_password(c["password"]),
                    role="fl_client",
                    is_active=True
                )
                db_session.add(db_client)
                logger.info("Seeded client: %s", c["client_id"])
        db_session.commit()
    except Exception as e:
        logger.error("Failed to seed database: %s", e)
        db_session.rollback()
        raise
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Backward-Compatible Helper Methods
# ---------------------------------------------------------------------------

def create_client(client_id: str, hashed_pw: str, bank_name: str, created_by_admin_id: str = None) -> bool:
    db = SessionLocal()
    try:
        existing = db.query(Client).filter(Client.client_id == client_id).first()
        if existing:
            return False
        new_client = Client(
            client_id=client_id,
            client_name=bank_name,
            client_secret_hash=hashed_pw,
            is_active=True,
            created_by_admin_id=created_by_admin_id
        )
        db.add(new_client)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def get_client(client_id: str) -> dict | None:
    db = SessionLocal()
    try:
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if client:
            return {
                "id": client.client_id,
                "client_id": client.client_id,
                "hashed_pw": client.client_secret_hash,
                "bank_name": client.client_name,
                "allowed": 1 if client.is_active else 0,
                "registered_at": client.created_at.isoformat(),
                "role": client.role,
                "last_seen": client.last_seen.isoformat() if client.last_seen else None
            }
        return None
    finally:
        db.close()


def update_client_last_seen(client_id: str) -> None:
    db = SessionLocal()
    try:
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if client:
            client.last_seen = datetime.datetime.utcnow()
            db.commit()
    except Exception as e:
        logger.error("Failed to update last seen: %s", e)
        db.rollback()
    finally:
        db.close()


def is_allowed(client_id: str) -> bool:
    client = get_client(client_id)
    return bool(client and client["allowed"])


def list_clients() -> list[dict]:
    db = SessionLocal()
    try:
        clients = db.query(Client).all()
        return [
            {
                "id": c.client_id,
                "client_id": c.client_id,
                "bank_name": c.client_name,
                "allowed": 1 if c.is_active else 0,
                "registered_at": c.created_at.isoformat(),
                "role": c.role,
                "last_seen": c.last_seen.isoformat() if c.last_seen else None
            }
            for c in clients
        ]
    finally:
        db.close()


def log_submission(
    client_id: str,
    round_num: int,
    status: str,
    reason: str = "",
    hmac_verified: bool = False,
    shape_valid: bool = False,
    nan_inf_clean: bool = False,
    event_type_override: str = None,
    submission_id: str = None,
) -> str:
    """Log a weight submission with security validation flags. Returns the submission_id."""
    db_session = SessionLocal()
    try:
        import uuid
        sub_id = submission_id or f"sub-{uuid.uuid4().hex[:8]}"
        is_valid = status == "accepted"
        submission = Submission(
            submission_id=sub_id,
            client_id=client_id,
            round_number=round_num,
            validation_status="VALID" if is_valid else "REJECTED",
            rejection_reason=reason if reason else None,
            hmac_verified=hmac_verified,
            shape_valid=shape_valid,
            nan_inf_clean=nan_inf_clean,
        )
        db_session.add(submission)

        # Determine event type and outcome
        if event_type_override:
            ev_type = event_type_override
        elif is_valid:
            ev_type = "UPLOAD"
        else:
            ev_type = "VALIDATION_FAILURE"

        outcome = "SUCCESS" if is_valid else "FAILURE"
        audit_log = AuditLog(
            event_type=ev_type,
            client_id=client_id,
            detail={"round_num": round_num, "reason": reason},
            ip_address="127.0.0.1",
            outcome=outcome,
        )
        db_session.add(audit_log)
        db_session.commit()
        return sub_id
    except Exception as exc:
        logger.error("Failed to log submission: %s", exc)
        db_session.rollback()
        return ""
    finally:
        db_session.close()


def get_round_logs(limit: int = 50) -> list[dict]:
    db = SessionLocal()
    try:
        subs = db.query(Submission).order_by(Submission.submitted_at.desc()).limit(limit).all()
        return [
            {
                "client_id": s.client_id,
                "round_num": s.round_number,
                "status": "accepted" if s.validation_status == "VALID" else "rejected",
                "reason": s.rejection_reason or "",
                "submitted_at": s.submitted_at.isoformat()
            }
            for s in subs
        ]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Token Utilities
# ---------------------------------------------------------------------------

def create_token_record(token_id: str, client_id: str, expires_at: datetime.datetime) -> None:
    db = SessionLocal()
    try:
        tok = Token(
            token_id=token_id,
            client_id=client_id,
            expires_at=expires_at,
            is_revoked=False
        )
        db.add(tok)
        db.commit()
    except Exception as e:
        logger.error("Failed to create token record: %s", e)
        db.rollback()
    finally:
        db.close()


def is_token_revoked(token_id: str) -> bool:
    """Return True if this specific token_id (JWT jti claim) has been revoked.
    
    Returns False if token_id is empty/None (legacy tokens without jti claim
    are not rejected — they rely on is_active check only).
    """
    if not token_id:
        return False
    db = SessionLocal()
    try:
        tok = db.query(Token).filter(Token.token_id == token_id).first()
        if tok is None:
            # Token not in DB — could be a newly issued token not yet stored.
            # Do not block; rely on is_active client check.
            return False
        return bool(tok.is_revoked)
    finally:
        db.close()


def revoke_client_tokens(client_id: str) -> None:
    db = SessionLocal()
    try:
        tokens = db.query(Token).filter(Token.client_id == client_id, Token.is_revoked == False).all()
        for tok in tokens:
            tok.is_revoked = True
        db.commit()
    except Exception as e:
        logger.error("Failed to revoke tokens: %s", e)
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Round Operations
# ---------------------------------------------------------------------------

def start_round(round_number: int, participating_clients: list, aggregation_method: str = "fedprox") -> None:
    db = SessionLocal()
    try:
        existing = db.query(Round).filter(Round.round_number == round_number).first()
        if not existing:
            r = Round(
                round_number=round_number,
                participating_clients=participating_clients,
                aggregation_method=aggregation_method
            )
            db.add(r)
            db.commit()
    except Exception as e:
        logger.error("Failed to start round: %s", e)
        db.rollback()
    finally:
        db.close()


def complete_round(
    round_number: int,
    model_snapshot_path: str,
    global_accuracy: float,
    global_loss: float,
    encrypted_snapshot_path: str = None,
) -> None:
    db = SessionLocal()
    try:
        r = db.query(Round).filter(Round.round_number == round_number).first()
        if r:
            r.completed_at = datetime.datetime.utcnow()
            r.model_snapshot_path = model_snapshot_path
            r.global_accuracy = global_accuracy
            r.global_loss = global_loss
            db.commit()

            # Write to model registry (Step 5B: include encrypted_snapshot_path)
            reg = ModelRegistry(
                round_number=round_number,
                snapshot_path=model_snapshot_path,
                encrypted_snapshot_path=encrypted_snapshot_path,
                accuracy=global_accuracy,
                loss=global_loss,
                participating_client_count=len(r.participating_clients)
            )
            db.add(reg)
            db.commit()
            logger.info(
                "Round %d completed. Registry updated (encrypted_path=%s).",
                round_number, encrypted_snapshot_path
            )
    except Exception as e:
        logger.error("Failed to complete round: %s", e)
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

def log_audit_event(
    event_type: str,
    client_id: str = None,
    admin_id: str = None,
    detail: dict = None,
    ip_address: str = "127.0.0.1",
    outcome: str = "SUCCESS",
) -> None:
    db_session = SessionLocal()
    try:
        log = AuditLog(
            event_type=event_type,
            client_id=client_id,
            admin_id=admin_id,
            detail=detail or {},
            ip_address=ip_address,
            outcome=outcome,
        )
        db_session.add(log)
        db_session.commit()
    except Exception as e:
        logger.error("Failed to log audit event: %s", e)
        db_session.rollback()
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Cross-Service Round Query (Step 4 — replaces _round_state import)
# ---------------------------------------------------------------------------

def get_current_round_from_db() -> dict:
    """Query the database for current FL round state.

    Use this for cross-service calls (e.g. admin dashboard on port 8002
    reading state from the gateway's database on port 8000).
    Never import _round_state from routes.py.
    """
    db_session = SessionLocal()
    try:
        # Find the most recent incomplete round
        current_round = (
            db_session.query(Round)
            .filter(Round.completed_at.is_(None))
            .order_by(Round.round_number.desc())
            .first()
        )
        if not current_round:
            # Fallback: use the most recently completed round
            current_round = (
                db_session.query(Round)
                .order_by(Round.round_number.desc())
                .first()
            )

        if not current_round:
            return {
                "round_number": 1,
                "started_at": None,
                "participating_clients": [],
                "total_submissions": 0,
                "valid_submissions": 0,
                "rejected_submissions": 0,
                "pending_submissions": 0,
                "submitted_by": [],
            }

        rn = current_round.round_number
        subs = db_session.query(Submission).filter(Submission.round_number == rn).all()
        valid_subs = [s for s in subs if s.validation_status == "VALID"]
        rejected_subs = [s for s in subs if s.validation_status == "REJECTED"]
        pending_subs = [s for s in subs if s.validation_status == "PENDING"]

        return {
            "round_number": rn,
            "started_at": current_round.started_at.isoformat() if current_round.started_at else None,
            "participating_clients": current_round.participating_clients or [],
            "total_submissions": len(subs),
            "valid_submissions": len(valid_subs),
            "rejected_submissions": len(rejected_subs),
            "pending_submissions": len(pending_subs),
            "submitted_by": [s.client_id for s in valid_subs],
        }
    finally:
        db_session.close()
