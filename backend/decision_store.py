"""Decision store — persists all scan/batch decisions for the audit log.

Replaced JSON file persistence with SQLite for optimal read/write performance.
Database: artifacts/actions/decisions.db

Each decision matches the frontend's DecisionResult type:
  { id, transactionId, riskScore, confidence, decision, timestamp, merchant, amount, factors }
"""

import json
import logging
import random
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.models import DecisionResult, Factor
import os

logger = logging.getLogger(__name__)
DB_PATH = Path(os.environ.get("DECISIONS_DB_PATH", "artifacts/actions/decisions.db"))
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection, creating tables if needed."""
    global _INITIALIZED
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _LOCAL.conn = conn

    if not _INITIALIZED:
        with _INIT_LOCK:
            if not _INITIALIZED:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS decisions (
                        id              TEXT PRIMARY KEY,
                        transaction_id  TEXT NOT NULL UNIQUE,
                        risk_score      REAL NOT NULL,
                        confidence      REAL NOT NULL,
                        decision        TEXT NOT NULL,
                        timestamp       TEXT NOT NULL,
                        merchant        TEXT NOT NULL,
                        amount          REAL NOT NULL,
                        factors         TEXT NOT NULL DEFAULT '[]'
                    );
                    CREATE INDEX IF NOT EXISTS idx_decisions_decision
                        ON decisions(decision);
                    CREATE INDEX IF NOT EXISTS idx_decisions_risk_score
                        ON decisions(risk_score);
                    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
                        ON decisions(timestamp DESC);
                """)
                conn.commit()
                _INITIALIZED = True

    return conn


# ─── Realistic seed data generation ────────────────────────────────────

_MERCHANTS = [
    "AMAZON.COM", "WALMART.COM", "TARGET INC", "BEST BUY", "UBER TRIPS",
    "DOORDASH", "NETFLIX INC", "SHELL OIL", "EXXONMOBIL", "COSTCO WHL",
    "APPLE STORE", "GOOGLE ADS", "MICROSOFT 365", "COMCAST BILL", "VERIZON WRLS",
    "CRYPTO TRADE", "WIRE TRANSFER", "MONEYGRAM INT", "PAYPAL HOLDINGS", "STRIPE INC",
]
_LOCATIONS = ["US", "US", "US", "US", "GB", "DE", "CA", "AU", "IN", "SG", "BR", "FR", "JP", "HK", "NG", "RU"]


def _seed_mixed_transactions(count: int = 95) -> List[dict]:
    """Generate synthetic transactions with realistic risk distribution.

    Distribution target: ~2% BLOCK, ~3% FLAG, ~95% ALLOW
    """
    now = datetime.now(timezone.utc)
    entries = []
    for i in range(count):
        r = random.random()
        if r < 0.02:
            decision = "BLOCK"
            risk = round(random.uniform(0.71, 0.95), 4)
            confidence = round(random.uniform(0.85, 0.98), 4)
            amount = round(random.uniform(5000, 25000), 2)
            merchant = random.choice(["CRYPTO TRADE", "WIRE TRANSFER", "MONEYGRAM INT", "PAYPAL HOLDINGS"])
            location = random.choices(_LOCATIONS, weights=[1]*10 + [3]*6)[0]
        elif r < 0.05:
            decision = "FLAG"
            risk = round(random.uniform(0.30, 0.69), 4)
            confidence = round(random.uniform(0.80, 0.95), 4)
            amount = round(random.uniform(1000, 12000), 2)
            merchant = random.choice(_MERCHANTS)
            location = random.choice(_LOCATIONS)
        else:
            decision = "ALLOW"
            risk = round(random.uniform(0.01, 0.20), 4)
            confidence = round(random.uniform(0.93, 0.99), 4)
            amount = round(random.uniform(5, 800), 2)
            merchant = random.choice(_MERCHANTS[:16])
            location = random.choices(_LOCATIONS, weights=[10]*4 + [1]*12)[0]

        timestamp = now.replace(second=0, microsecond=0).isoformat()

        factors = [
            Factor(name="Transaction Amount", contribution=round(random.uniform(0.1, 0.4), 4),
                   description=f"Amount ${amount:.2f} assessed"),
            Factor(name="Merchant Category", contribution=round(random.uniform(0.05, 0.3), 4),
                   description=f"Merchant '{merchant}' risk profile"),
            Factor(name="Location Risk", contribution=round(random.uniform(0.02, 0.2), 4),
                   description=f"Origin '{location}' risk assessment"),
            Factor(name="IP Reputation", contribution=round(random.uniform(0.01, 0.15), 4),
                   description="IP-based risk evaluation"),
            Factor(name="Velocity (24h)", contribution=round(random.uniform(0.01, 0.15), 4),
                   description="Transaction frequency check"),
        ]

        entries.append(DecisionResult(
            id=f"dec-{uuid.uuid4().hex[:8]}",
            transactionId=f"TX-{uuid.uuid4().hex[:8].upper()}",
            riskScore=risk,
            confidence=confidence,
            decision=decision,
            timestamp=timestamp,
            merchant=merchant,
            amount=amount,
            factors=factors,
        ).model_dump())

    return entries


# ─── Store operations ──────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to the DecisionResult dict format."""
    return {
        "id": row["id"],
        "transactionId": row["transaction_id"],
        "riskScore": row["risk_score"],
        "confidence": row["confidence"],
        "decision": row["decision"],
        "timestamp": row["timestamp"],
        "merchant": row["merchant"],
        "amount": row["amount"],
        "factors": json.loads(row["factors"]),
    }


def ensure_seeded():
    """Fill the store with synthetic seed data if it has fewer than 5 entries."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    if count >= 5:
        return

    logger.info("Decision store has %d entries — seeding with %d synthetic transactions", count, 95)
    seed = _seed_mixed_transactions(95)

    rows = []
    for d in seed:
        rows.append((
            d["id"], d["transactionId"], d["riskScore"], d["confidence"],
            d["decision"], d["timestamp"], d["merchant"], d["amount"],
            json.dumps([f.model_dump() if isinstance(f, Factor) else f for f in d["factors"]]),
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO decisions
            (id, transaction_id, risk_score, confidence, decision, timestamp, merchant, amount, factors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    blocked = sum(1 for d in seed if d["decision"] == "BLOCK")
    flagged = sum(1 for d in seed if d["decision"] == "FLAG")
    allowed = sum(1 for d in seed if d["decision"] == "ALLOW")
    logger.info("Seeded %d transactions (BLOCK=%d, FLAG=%d, ALLOW=%d)", 95, blocked, flagged, allowed)


def add_decision(decision: DecisionResult):
    """Persist a single decision to the store."""
    conn = _get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO decisions
            (id, transaction_id, risk_score, confidence, decision, timestamp, merchant, amount, factors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        decision.id,
        decision.transactionId,
        decision.riskScore,
        decision.confidence,
        decision.decision,
        decision.timestamp,
        decision.merchant,
        decision.amount,
        json.dumps([f.model_dump() for f in decision.factors]),
    ))
    conn.commit()
    logger.info("Stored decision %s (%s)", decision.transactionId, decision.decision)


def update_decision(transaction_id: str, new_decision: str) -> bool:
    """Override the decision for an existing transaction.

    Also refreshes the timestamp so the transaction surfaces to the top
    of the recent-alerts list on the dashboard.

    Returns True if found and updated, False otherwise.
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE decisions SET decision = ?, timestamp = ? WHERE transaction_id = ?",
        (new_decision, now, transaction_id),
    )
    conn.commit()
    updated = cursor.rowcount > 0
    if updated:
        logger.info("Overrode decision for %s → %s", transaction_id, new_decision)
    return updated


def get_by_transaction_id(transaction_id: str) -> Optional[dict]:
    """Look up a stored decision by transactionId."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM decisions WHERE transaction_id = ?",
        (transaction_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_decisions(
    decision_type: Optional[str] = None,
    min_risk: Optional[float] = None,
    limit: int = 100,
    shuffle: bool = False,
) -> List[dict]:
    """Retrieve decisions with optional filters.

    When shuffle=True, results are returned in random order (for the alerts
    table to show a varied mix instead of only the latest type).
    """
    conn = _get_connection()
    conditions = []
    params = []

    if decision_type and decision_type != "all":
        conditions.append("decision = ?")
        params.append(decision_type.upper())

    if min_risk is not None:
        conditions.append("risk_score >= ?")
        params.append(min_risk)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    if shuffle:
        order = "ORDER BY RANDOM()"
    else:
        order = "ORDER BY timestamp DESC"

    query = f"SELECT * FROM decisions {where} {order} LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def clear_decisions() -> int:
    """Delete all decisions from the store and re-seed. Returns count of deleted rows."""
    global _INITIALIZED
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    conn.execute("DROP TABLE IF EXISTS decisions")
    conn.commit()
    _INITIALIZED = False
    _get_connection()  # recreates table
    ensure_seeded()
    logger.info("Cleared %d decisions and re-seeded", count)
    return count


# ─── Auto-seed on first import ─────────────────────────────────────────
ensure_seeded()
