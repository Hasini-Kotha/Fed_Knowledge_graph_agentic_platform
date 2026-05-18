"""Decision store — persists all scan/batch decisions for the audit log.

File: artifacts/actions/decisions_store.json (JSON array)

Each decision matches the frontend's DecisionResult type:
  { id, transactionId, riskScore, confidence, decision, timestamp, merchant, amount, factors }
"""

import json
import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.models import DecisionResult, Factor

logger = logging.getLogger(__name__)
STORE_PATH = Path("artifacts/actions/decisions_store.json")

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
    Each gets a random amount, merchant, location, and timestamp
    so the dashboard pie chart + alerts table show diverse data.
    """
    now = datetime.now(timezone.utc)
    entries = []
    for i in range(count):
        # Roll the decision with realistic probabilities
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
            risk = round(random.uniform(0.36, 0.70), 4)
            confidence = round(random.uniform(0.80, 0.95), 4)
            amount = round(random.uniform(1000, 12000), 2)
            merchant = random.choice(_MERCHANTS)
            location = random.choice(_LOCATIONS)
        else:
            decision = "ALLOW"
            risk = round(random.uniform(0.01, 0.20), 4)
            confidence = round(random.uniform(0.93, 0.99), 4)
            amount = round(random.uniform(5, 800), 2)
            merchant = random.choice(_MERCHANTS[:16])  # avoid high-risk merchants
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

def _load_all() -> List[dict]:
    if STORE_PATH.exists():
        try:
            with open(STORE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt decision store, resetting: %s", e)
    return []


def _save_all(decisions: List[dict]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(decisions, f, indent=2)


def ensure_seeded():
    """Fill the store with synthetic seed data if it has fewer than 5 entries.

    Called once at startup so the dashboard shows mixed ALLOW/FLAG/BLOCK data
    without requiring the user to manually scan dozens of transactions.
    """
    decisions = _load_all()
    if len(decisions) >= 5:
        return
    logger.info("Decision store has %d entries — seeding with %d synthetic transactions", len(decisions), 95)
    seed = _seed_mixed_transactions(95)
    decisions = seed + decisions
    _save_all(decisions)
    logger.info("Seeded %d transactions (BLOCK=%d, FLAG=%d, ALLOW=%d)",
                95,
                sum(1 for d in seed if d["decision"] == "BLOCK"),
                sum(1 for d in seed if d["decision"] == "FLAG"),
                sum(1 for d in seed if d["decision"] == "ALLOW"))


def add_decision(decision: DecisionResult):
    """Persist a single decision to the store."""
    decisions = _load_all()
    decisions.append(decision.model_dump())
    _save_all(decisions)
    logger.info("Stored decision %s (%s)", decision.transactionId, decision.decision)


def update_decision(transaction_id: str, new_decision: str) -> bool:
    """Override the decision for an existing transaction.

    Returns True if found and updated, False otherwise.
    """
    decisions = _load_all()
    for d in decisions:
        if d.get("transactionId") == transaction_id:
            d["decision"] = new_decision
            _save_all(decisions)
            logger.info("Overrode decision for %s → %s", transaction_id, new_decision)
            return True
    return False


def get_by_transaction_id(transaction_id: str) -> Optional[dict]:
    """Look up a stored decision by transactionId."""
    decisions = _load_all()
    for d in decisions:
        if d.get("transactionId") == transaction_id:
            return d
    return None


def get_decisions(
    decision_type: Optional[str] = None,
    min_risk: Optional[float] = None,
    limit: int = 100,
    shuffle: bool = False,
) -> List[dict]:
    """Retrieve decisions with optional filters, newest first.

    When shuffle=True, results are returned in random order (for the alerts
    table to show a varied mix instead of only the latest type).
    """
    decisions = _load_all()

    if decision_type and decision_type != "all":
        decisions = [d for d in decisions if d.get("decision", "").upper() == decision_type.upper()]

    if min_risk is not None:
        decisions = [d for d in decisions if d.get("riskScore", 0) >= min_risk]

    if shuffle:
        random.shuffle(decisions)
    else:
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)

    return decisions[:limit]


# ─── Auto-seed on first import ─────────────────────────────────────────
ensure_seeded()
