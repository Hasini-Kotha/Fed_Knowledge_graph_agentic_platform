"""Decision store — persists all scan/batch decisions for the audit log.

File: artifacts/actions/decisions_store.json (JSON array)

Each decision matches the frontend's DecisionResult type:
  { id, transactionId, riskScore, confidence, decision, timestamp, merchant, amount, factors }
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from backend.models import DecisionResult, Factor

logger = logging.getLogger(__name__)
STORE_PATH = Path("artifacts/actions/decisions_store.json")


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


def add_decision(decision: DecisionResult):
    """Persist a single decision to the store."""
    decisions = _load_all()
    decisions.append(decision.model_dump())
    _save_all(decisions)
    logger.info("Stored decision %s (%s)", decision.transactionId, decision.decision)


def get_decisions(
    decision_type: Optional[str] = None,
    min_risk: Optional[float] = None,
    limit: int = 100,
) -> List[dict]:
    """Retrieve decisions with optional filters, newest first."""
    decisions = _load_all()

    if decision_type and decision_type != "all":
        decisions = [d for d in decisions if d.get("decision", "").upper() == decision_type.upper()]

    if min_risk is not None:
        decisions = [d for d in decisions if d.get("riskScore", 0) >= min_risk]

    decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    return decisions[:limit]
