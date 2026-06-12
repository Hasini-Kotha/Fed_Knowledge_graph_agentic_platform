"""GET /api/decisions — Audit log of all AI decisions.

Frontend call (in lib/api.ts):
    api.fetchDecisions({ decision, minRisk, dateFrom, dateTo })

    → calls apiFetch("/api/decisions", { params: { type, minRisk, dateFrom, dateTo } })
    → which becomes: GET /api/decisions?type=BLOCK&minRisk=0.5

Response — array of DecisionResult:
    [{ id, transactionId, riskScore, confidence, decision, timestamp, merchant, amount, factors }, ...]

Pipeline layers used:
  5. AGENT: All decisions logged via decision_store when scan/batch runs.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Query

from backend.decision_store import get_decisions
from backend.models import DecisionResult, Factor
from backend.routers.scan import _generate_factors

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/decisions", response_model=List[DecisionResult])
async def list_decisions(
    type: Optional[str] = Query(None, description="Filter by decision type: ALLOW, FLAG, BLOCK"),
    minRisk: Optional[float] = Query(None, alias="minRisk", ge=0.0, le=1.0, description="Minimum risk score filter"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
):
    """Return all decisions with optional filters, newest first."""
    raw = get_decisions(decision_type=type, min_risk=minRisk, limit=limit)
    results = []
    for d in raw:
        factors_raw = d.get("factors", [])
        if not factors_raw:
            factors_raw = _generate_factors(
                risk_score=d["riskScore"],
                amount=d.get("amount", 0),
                merchant=d.get("merchant"),
                location=None,
                ip=None,
            )
            d = dict(d)
            d["factors"] = [f.model_dump() for f in factors_raw]
        results.append(DecisionResult(**d))
    return results
