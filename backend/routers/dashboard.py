"""GET /api/dashboard/* — Dashboard page data.

Frontend calls:
    getSystemStats()  → GET /api/dashboard/stats
    getRecentAlerts() → GET /api/dashboard/alerts

Response shapes (from types.ts):
    SystemStats: { totalTransactions, fraudRate, blockedToday, pendingReview,
                   modelAccuracy, approvalBreakdown: { approvedPercent, flaggedPercent, blockedPercent } }
    AlertItem:   [{ id, transactionId, riskScore, decision, timestamp, merchant, amount }]
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from backend.decision_store import get_decisions
from backend.models import AlertItem, ApprovalBreakdown, SystemStats

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_model_accuracy() -> float:
    """Read model accuracy from training_history.json last entry."""
    path = Path("artifacts/global_model/training_history.json")
    if not path.exists():
        return 0.0
    try:
        with open(path) as f:
            data = json.load(f)
        history = data if isinstance(data, list) else data.get("history", data.get("rounds", []))
        if history:
            last = history[-1]
            return last.get("accuracy", 0.0)
    except Exception as e:
        logger.warning("Failed to read training_history.json: %s", e)
    return 0.0


@router.get("/api/dashboard/stats", response_model=SystemStats)
async def dashboard_stats():
    """Dashboard KPI cards + approval breakdown donut.

    All values computed from live data:
      - Pie breakdown from stored decisions
      - Fraud rate from blocked / total
      - Model accuracy from FL training history
    """
    decisions = get_decisions(limit=10000)
    total = len(decisions)

    blocked = sum(1 for d in decisions if d.get("decision") == "BLOCK")
    flagged = sum(1 for d in decisions if d.get("decision") == "FLAG")
    allowed = total - blocked - flagged

    fraud_rate = round((blocked / total) * 100, 1) if total > 0 else 0.0
    p_approved = round((allowed / total) * 100, 1) if total > 0 else 0.0
    p_flagged = round((flagged / total) * 100, 1) if total > 0 else 0.0
    p_blocked = round((blocked / total) * 100, 1) if total > 0 else 0.0

    model_acc = _get_model_accuracy()

    return SystemStats(
        totalTransactions=total,
        fraudRate=fraud_rate,
        blockedToday=blocked,
        pendingReview=flagged,
        modelAccuracy=round(model_acc * 100, 1),
        approvalBreakdown=ApprovalBreakdown(
            approvedPercent=p_approved,
            flaggedPercent=p_flagged,
            blockedPercent=p_blocked,
        ),
    )


@router.get("/api/dashboard/alerts", response_model=list[AlertItem])
async def dashboard_alerts():
    """Return top 100 alerts, newest first."""
    decisions = get_decisions(limit=100, shuffle=False)
    return [AlertItem(**d) for d in decisions]
