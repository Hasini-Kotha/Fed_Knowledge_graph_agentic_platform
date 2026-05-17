"""POST /api/batch — Batch analyze multiple transactions from CSV data.

Frontend call (in lib/api.ts, commented-out real call):
    apiFetch("/api/batch", { method: "POST", body: JSON.stringify(rows) })

Request body — array of { amount: number, merchant?: string }
    [ { "amount": 125.50, "merchant": "AMAZON.COM" }, ... ]

Response — array of BatchRowResult:
    [{ rowIndex, transactionId, amount, riskScore, confidence, decision, merchant }, ...]

Pipeline layers used:
  1. DATA:   Construct DataFrame from each row → DynamicVectorizer transform
  2. FL:     GlobalModelPredictor.predict() for risk scores
  3. AGENT:  Map scores to business decisions
"""

import logging
import uuid
from typing import List

from fastapi import APIRouter

from backend.models import BatchRowResult
from backend.routers.scan import _compute_risk_score, _decide, _load_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def _generate_tx_id() -> str:
    return f"TX-{uuid.uuid4().hex[:8].upper()}"


@router.post("/api/batch", response_model=List[BatchRowResult])
async def batch_analyze(rows: List[dict]):
    """Run the prediction pipeline on a batch of transactions.

    Each row is processed individually through the same 5-stage pipeline
    as a single scan.  Results are returned in the same order as input.
    """
    _load_pipeline()

    results: List[BatchRowResult] = []
    for i, row in enumerate(rows):
        amount = float(row.get("amount", 0))
        merchant = str(row.get("merchant", "")) if row.get("merchant") else None

        risk_score, confidence = _compute_risk_score(amount, merchant, None, None)
        decision = _decide(risk_score)

        results.append(BatchRowResult(
            rowIndex=i,
            transactionId=_generate_tx_id(),
            amount=amount,
            riskScore=risk_score,
            confidence=confidence,
            decision=decision,
            merchant=merchant or "Unknown",
        ))

    return results
