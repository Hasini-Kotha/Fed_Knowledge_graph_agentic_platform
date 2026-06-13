"""POST /api/batch — Batch analyze multiple transactions.

Uses the trained ML model via GlobalModelPredictor for all scoring.
No hardcoded rule-based thresholds — the model drives every risk score.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.decision_store import add_decision
from backend.models import BatchRowResult, DecisionResult
import backend.routers.scan as _scan_module

logger = logging.getLogger(__name__)
router = APIRouter()

# Columns expected by the ML model
_FEATURE_COLS = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                 "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                 "V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
_HELP_COLS_ML = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                 "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                 "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"}


def _generate_tx_id() -> str:
    return f"TX-{uuid.uuid4().hex[:8].upper()}"


def _get_str(row: dict, *keys: str) -> str:
    """Read a string value from row dict, trying multiple keys."""
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            return str(v) if not isinstance(v, (int, float)) else str(v)
    return ""


def _get_float(row: dict, *keys: str, default: float = 0.0) -> float:
    """Read a float value from row dict, trying multiple keys."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return default


def _has_full_features(rows: List[dict]) -> bool:
    """Check if the first row contains V1-V28 feature columns."""
    if not rows:
        return False
    return _HELP_COLS_ML.issubset(rows[0].keys())


# ─── ML Model Path ─────────────────────────────────────────────────────

def _compute_confidence(risk: float) -> float:
    """Confidence decreases as risk approaches 0.5 (maximum uncertainty)."""
    c = 0.97 - 0.20 * (1 - abs(risk - 0.5) * 2)
    return max(0.75, min(0.99, c))


def _batch_predict_ml(rows: List[dict]) -> List[BatchRowResult]:
    """Run the trained ML model on all rows at once (batched inference).

    All risk scores come from the model. No hardcoded rule fallback.
    """
    predictor = _scan_module._predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="The latest LiteFraudNet global model is not loaded. Please verify that a model checkpoint exists in the registry."
        )

    records = []
    for row in rows:
        rec = {"Time": 0, "Class": 0}
        for col in _FEATURE_COLS:
            if col == "Amount":
                rec[col] = _get_float(row, "Amount", "amount")
            else:
                rec[col] = _get_float(row, col)
        records.append(rec)

    df = pd.DataFrame(records)
    result_df = predictor.predict(df)
    scores = result_df["fraud_risk_score"].values

    results: List[BatchRowResult] = []
    for i, row in enumerate(rows):
        risk = float(scores[i]) if i < len(scores) else 0.02
        risk = max(0.0, min(1.0, risk))
        confidence = _compute_confidence(risk)
        decision = _scan_module._decide(risk)
        amount = _get_float(row, "Amount", "amount")
        merchant = _get_str(row, "merchant", "Merchant")

        results.append(BatchRowResult(
            rowIndex=i,
            transactionId=_generate_tx_id(),
            amount=amount,
            riskScore=round(risk, 4),
            confidence=round(confidence, 4),
            decision=decision,
            merchant=merchant,
        ))

        add_decision(DecisionResult(
            id=f"dec-{uuid.uuid4().hex[:8]}",
            transactionId=results[-1].transactionId,
            riskScore=round(risk, 4),
            confidence=round(confidence, 4),
            decision=decision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            merchant=merchant,
            amount=amount,
            factors=[],
        ))

    return results


def _empty_results(rows: List[dict]) -> List[BatchRowResult]:
    """Return safe ALLOW results when no ML model is available."""
    results = []
    for i, row in enumerate(rows):
        amount = _get_float(row, "Amount", "amount")
        merchant = _get_str(row, "merchant", "Merchant")
        results.append(BatchRowResult(
            rowIndex=i,
            transactionId=_generate_tx_id(),
            amount=amount,
            riskScore=0.02,
            confidence=0.97,
            decision="ALLOW",
            merchant=merchant,
        ))
    return results


# ─── Main Endpoint ─────────────────────────────────────────────────────

@router.post("/api/batch", response_model=List[BatchRowResult])
async def batch_analyze(rows: List[dict]):
    """Analyze a batch of transactions using the ML model.

    Results persist to the decision store (dashboard reflects them).
    """
    _scan_module._load_pipeline()

    if not rows:
        return []

    if _scan_module._predictor is not None:
        logger.info("Batch: using ML model (%d rows)", len(rows))
        return _batch_predict_ml(rows)

    raise HTTPException(
        status_code=503,
        detail="The latest LiteFraudNet global model is not loaded. Please verify that a model checkpoint exists in the registry."
    )
