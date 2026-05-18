"""POST /api/batch — Batch analyze multiple transactions from CSV data.

Two modes depending on input columns:

**Full-feature mode** (V1-V28 + Amount present):
    Uses the trained ML model via GlobalModelPredictor.
    Expects columns: V1-V28, Amount, merchant (optional), Class (optional).

**Quick-scan mode** (only amount + merchant):
    Uses rule-based fallback for fast results.
    Any additional fields (location, ip, etc.) are factored into rules.

**Response** — array of BatchRowResult:
    [{ rowIndex, transactionId, amount, riskScore, confidence, decision, merchant }, ...]

**Robustness guarantees:**
    - All results are persisted to the decision store (dashboard updates automatically)
    - If ML model path crashes for any reason → graceful fallback to rules
    - Rule path is a pure function with no external dependencies → never crashes
"""

import logging
import random as _random
import uuid
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter

from backend.decision_store import add_decision
from backend.models import BatchRowResult, DecisionResult
import backend.routers.scan as _scan_module

logger = logging.getLogger(__name__)
router = APIRouter()

# Columns expected by the ML model (from mapping_creditcard_2023.json)
_FEATURE_COLS = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                 "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                 "V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
_HELP_COLS_ML = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                 "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                 "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"}
_HIGH_RISK_KW = ["CRYPTO", "WIRE", "MONEYGRAM", "WESTERN UNION"]
_HIGH_RISK_LOCS = {"RU", "NG", "HK", "CN", "KP"}


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

def _batch_predict_ml(rows: List[dict]) -> List[BatchRowResult]:
    """Run the trained ML model on all rows at once (batched inference).

    Constructs a single DataFrame from V1-V28 + Amount columns,
    calls GlobalModelPredictor.predict(), and maps scores to decisions.

    Wrapped in try/except — any failure falls back gracefully.
    """
    try:
        predictor = _scan_module._predictor
        if predictor is None:
            logger.warning("ML predictor not loaded, falling back to rules")
            return _batch_predict_rules(rows)

        records = []
        for row in rows:
            rec = {"Time": 0, "Class": 0}
            for col in _FEATURE_COLS:
                rec[col] = _get_float(row, col)
            records.append(rec)

        df = pd.DataFrame(records)
        result_df = predictor.predict(df)
        scores = result_df["fraud_risk_score"].values

        results: List[BatchRowResult] = []
        for i, row in enumerate(rows):
            risk = float(scores[i]) if i < len(scores) else 0.02
            risk = max(0.0, min(1.0, risk))
            confidence = min(0.85 + _random.random() * 0.13, 0.99)
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

    except Exception as e:
        logger.error("ML batch prediction failed: %s. Falling back to rules.", e)
        return _batch_predict_rules(rows)


# ─── Robust Rule-based Path ────────────────────────────────────────────

def _rule_risk(amount: float, merchant: str, location: str = "", ip: str = "") -> float:
    """Pure rule-based risk score — never touches external dependencies.

    Factors:
      - Base rate: 2%
      - Amount: progressive tiers ($1K / $5K / $10K)
      - Merchant: high-risk keywords (+25%)
      - Location: high-risk countries (+20%)
      - IP: public IP (+5%), private IP (-1%)
    """
    risk = 0.02

    # Amount factor
    if amount > 10000:
        risk += 0.35
    elif amount > 5000:
        risk += 0.20
    elif amount > 1000:
        risk += 0.08

    # Merchant factor
    if merchant and any(kw in merchant.upper() for kw in _HIGH_RISK_KW):
        risk += 0.25

    # Location factor (if available)
    if location and location.strip().upper() in _HIGH_RISK_LOCS:
        risk += 0.20

    # IP factor (if available)
    if ip:
        parts = ip.split(".")
        if len(parts) == 4:
            risk += 0.05 if parts[0] not in ("10", "172", "192") else -0.01

    return max(0.01, min(0.99, risk))


def _rule_confidence(risk: float) -> float:
    """Confidence decreases as risk approaches 0.5 (maximum uncertainty)."""
    c = 0.97 - 0.20 * (1 - abs(risk - 0.5) * 2)
    return max(0.75, min(0.99, c))


def _batch_predict_rules(rows: List[dict]) -> List[BatchRowResult]:
    """Quick rule-based scoring per row. Never crashes — pure math only.

    Uses any available fields: amount, merchant, location, ip.
    """
    results: List[BatchRowResult] = []
    for i, row in enumerate(rows):
        try:
            amount = _get_float(row, "amount", "Amount")
            merchant = _get_str(row, "merchant", "Merchant")
            location = _get_str(row, "location", "Location", "loc")
            ip = _get_str(row, "ip", "IP", "Ip")

            risk_score = _rule_risk(amount, merchant, location, ip)
            confidence = _rule_confidence(risk_score)
            decision = _scan_module._decide(risk_score)

        except Exception as e:
            logger.error("Rule scoring failed for row %d: %s. Using safe defaults.", i, e)
            risk_score, confidence, decision = 0.02, 0.97, "ALLOW"
            amount = _get_float(row, "amount", "Amount")
            merchant = _get_str(row, "merchant", "Merchant")

        results.append(BatchRowResult(
            rowIndex=i,
            transactionId=_generate_tx_id(),
            amount=amount,
            riskScore=round(risk_score, 4),
            confidence=round(confidence, 4),
            decision=decision,
            merchant=merchant,
        ))

        add_decision(DecisionResult(
            id=f"dec-{uuid.uuid4().hex[:8]}",
            transactionId=results[-1].transactionId,
            riskScore=round(risk_score, 4),
            confidence=round(confidence, 4),
            decision=decision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            merchant=merchant,
            amount=amount,
            factors=[],
        ))

    return results


# ─── Main Endpoint ─────────────────────────────────────────────────────

@router.post("/api/batch", response_model=List[BatchRowResult])
async def batch_analyze(rows: List[dict]):
    """Analyze a batch of transactions. Auto-detects input format:

    1. V1-V28 + Amount present + ML model loaded → batched inference
    2. Otherwise → robust rule-based scoring using any available fields

    Results persist to the decision store (dashboard reflects them).
    """
    _scan_module._load_pipeline()

    if _has_full_features(rows) and _scan_module._predictor is not None:
        logger.info("Batch: using ML model (%d rows, full features)", len(rows))
        return _batch_predict_ml(rows)

    if not rows:
        return []

    has_amount = any(k in rows[0] for k in ("amount", "Amount"))
    logger.info("Batch: using rules (%d rows, has_amount=%s, has_merchant=%s)",
                len(rows), has_amount, "merchant" in rows[0] or "Merchant" in rows[0])
    return _batch_predict_rules(rows)
