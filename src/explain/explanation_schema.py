from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SuggestedAction(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class Explanation(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID from the KG evidence bundle")
    reasoning: str = Field(
        ...,
        description="Step-by-step logical reasoning that led to the decision, referencing specific evidence fields",
    )
    explanation: str = Field(
        ...,
        description="Plain-English explanation suitable for a human analyst or end customer",
    )
    suggested_action: SuggestedAction = Field(
        ...,
        description=(
            "Recommended action: ALLOW = no concern, FLAG = notify analyst, "
            "BLOCK = stop transaction, ESCALATE = senior review required"
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the recommendation (0.0–1.0)",
    )
    key_factors: list[str] = Field(
        ...,
        min_length=1, max_length=5,
        description="Top 3–5 factors driving the decision, each a concise statement",
    )


class EvidenceBundle(BaseModel):
    raw_evidence: dict[str, Any] = Field(..., description="Original evidence bundle from KG JSON")
    explanation: Explanation = Field(..., description="LLM-generated explanation for this transaction")


class ExplainReport(BaseModel):
    generated_at: str = Field(..., description="ISO-8601 timestamp")
    model: str = Field(..., description="LLM model used")
    total_transactions: int = Field(..., description="Number of transactions explained")
    explanations: list[Explanation] = Field(..., description="Ordered list of explanations")

    def count_by_action(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exp in self.explanations:
            counts[exp.suggested_action.value] = counts.get(exp.suggested_action.value, 0) + 1
        return counts

    def high_confidence_flags(self) -> list[Explanation]:
        return [e for e in self.explanations if e.suggested_action in (SuggestedAction.BLOCK, SuggestedAction.ESCALATE) and e.confidence >= 0.8]

    def get_actionable_summary(self) -> dict[str, Any]:
        return {
            "total": self.total_transactions,
            "decisions": self.count_by_action(),
            "critical_alerts": [
                {
                    "transaction_id": e.transaction_id,
                    "action": e.suggested_action.value,
                    "confidence": e.confidence,
                    "key_factors": e.key_factors,
                    "explanation": e.explanation,
                }
                for e in self.high_confidence_flags()
            ],
        }
