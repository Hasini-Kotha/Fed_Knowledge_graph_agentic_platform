"""Tool: flag_for_review — Queues a suspicious transaction for analyst review.

In production this creates a ticket in a case management system.
"""

import logging

from src.agent.agent_schema import AgentDecision, ToolResult, SuggestedAction

logger = logging.getLogger(__name__)


def flag_for_review(decision: AgentDecision) -> ToolResult:
    if decision.action != SuggestedAction.FLAG:
        return ToolResult(
            tool_name="flag_for_review",
            success=False,
            message=f"Called flag_for_review but action was {decision.action.value}",
        )

    alert = (
        f"[FLAG] Transaction {decision.transaction_id} flagged for analyst review. "
        f"Confidence: {decision.confidence:.2f}. "
        f"Factors: {'; '.join(decision.key_factors)}"
    )
    logger.info(alert)
    print(alert)

    return ToolResult(
        tool_name="flag_for_review",
        success=True,
        message=f"Transaction {decision.transaction_id} queued for analyst review",
        details={
            "transaction_id": decision.transaction_id,
            "confidence": decision.confidence,
            "status": "PENDING_REVIEW",
        },
    )
