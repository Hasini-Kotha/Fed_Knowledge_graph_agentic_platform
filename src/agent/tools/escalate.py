"""Tool: escalate — Escalates a high-stakes transaction to senior reviewer.

Triggered when the LLM is unsure (ESCALATE action) or when a tool
execution itself fails.
"""

import logging

from src.agent.agent_schema import AgentDecision, ToolResult, SuggestedAction

logger = logging.getLogger(__name__)


def escalate(decision: AgentDecision) -> ToolResult:
    if decision.action != SuggestedAction.ESCALATE:
        return ToolResult(
            tool_name="escalate",
            success=False,
            message=f"Called escalate but action was {decision.action.value}",
        )

    alert = (
        f"[ESCALATE] Transaction {decision.transaction_id} requires senior review. "
        f"Confidence: {decision.confidence:.2f}. "
        f"Reason: {'; '.join(decision.key_factors)}"
    )
    logger.warning(alert)
    print(alert)

    return ToolResult(
        tool_name="escalate",
        success=True,
        message=f"Transaction {decision.transaction_id} escalated to senior reviewer",
        details={
            "transaction_id": decision.transaction_id,
            "confidence": decision.confidence,
            "priority": "HIGH",
        },
    )
