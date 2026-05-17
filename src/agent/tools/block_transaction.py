"""Tool: block_transaction — Simulates blocking a fraudulent transaction.

In production this would call a banking API.  Currently logs the decision
to the action log and prints a console alert.
"""

import logging

from src.agent.agent_schema import AgentDecision, ToolResult, SuggestedAction

logger = logging.getLogger(__name__)


def block_transaction(decision: AgentDecision) -> ToolResult:
    if decision.action != SuggestedAction.BLOCK:
        return ToolResult(
            tool_name="block_transaction",
            success=False,
            message=f"Called block_transaction but action was {decision.action.value}",
        )

    alert = (
        f"[BLOCK] Transaction {decision.transaction_id} blocked. "
        f"Confidence: {decision.confidence:.2f}. "
        f"Key factors: {'; '.join(decision.key_factors)}"
    )
    logger.warning(alert)
    print(alert)

    return ToolResult(
        tool_name="block_transaction",
        success=True,
        message=f"Transaction {decision.transaction_id} blocked successfully",
        details={
            "transaction_id": decision.transaction_id,
            "confidence": decision.confidence,
            "key_factors": decision.key_factors,
        },
    )
