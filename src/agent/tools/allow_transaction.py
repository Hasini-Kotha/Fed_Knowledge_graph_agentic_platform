"""Tool: allow_transaction — Approves a legitimate transaction.

This is the default for low-risk transactions.  In production this
would be a no-op or a lightweight log entry.
"""

import logging

from src.agent.agent_schema import AgentDecision, ToolResult, SuggestedAction

logger = logging.getLogger(__name__)


def allow_transaction(decision: AgentDecision) -> ToolResult:
    if decision.action != SuggestedAction.ALLOW:
        return ToolResult(
            tool_name="allow_transaction",
            success=False,
            message=f"Called allow_transaction but action was {decision.action.value}",
        )

    logger.info(
        "[ALLOW] Transaction %s approved (confidence=%.2f)",
        decision.transaction_id,
        decision.confidence,
    )
    print(f"[ALLOW] Transaction {decision.transaction_id} approved.")

    return ToolResult(
        tool_name="allow_transaction",
        success=True,
        message=f"Transaction {decision.transaction_id} approved",
        details={
            "transaction_id": decision.transaction_id,
            "confidence": decision.confidence,
            "status": "APPROVED",
        },
    )
