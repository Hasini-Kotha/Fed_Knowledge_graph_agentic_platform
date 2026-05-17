import logging

from src.agent.agent_schema import AgentDecision, AgentState, SuggestedAction, ToolResult

logger = logging.getLogger(__name__)

ACTION_TO_TOOL: dict[SuggestedAction, str] = {
    SuggestedAction.BLOCK: "block_transaction",
    SuggestedAction.FLAG: "flag_for_review",
    SuggestedAction.ESCALATE: "escalate",
    SuggestedAction.ALLOW: "allow_transaction",
}


def decide_node(state: AgentState) -> AgentState:
    current = state.current
    if current is None:
        return state.model_copy(update={"completed": not bool(state.remaining)})

    action = current.suggested_action
    tool_name = ACTION_TO_TOOL.get(action, "flag_for_review")

    decision = AgentDecision(
        transaction_id=current.transaction_id,
        action=action,
        confidence=current.confidence,
        key_factors=list(current.key_factors),
        tool_used=tool_name,
        tool_result=ToolResult(
            tool_name=tool_name,
            success=False,
            message="Pending execution",
        ),
    )

    logger.info(
        "DECIDE: %s -> %s (tool=%s, confidence=%.2f)",
        current.transaction_id, action.value, tool_name, current.confidence,
    )

    return state.model_copy(update={"decision": decision})
