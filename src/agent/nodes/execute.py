import logging

from src.agent.agent_schema import AgentDecision, AgentState, ToolResult
from src.agent.tools import (
    allow_transaction,
    append_to_log,
    block_transaction,
    escalate,
    flag_for_review,
)

logger = logging.getLogger(__name__)

TOOL_REGISTRY: dict[str, callable] = {
    "block_transaction": block_transaction,
    "flag_for_review": flag_for_review,
    "escalate": escalate,
    "allow_transaction": allow_transaction,
}


def execute_node(state: AgentState) -> AgentState:
    decision = state.decision
    if decision is None:
        return state.model_copy(
            update={"errors": state.errors + ["No decision to execute"]}
        )

    tool_fn = TOOL_REGISTRY.get(decision.tool_used)
    if tool_fn is None:
        result = ToolResult(
            tool_name=decision.tool_used,
            success=False,
            message=f"Unknown tool: {decision.tool_used}",
        )
    else:
        try:
            result = tool_fn(decision)
        except Exception as e:
            logger.error("Tool %s failed with exception: %s", decision.tool_used, e)
            result = ToolResult(
                tool_name=decision.tool_used,
                success=False,
                message=f"Exception: {e}",
            )

    completed_decision = decision.model_copy(update={"tool_result": result})

    try:
        append_to_log(completed_decision, f"{state.config.output_dir}/{state.config.log_file}")
    except Exception as e:
        logger.warning("Failed to append to action log: %s", e)

    all_decisions = list(state.decisions) + [completed_decision]
    remaining = bool(state.remaining)

    return state.model_copy(
        update={
            "decision": completed_decision,
            "decisions": all_decisions,
            "completed": not remaining,
            "current": None,
        }
    )
