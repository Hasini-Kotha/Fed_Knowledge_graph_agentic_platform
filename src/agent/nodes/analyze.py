import logging

from src.agent.agent_schema import AgentState, SuggestedAction
from src.explain.explanation_schema import Explanation

logger = logging.getLogger(__name__)


def analyze_node(state: AgentState) -> AgentState:
    if not state.remaining:
        return state.model_copy(update={"completed": True, "current": None})

    current: Explanation = state.remaining[0]
    remaining = list(state.remaining[1:])

    min_conf = state.config.min_confidence

    if current.confidence < min_conf:
        original_action = current.suggested_action
        current.suggested_action = SuggestedAction.FLAG
        logger.info(
            "ANALYZE: %s confidence=%.2f < min=%.2f — downgraded %s -> FLAG",
            current.transaction_id, current.confidence, min_conf, original_action.value,
        )

    return state.model_copy(
        update={
            "current": current,
            "remaining": remaining,
        }
    )
