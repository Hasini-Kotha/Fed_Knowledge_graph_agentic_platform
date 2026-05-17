"""LangGraph Workflow — Agentic Engine state graph.

Graph structure (linear with conditional loop):

    START → INGEST → ANALYZE → DECIDE → EXECUTE
                                          │
                                ┌─────────┴─────────┐
                                │                   │
                          remaining > 0       remaining == 0
                                │                   │
                                ▼                   ▼
                            ANALYZE               END

Why this structure:
  - Linear pipeline per transaction: each step depends on the previous.
  - The EXECUTE → ANALYZE loop processes the queue one-by-one without
    recursion or nested subgraphs.
  - LangGraph's built-in checkpointing means the graph can be resumed
    mid-way if interrupted.
"""

import logging

from langgraph.graph import END, StateGraph

from src.agent.agent_schema import AgentConfig, AgentState
from src.agent.nodes import analyze_node, decide_node, execute_node, ingest_node

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """Conditional edge: continue to ANALYZE if queue not empty, else END."""
    if state.completed or not state.remaining:
        logger.info("WORKFLOW: All transactions processed — ending.")
        return "end"
    return "analyze"


def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("execute", execute_node)

    # Define edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyze")
    workflow.add_edge("analyze", "decide")
    workflow.add_edge("decide", "execute")

    # Conditional loop: EXECUTE → ANALYZE (if more work) or END
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "analyze": "analyze",
            "end": END,
        },
    )

    return workflow.compile()


def run_agent(
    report_path: str = "artifacts/explanations/explain_report.json",
    config: AgentConfig | None = None,
) -> AgentState:
    """Convenience: build graph, initialise state, run, return final state."""
    if config is None:
        config = AgentConfig()

    graph = build_agent_graph()
    initial = AgentState(config=config, report_path=report_path)
    final = graph.invoke(initial)

    if isinstance(final, dict):
        return AgentState(**final)
    return final
