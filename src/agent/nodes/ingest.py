import json
import logging
from pathlib import Path
from typing import Optional

from src.agent.agent_schema import AgentState
from src.explain.explanation_schema import ExplainReport

logger = logging.getLogger(__name__)


def ingest_node(state: AgentState) -> AgentState:
    path = Path(state.report_path)

    if not path.exists():
        return state.model_copy(
            update={
                "errors": state.errors + [f"Report not found: {path}"],
                "completed": True,
            }
        )

    try:
        with open(path) as f:
            raw = json.load(f)
        report = ExplainReport.model_validate(raw)
    except (json.JSONDecodeError, Exception) as e:
        return state.model_copy(
            update={
                "errors": state.errors + [f"Failed to parse report: {e}"],
                "completed": True,
            }
        )

    return state.model_copy(
        update={
            "report": report,
            "remaining": list(report.explanations),
        }
    )
