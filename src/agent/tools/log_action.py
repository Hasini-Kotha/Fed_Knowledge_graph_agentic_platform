"""Utility: append_to_log — Appends a decision to the action_log.json file.

This is called AFTER successful tool execution to persist the audit trail.
The log is human-readable JSON Lines format for easy inspection.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.agent.agent_schema import AgentDecision, ExecutionReport

logger = logging.getLogger(__name__)


def append_to_log(
    decision: AgentDecision,
    log_path: str | Path = "artifacts/actions/action_log.json",
) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": decision.transaction_id,
        "action": decision.action.value,
        "confidence": decision.confidence,
        "key_factors": decision.key_factors,
        "tool_used": decision.tool_used,
        "tool_success": decision.tool_result.success,
        "tool_message": decision.tool_result.message,
    }

    existing: list[dict] = []
    if path.exists():
        with open(path) as f:
            try:
                data = json.load(f)
                existing = data if isinstance(data, list) else []
            except json.JSONDecodeError:
                existing = []

    existing.append(entry)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.debug("Appended to action log: %s", entry["transaction_id"])


def save_execution_report(
    report: ExecutionReport,
    log_path: str | Path = "artifacts/actions/action_log.json",
) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report.summary_dict(), f, indent=2, default=str)

    logger.info("Execution report saved: %s (%d decisions)", path, report.total_processed)
    return path
