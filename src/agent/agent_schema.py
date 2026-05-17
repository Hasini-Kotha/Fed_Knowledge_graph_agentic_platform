from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

from src.explain.explanation_schema import ExplainReport, Explanation, SuggestedAction


class ToolResult(BaseModel):
    """Result from executing a single agent tool."""

    tool_name: str = Field(..., description="Name of the tool that executed")
    success: bool = Field(..., description="Whether execution succeeded")
    message: str = Field(..., description="Human-readable result description")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional structured output")


class AgentDecision(BaseModel):
    """Complete record of one agent decision cycle for a single transaction."""

    transaction_id: str = Field(..., description="Transaction ID")
    action: SuggestedAction = Field(..., description="Action taken")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the action")
    key_factors: list[str] = Field(..., description="Evidence factors that drove the decision")
    tool_used: str = Field(..., description="Name of the tool executed")
    tool_result: ToolResult = Field(..., description="Outcome of tool execution")
    executed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of execution",
    )


class ExecutionReport(BaseModel):
    """Final audit report produced by the agent after processing all transactions."""

    run_id: str = Field(..., description="Unique run identifier (timestamp-based)")
    source_report: str = Field(..., description="Path to the source explain_report.json")
    total_processed: int = Field(..., ge=0, description="Number of transactions processed")
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each action taken, e.g. {BLOCK: 4, FLAG: 1}",
    )
    decisions: list[AgentDecision] = Field(
        default_factory=list,
        description="All decisions made, in processing order",
    )

    def count_by_action(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return counts

    def failed_actions(self) -> list[AgentDecision]:
        return [d for d in self.decisions if not d.tool_result.success]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_report": self.source_report,
            "total_processed": self.total_processed,
            "summary": self.count_by_action(),
            "failed_count": len(self.failed_actions()),
            "actions": [
                {
                    "transaction_id": d.transaction_id,
                    "action": d.action.value,
                    "confidence": d.confidence,
                    "tool_used": d.tool_used,
                    "success": d.tool_result.success,
                    "message": d.tool_result.message,
                    "executed_at": d.executed_at,
                }
                for d in self.decisions
            ],
        }


class AgentConfig(BaseModel):
    """Runtime configuration for the agent."""

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_auto_block: int = Field(default=10, ge=1)
    require_confirm_escalation: bool = Field(default=True)
    output_dir: str = Field(default="artifacts/actions")
    log_file: str = Field(default="action_log.json")


class AgentState(BaseModel):
    """LangGraph runtime state — queue-based processing."""

    report_path: str = Field(default="artifacts/explanations/explain_report.json", description="Path to explain_report.json")
    report: Optional[ExplainReport] = Field(default=None, description="Loaded explain report")
    remaining: list[Explanation] = Field(default_factory=list, description="Explanations not yet processed (FIFO)")
    current: Optional[Explanation] = Field(default=None, description="Explanation currently being processed")
    decision: Optional[AgentDecision] = Field(default=None, description="Latest decision made")
    decisions: list[AgentDecision] = Field(default_factory=list, description="All decisions accumulated")
    errors: list[str] = Field(default_factory=list, description="Non-fatal errors encountered")
    completed: bool = Field(default=False, description="True when all explanations processed")
    config: AgentConfig = Field(default_factory=AgentConfig, description="Runtime configuration")
