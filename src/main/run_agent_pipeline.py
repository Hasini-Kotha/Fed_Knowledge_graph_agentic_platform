"""Layer 5 Runner — Agentic Engine Pipeline.

Usage:
    python src/main/run_agent_pipeline.py
    python src/main/run_agent_pipeline.py --report artifacts/explanations/explain_report.json
    python src/main/run_agent_pipeline.py --report artifacts/explanations/explain_report.json --min-confidence 0.7
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml

from src.agent import AgentConfig, run_agent
from src.agent.tools.log_action import save_execution_report
from src.agent.agent_schema import ExecutionReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/agent_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Agentic Engine (Layer 5).")
    parser.add_argument(
        "--report",
        default="artifacts/explanations/explain_report.json",
        help="Path to explain_report.json from Layer 4",
    )
    parser.add_argument(
        "--config",
        default="configs/agent_config.yaml",
        help="Agent config path",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Override minimum confidence threshold",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_cfg = cfg["agent"]
    output_cfg = cfg["output"]

    output_dir = args.output_dir or output_cfg["dir"]
    log_file = output_cfg["log_file"]

    config = AgentConfig(
        min_confidence=args.min_confidence if args.min_confidence is not None else agent_cfg["min_confidence"],
        max_auto_block=agent_cfg.get("max_auto_block", 10),
        require_confirm_escalation=agent_cfg.get("require_confirm_escalation", True),
        output_dir=output_dir,
        log_file=log_file,
    )

    report_path = Path(args.report)
    if not report_path.exists():
        logger.error("Report not found: %s", report_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("AGENTIC ENGINE (LAYER 5)")
    logger.info("=" * 60)
    logger.info("Source report: %s", report_path)
    logger.info("Config: min_confidence=%.2f, output=%s/%s",
                 config.min_confidence, config.output_dir, config.log_file)

    start = time.time()

    # Run the LangGraph agent
    final_state = run_agent(report_path=str(report_path), config=config)

    elapsed = time.time() - start

    # Build execution report
    action_counts: dict[str, int] = {}
    decisions_data = []
    for d in final_state.decisions:
        action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1
        decisions_data.append({
            "transaction_id": d.transaction_id,
            "action": d.action.value,
            "confidence": d.confidence,
            "tool_used": d.tool_used,
            "success": d.tool_result.success,
            "message": d.tool_result.message,
            "executed_at": d.executed_at,
        })

    exec_report = ExecutionReport(
        run_id=f"run_{int(start)}",
        source_report=str(report_path),
        total_processed=len(final_state.decisions),
        summary=action_counts,
        decisions=list(final_state.decisions),
    )

    # Save execution report
    log_path = Path(output_dir) / log_file
    save_execution_report(exec_report, str(log_path))

    # Print summary
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    logger.info("Transactions processed: %d", exec_report.total_processed)
    logger.info("Time taken: %.1f seconds (%.1f s/txn)", elapsed,
                 elapsed / max(exec_report.total_processed, 1))
    for action, count in sorted(action_counts.items()):
        logger.info("  %s: %d", action, count)

    failed = exec_report.failed_actions()
    if failed:
        logger.warning("Failed actions: %d", len(failed))
        for d in failed:
            logger.warning("  %s: %s", d.transaction_id, d.tool_result.message)

    logger.info("Full action log: %s", log_path)

    banner = f"""
    ============================================
    AGENTIC ENGINE (LAYER 5) COMPLETE
    ============================================
    Transactions processed: {exec_report.total_processed}
    Decisions:
      BLOCK:    {action_counts.get('BLOCK', 0)}
      FLAG:     {action_counts.get('FLAG', 0)}
      ALLOW:    {action_counts.get('ALLOW', 0)}
      ESCALATE: {action_counts.get('ESCALATE', 0)}
    Failed actions: {len(failed)}

    Full log: {log_path}

    The loop is closed:
      Data -> FL -> KG -> Explain -> Agent -> Action
    Completed in {elapsed:.1f} seconds.
    ============================================
    """
    print(banner)


if __name__ == "__main__":
    main()
