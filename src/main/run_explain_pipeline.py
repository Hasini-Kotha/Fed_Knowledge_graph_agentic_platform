"""Layer 4 Runner — Explainability Pipeline.

Usage:
    python src/main/run_explain_pipeline.py
    python src/main/run_explain_pipeline.py --evidence artifacts/knowledge_graph/top_risk_evidence.json
    python src/main/run_explain_pipeline.py --evidence artifacts/knowledge_graph/top_risk_evidence.json --top-k 3
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml

from src.explain import ExplanationGenerator, OllamaClient, ExplainReport
from src.explain.explanation_schema import SuggestedAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/explain_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Explainability pipeline (Layer 4).")
    parser.add_argument(
        "--evidence",
        default="artifacts/knowledge_graph/top_risk_evidence.json",
        help="Path to KG evidence JSON (top_risk_evidence.json)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Only explain the first N transactions")
    parser.add_argument("--config", default="configs/explain_config.yaml", help="Explain config path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    llm_cfg = config["llm"]
    output_cfg = config["output"]
    output_dir = args.output_dir or output_cfg["dir"]

    logger.info("=" * 60)
    logger.info("EXPLAINABILITY PIPELINE (LAYER 4)")
    logger.info("=" * 60)
    logger.info("LLM: %s (%s)", llm_cfg["model"], llm_cfg["provider"])
    logger.info("Output: %s", output_dir)

    # Initialise LLM client
    client = OllamaClient(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        timeout_seconds=llm_cfg["timeout_seconds"],
    )

    if not client.health_check():
        logger.error(
            "Ollama is not running at %s. Start it with: ollama serve",
            llm_cfg["base_url"],
        )
        sys.exit(1)

    # Initialise generator
    generator = ExplanationGenerator(llm_client=client, output_dir=output_dir)

    # Load evidence
    evidence_path = Path(args.evidence)
    if not evidence_path.exists():
        logger.error("Evidence file not found: %s", evidence_path)
        sys.exit(1)

    bundles = generator.load_evidence(str(evidence_path))
    if args.top_k:
        bundles = bundles[: args.top_k]

    logger.info("Loaded %d evidence bundles from %s", len(bundles), evidence_path)

    start = time.time()

    # Run explanations
    report: ExplainReport = generator.explain_all(bundles)

    elapsed = time.time() - start

    # Save report
    report_path = generator.save_report(report)

    # Print summary
    action_counts = report.count_by_action()
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    logger.info("Total transactions explained: %d", report.total_transactions)
    logger.info("Time taken: %.1f seconds (%.1f s/txn)", elapsed, elapsed / max(len(bundles), 1))
    for action, count in sorted(action_counts.items()):
        logger.info("  %s: %d", action, count)

    critical = report.high_confidence_flags()
    if critical:
        logger.info("-" * 60)
        logger.info("CRITICAL ALERTS (BLOCK/ESCALATE with confidence >= 0.8):")
        logger.info("-" * 60)
        for exp in critical:
            logger.info("  %s → %s (confidence=%.2f)", exp.transaction_id, exp.suggested_action.value, exp.confidence)
            for factor in exp.key_factors:
                logger.info("    - %s", factor)

    banner = f"""
    ============================================
    EXPLAINABILITY LAYER (LAYER 4) COMPLETE
    ============================================
    Transactions explained: {report.total_transactions}
    Decisions:
      ALLOW:    {action_counts.get('ALLOW', 0)}
      FLAG:     {action_counts.get('FLAG', 0)}
      BLOCK:    {action_counts.get('BLOCK', 0)}
      ESCALATE: {action_counts.get('ESCALATE', 0)}
    Critical alerts: {len(critical)}

    Report saved to: {report_path}

    Ready for: Agentic Engine (Layer 5) -> React Agent
    Completed in {elapsed:.1f} seconds.
    ============================================
    """
    print(banner)


if __name__ == "__main__":
    main()
