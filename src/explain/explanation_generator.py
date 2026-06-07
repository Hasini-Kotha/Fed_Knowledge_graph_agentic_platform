"""Explanation Generator — Orchestrates the full explainability pipeline.

Flow per transaction:
  1. Load evidence bundle from KG JSON
  2. Build structured prompt via PromptBuilder
  3. Send prompt to LLM via OllamaClient
  4. Validate response into Explanation schema
  5. Collect all explanations and save ExplainReport
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.explain.explanation_schema import (
    ExplainReport,
    Explanation,
    SuggestedAction,
    EvidenceBundle,
    SuggestedAction,
)
from src.explain.llm_client import OllamaClient, LLMError
from src.explain.prompt_builder import build_full_prompt

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    def __init__(
        self,
        llm_client: OllamaClient,
        output_dir: str = "artifacts/explanations",
    ):
        self.llm = llm_client
        self.output_dir = Path(output_dir)

    def _validate_llm_response(self, raw: dict[str, Any], expected_id: str) -> Explanation:
        tid = raw.get("transaction_id", expected_id)

        action_raw = raw.get("suggested_action", "FLAG")
        try:
            action = SuggestedAction(action_raw.upper())
        except ValueError:
            logger.warning("Unknown action '%s' for %s, defaulting to FLAG", action_raw, tid)
            action = SuggestedAction.FLAG

        key_factors = raw.get("key_factors", [])
        if not key_factors:
            key_factors = [f"Risk score {raw.get('risk_score', 'N/A')}"]

        # reasoning may be a string or an array of step objects
        reasoning_raw = raw.get("reasoning", "")
        if isinstance(reasoning_raw, list):
            reasoning = " ".join(
                s.get("step", str(s)) if isinstance(s, dict) else str(s)
                for s in reasoning_raw
            )
        else:
            reasoning = str(reasoning_raw)

        return Explanation(
            transaction_id=tid,
            reasoning=reasoning or "No reasoning provided.",
            explanation=raw.get("explanation", ""),
            suggested_action=action,
            confidence=min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0),
            key_factors=key_factors[:5],
        )

    def explain_one(self, evidence: dict[str, Any]) -> EvidenceBundle:
        tid = evidence.get("transaction_id", "unknown")
        logger.info("Explaining transaction: %s", tid)

        prompt = build_full_prompt(evidence)

        raw_response = self.llm.generate(prompt)
        explanation = self._validate_llm_response(raw_response, tid)

        return EvidenceBundle(raw_evidence=evidence, explanation=explanation)

    def explain_all(self, bundles: list[dict[str, Any]]) -> ExplainReport:
        explanations: list[Explanation] = []

        for idx, evidence in enumerate(bundles):
            logger.info("[%d/%d] Explaining %s ...", idx + 1, len(bundles), evidence.get("transaction_id", "?"))
            try:
                bundle = self.explain_one(evidence)
                explanations.append(bundle.explanation)
            except LLMError as e:
                logger.error("Failed to explain %s: %s", evidence.get("transaction_id", "?"), e)
                explanations.append(
                    Explanation(
                        transaction_id=evidence.get("transaction_id", "unknown"),
                        reasoning="LLM explanation failed.",
                        explanation="Could not generate explanation due to LLM error.",
                        suggested_action=SuggestedAction.ESCALATE,
                        confidence=0.0,
                        key_factors=["LLM explanation failed"],
                    )
                )

        return ExplainReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            model=self.llm.model,
            total_transactions=len(bundles),
            explanations=explanations,
        )

    def save_report(self, report: ExplainReport, filename: str | None = None) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / (filename or "explain_report.json")

        report_dict = report.model_dump()
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info("Explain report saved: %s (%d transactions)", path, report.total_transactions)
        return path

    def load_evidence(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evidence file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return data
