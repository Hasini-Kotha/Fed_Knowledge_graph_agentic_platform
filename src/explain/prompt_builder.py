"""Prompt Builder — Converts KG evidence bundles into structured LLM prompts.

This module is the only place prompt templates live.  Changing the prompt
structure, adding few-shot examples, or switching languages requires editing
only this file.
"""

import logging
from typing import Any

from src.explain.explanation_schema import SuggestedAction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fraud investigation AI. Your job is to analyse structured evidence from a financial knowledge graph and produce a clear, auditable explanation for each flagged transaction.

Rules:
1. Base your analysis ONLY on the evidence provided — do not invent facts.
2. Be specific: reference actual values (risk scores, neighbour counts, community fraud rates).
3. Output valid JSON matching the required schema exactly.
4. Choose a suggested_action from: ALLOW (no concern), FLAG (notify analyst), BLOCK (stop transaction), ESCALATE (senior review required).
5. Set confidence between 0.0 and 1.0 reflecting how strongly the evidence supports your recommendation.
6. List 3–5 key_factors — each a concise statement of one evidence point that drove your decision.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_evidence_section(evidence: dict[str, Any]) -> str:
    parts = ["=== EVIDENCE ==="]

    # Core risk
    parts.append(f"Transaction ID: {evidence.get('transaction_id', 'unknown')}")
    parts.append(f"Risk score: {evidence.get('risk_score', 'N/A'):.4f}")
    parts.append(f"Risk tier: {evidence.get('risk_tier', 'unknown')}")
    parts.append(f"Model prediction: {evidence.get('model_prediction', 'unknown')}")
    parts.append(f"Ground truth (if available): {evidence.get('ground_truth', 'N/A')}")

    # Financial context
    amt = evidence.get("amount")
    if amt is not None:
        parts.append(f"Amount: ${amt:.2f}")

    # Neighbourhood risk (SIMILAR_PATTERN peers)
    nr = evidence.get("neighborhood_risk")
    nc = evidence.get("neighbor_count", 0)
    if nc > 0 and nr is not None:
        parts.append(f"Similar-pattern peers: {nc} peers with avg risk {nr:.4f}")

    # Connected entity context
    entities = evidence.get("connected_entities", {})
    if entities:
        entity_str = "; ".join(f"{k}: {v}" for k, v in entities.items())
        parts.append(f"Connected entities: {entity_str}")

    # Direct high-risk neighbours
    hr_count = evidence.get("connected_high_risk_count", 0)
    if hr_count > 0:
        parts.append(f"Direct high-risk neighbours: {hr_count}")

    # Similar flagged transactions
    similar = evidence.get("similar_flagged_transactions", [])
    if similar:
        parts.append(f"Similar flagged transactions ({len(similar)} found):")
        for s in similar[:5]:
            parts.append(
                f"  - {s.get('id', '?')}: risk={s.get('risk_score', 0):.4f}, "
                f"tier={s.get('risk_tier', '?')}, similarity={s.get('similarity', 0):.4f}"
            )

    # Community context
    community = evidence.get("community", {})
    if community.get("community_id", -1) >= 0:
        parts.append(
            f"Behavioural cluster #{community['community_id']}: "
            f"{community.get('size', 0)} members, "
            f"avg risk {community.get('avg_risk', 0):.4f}, "
            f"fraud ratio {community.get('high_risk_ratio', 0):.0%}"
        )

    # Time window
    tw = evidence.get("time_window_context", {})
    if tw and tw.get("fraud_rate") is not None:
        parts.append(
            f"Time window {tw.get('window_id', '?')}: "
            f"fraud rate {tw.get('fraud_rate', 0):.2%} vs "
            f"baseline {tw.get('baseline_rate', 0):.2%} "
            f"({tw.get('elevation_factor', 1):.1f}x elevation)"
        )

    parts.append("=== END EVIDENCE ===")
    return "\n".join(parts)


OUTPUT_INSTRUCTION = """Now produce your analysis as valid JSON with exactly these fields:
{
  "transaction_id": "<same as input>",
  "reasoning": "<step-by-step logic>",
  "explanation": "<plain-English for human>",
  "suggested_action": "<ALLOW|FLAG|BLOCK|ESCALATE>",
  "confidence": <0.0-1.0>,
  "key_factors": ["<factor 1>", "<factor 2>", ...]
}
"""


def build_user_prompt(evidence: dict[str, Any]) -> str:
    sections = [
        build_evidence_section(evidence),
        "",
        OUTPUT_INSTRUCTION,
    ]
    return "\n".join(sections)


def build_full_prompt(evidence: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(evidence)},
    ]
