from fastapi import APIRouter, HTTPException
import httpx
import os
import json
import re
from pathlib import Path

from backend.decision_store import get_decisions, get_by_transaction_id

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

KG_STATS = "No KG data available."
try:
    kg_report = Path("artifacts/knowledge_graph/kg_build_report.json")
    if kg_report.exists():
        r = json.loads(kg_report.read_text())
        KG_STATS = f"Knowledge graph: {r.get('nodes',0)} nodes, {r.get('edges',0)} edges, {r.get('communities',0)} communities."
except Exception:
    pass

SYSTEM_PROMPT = """You are a senior fraud intelligence analyst for Trace.ai, a transaction monitoring platform. You respond in a professional, precise, and analytical tone.

{TRANSACTION_DATA}

{KG_STATS}

Transaction data includes: ID, merchant, amount, risk score (%), confidence (%), decision (ALLOW/FLAG/BLOCK), and risk factors with percentage contributions.

The user may type with typos or shorthand. Always infer intent from context.

Answer questions about transactions, fraud patterns, risk analysis, and behavioral trends. When asked about patterns, examine:
- Distribution of flagged/blocked/allowed transactions
- Average risk scores by decision type
- Common merchants among flagged transactions
- Recurring risk factors and their contribution weights
- Amount thresholds that correlate with elevated risk

When asked why a specific transaction was flagged or blocked, cite its risk factors shown in brackets — e.g. [Transaction Amount (35%); Merchant Category (25%); Location Risk (20%)] — and use those as the analytical basis.

When asked about system architecture or methodology, provide a concise professional overview covering:
- The multi-layer risk assessment pipeline (data preprocessing, federated model inference, knowledge graph entity resolution, and ML-driven decision logic)
- The machine learning model (trained via federated learning across distributed client nodes, using a neural network architecture optimized for fraud detection)
- The knowledge graph layer that enriches transaction context with entity relationships, community detection, and risk propagation
- The ReAct-style reasoning agent that synthesizes all signals into a final decision with explainable rationale

Format lists cleanly — one transaction per line: TX-ID — Merchant, $amount, risk X%, decision Y.
Keep responses concise and analytical. If the user asks multiple questions, answer each briefly.
Only answer based on the real transaction data provided above. Never fabricate transaction details.
If asked a personal or off-topic question, respond: "I can only assist with transaction analysis and fraud intelligence queries."""


def _format_txn(d):
    factors_str = ""
    factors = d.get("factors", [])
    if factors:
        top = factors[:3]
        parts = []
        for f in top:
            desc = f.get("description", f.get("name", ""))
            contrib = f.get("contribution", 0)
            parts.append(f"{desc} ({contrib*100:.0f}%)")
        factors_str = " [" + "; ".join(parts) + "]"
    return (
        f"- {d['transactionId']}: {d.get('merchant','?')}, ${d.get('amount',0):,.2f}, "
        f"risk {d.get('riskScore',0)*100:.0f}%, confidence {d.get('confidence',0)*100:.0f}%, "
        f"decision {d.get('decision','?')}{factors_str}"
    )


@router.post("/api/chat")
async def chat_endpoint(payload: dict):
    user_messages = payload.get("messages", [])
    if not user_messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    all_txns = get_decisions(limit=1000)

    # Context: aggregate stats for pattern analysis + recent transactions
    from collections import Counter
    flagged = [d for d in all_txns if d.get("decision") == "FLAG"]
    blocked = [d for d in all_txns if d.get("decision") == "BLOCK"]
    allowed = [d for d in all_txns if d.get("decision") == "ALLOW"]

    flagged_risks = [d.get("riskScore", 0) for d in flagged]
    blocked_risks = [d.get("riskScore", 0) for d in blocked]
    avg_flagged_risk = (sum(flagged_risks) / len(flagged_risks) * 100) if flagged_risks else 0
    avg_blocked_risk = (sum(blocked_risks) / len(blocked_risks) * 100) if blocked_risks else 0

    flagged_merchants = Counter(d.get("merchant", "?") for d in flagged)
    top_flagged_merchants = flagged_merchants.most_common(5)

    all_factors = []
    for d in all_txns:
        for f in d.get("factors", []):
            desc = f.get("description", f.get("name", ""))
            if desc:
                all_factors.append(desc)
    factor_counter = Counter(all_factors)
    top_factors = factor_counter.most_common(5)

    flagged_amounts = [d.get("amount", 0) for d in flagged]
    avg_flagged_amount = sum(flagged_amounts) / len(flagged_amounts) if flagged_amounts else 0

    summary = (
        f"System overview: {len(all_txns)} total transactions — "
        f"{len(flagged)} flagged, {len(blocked)} blocked, {len(allowed)} allowed.\n"
        f"KG metadata: {KG_STATS}\n"
        f"Pattern analysis data:\n"
        f"- Average risk score for flagged: {avg_flagged_risk:.0f}%, for blocked: {avg_blocked_risk:.0f}%\n"
        f"- Top merchants among flagged: {', '.join(f'{m} ({c}x)' for m,c in top_flagged_merchants)}\n"
        f"- Most frequent risk factors: {', '.join(f'{f}' for f,_ in top_factors[:3])}\n"
        f"- Average flagged transaction amount: ${avg_flagged_amount:,.0f}"
    )
    recent = all_txns[:100]
    lines = [summary]
    for d in recent:
        lines.append(_format_txn(d))
    txn_summary = "\n".join(lines)

    # For specific TX ID queries, inject full details via the user message
    user_content = user_messages[-1]["content"]
    txn_match = re.search(r'TX[-_][A-Za-z0-9]+', user_content)
    extra_context = ""
    if txn_match:
        txn_id = txn_match.group().upper()
        d = get_by_transaction_id(txn_id)
        if d:
            extra_context = (
                f"\n\nFull details for {txn_id} requested by user:\n"
                f"{_format_txn(d)}"
            )

    system_with_data = SYSTEM_PROMPT.replace("{TRANSACTION_DATA}", txn_summary).replace("{KG_STATS}", KG_STATS)

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY not set")
            grok_messages = [{"role": "system", "content": system_with_data}]
            for m in user_messages[:-1]:
                grok_messages.append({"role": m["role"], "content": m["content"]})
            # Append extra context to current user message if available
            content_to_send = user_content + extra_context
            grok_messages.append({"role": "user", "content": content_to_send})

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": grok_messages
                }
            )
            response.raise_for_status()
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "Unable to respond.")
            return {"reply": reply}
        except Exception:
            txn_match = re.search(r'TX[-_][A-Za-z0-9]+', user_messages[-1]["content"] if user_messages else "")
            if txn_match:
                txn_id = txn_match.group().upper()
                d = get_by_transaction_id(txn_id)
                if d:
                    factors = d.get("factors", [])
                    factors_str = ""
                    if factors:
                        parts = []
                        for f in factors[:3]:
                            desc = f.get("description", f.get("name", ""))
                            contrib = f.get("contribution", 0)
                            parts.append(f"{desc} ({contrib*100:.0f}%)")
                        factors_str = " Top reasons: " + "; ".join(parts) + "."
                    return {"reply": f"{txn_id} at {d.get('merchant','?')} (${d.get('amount',0):,.2f}) — {d.get('riskScore',0)*100:.0f}% risk, confidence {d.get('confidence',0)*100:.0f}%, decision {d.get('decision','?')}.{factors_str}"}
                return {"reply": f"Transaction {txn_id} not found in our records."}
            query_lower = user_messages[-1]["content"].lower() if user_messages else ""

            def _is_transaction_query(q: str) -> bool:
                """Return True if the query looks transaction-related."""
                personal_signals = [
                    "your name", "how are you", "where are you from", "what are you",
                    "tell me a joke", "tell me a story", "sing", "dance", "cook",
                    "recipe", "your favorite", "do you like", "are you real",
                    "can you feel", "are you sentient", "who created",
                    "i love", "i like", "i think", "i am", "i'm",
                    "my name", "you are", "you're",
                ]
                if any(s in q for s in personal_signals):
                    return False
                if len(q.strip()) < 3:
                    return False
                if q.strip() in ("hi", "hello", "hey", "yo", "bye", "goodbye", "thanks", "ok"):
                    return False
                # Must contain at least one transaction-related keyword
                tx_keywords = [
                    "tx-", "transaction", "fraud", "risk", "amount", "merchant",
                    "flag", "block", "allow", "decision", "score", "confidence",
                    "explain", "pattern", "trend", "analysis",
                ]
                if not any(k in q for k in tx_keywords):
                    return False
                return True

            if _is_transaction_query(query_lower):
                factors_formatted = ", ".join(f'"{f}"' for f,_ in top_factors[:3]) if top_factors else "insufficient factor data"
                return {
                    "reply": (
                        f"Analysis of {len(all_txns)} total transactions in the current dataset: "
                        f"{len(flagged)} flagged (mean risk score: {avg_flagged_risk:.0f}%), "
                        f"{len(blocked)} blocked (mean risk score: {avg_blocked_risk:.0f}%), "
                        f"{len(allowed)} allowed. "
                        f"Most frequent merchants among flagged transactions: {', '.join(f'{m} ({c} occurrences)' for m,c in top_flagged_merchants) or 'none'}. "
                        f"Predominant risk indicators: {factors_formatted}."
                    )
                }
            return {"reply": "I can only assist with transaction analysis and fraud intelligence queries."}


@router.post("/api/explain-transaction")
async def explain_transaction(payload: dict):
    txn_id = payload.get("transaction_id", "")
    merchant = payload.get("merchant", "Unknown")
    amount = payload.get("amount", 0)
    risk_score = payload.get("risk_score", 0)
    decision_type = payload.get("decision_type", "ALLOW")
    key_factors = payload.get("key_factors", [])
    neighbors_flagged = payload.get("neighbors_flagged", 0)
    neighbors_total = payload.get("neighbors_total", 8)
    community_fraud_rate = payload.get("community_fraud_rate", 0.0)
    community_label = payload.get("community_label", "unknown cluster")
    propagated_risk = payload.get("propagated_risk", risk_score)
    risk_tier = payload.get("risk_tier", "MEDIUM")

    risk_pct = round(risk_score * 100)
    prop_pct = round(propagated_risk * 100)
    comm_pct = round(community_fraud_rate * 100)

    # Regenerate factors if none provided (batch-created transactions store empty factors)
    if not key_factors:
        location = payload.get("location", payload.get("ip", ""))
        ip = payload.get("ip", "")
        from backend.routers.scan import _generate_factors as gf
        generated = gf(risk_score, amount, merchant, location, ip)
        key_factors = [f.name for f in generated]

    # Build factor details string
    if key_factors:
        factor_details = "; ".join(key_factors[:4])
    else:
        factor_details = "general transaction risk assessment"

    if risk_score >= 0.70:
        threshold_note = "exceeds BLOCK threshold (70%)"
    elif risk_score >= 0.30:
        threshold_note = "exceeds FLAG threshold (30%)"
    else:
        threshold_note = "within ALLOW range (below 30%)"

    factors_str = ", ".join(key_factors) if key_factors else "none"
    prompt = f"""Transaction {txn_id} at {merchant} for ${amount:,.2f}. Risk score: {risk_pct}% ({threshold_note}). Decision: {decision_type}. Key factors: {factors_str}. Flagged neighbors: {neighbors_flagged}/{neighbors_total}. Cluster fraud rate: {comm_pct}%. Explain this decision in 1-2 sentences using the actual data values. Never mention models, machine learning, or knowledge graphs."""

    if GROQ_API_KEY:
        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {GROQ_API_KEY}"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a senior fraud intelligence analyst. Explain decisions concisely using actual risk score, key factors, peer comparison data, and threshold analysis."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
                response.raise_for_status()
                data = response.json()
                explanation = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if explanation:
                    return {"explanation": explanation, "model_used": "groq"}
            except Exception:
                pass

    # Fallback: generate a substantive explanation from available data
    explanation = (
        f"Transaction {txn_id} at {merchant} (${amount:,.2f}) was assigned "
        f"a risk score of {risk_pct}% ({threshold_note}), resulting in a "
        f"{decision_type} decision. Key contributing factors: {factor_details}. "
        f"Peer comparison: {neighbors_flagged} of {neighbors_total} similar transactions "
        f"in this cluster were flagged (cluster fraud rate: {comm_pct}%)."
    )
    return {"explanation": explanation, "model_used": "fallback"}