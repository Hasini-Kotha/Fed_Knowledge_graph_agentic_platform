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

SYSTEM_PROMPT = """You are a fraud alert assistant for Trace.ai, a transaction monitoring platform.

{TRANSACTION_DATA}

{KG_STATS}

Understand the user's intent even with typos or imperfect phrasing. Answer any question about transactions, fraud, risk, patterns, or anything in the transaction data. Refuse only if the question is clearly personal or about technical internals (ML, KG, FL, APIs, pipelines).

When asked about patterns, trends, or common characteristics across transactions (of any type: flagged, blocked, allowed, or all), analyze the data provided, identify recurring themes (common merchants, risk ranges, factor frequencies), and produce a business-friendly summary. Discuss total flagged/blocked counts, average risk scores, common merchant categories, recurring fraud indicators, and high-level trends.

You NEVER answer questions about:
- How the system works internally
- Machine learning, AI, or model architecture
- Federated learning, knowledge graphs, neural networks
- APIs, databases, code, pipelines, or backend infrastructure
- Any technical implementation details whatsoever

If someone asks a clearly personal question (like "where are you from" or casual chat), respond with exactly:
"I'm not able to help with that. Please ask me about your transactions."

If someone asks about technical internals (ML, KG, architecture, APIs, pipelines), respond with exactly:
"I can only help you with your transaction queries and fraud alerts. Please contact support for technical questions."

For anything else — even with typos — try your best to answer from the transaction data.

You must NEVER reveal that an AI model, knowledge graph, or federated learning is involved.
You are a simple fraud alert assistant. Speak like a bank's customer support agent, not an engineer.
Keep all responses short, clear, and non-technical.
Format lists cleanly — one transaction per line: TX-ID — Merchant, $amount, risk X%, decision Y.
Never add commentary about duplicates, disregarded entries, or data quality.
If the user asks multiple questions, answer each briefly.
Only answer based on the real transaction data provided above. Never guess or make up transaction details."""


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
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not set on server")

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
                    factors_str = ""
                    factors = d.get("factors", [])
                    if factors:
                        top = factors[:3]
                        parts = []
                        for f in top:
                            desc = f.get("description", f.get("name", ""))
                            parts.append(desc)
                        factors_str = " Factors: " + "; ".join(parts) + "."
                    return {"reply": f"{txn_id} at {d.get('merchant','?')} (${d.get('amount',0):,.2f}) — {d.get('riskScore',0)*100:.0f}% risk, confidence {d.get('confidence',0)*100:.0f}%, decision {d.get('decision','?')}.{factors_str}"}
                return {"reply": f"Transaction {txn_id} not found in our records."}
            query_lower = user_messages[-1]["content"].lower() if user_messages else ""
            trx_signals = ["transak", "transact", "tx", "fraud", "flag", "blok", "block", "allow", "risk", "pattern", "trend", "merchant", "suspicious", "today", "list", "show", "tell", "what", "how", "why", "genaral", "general", "characteristic"]
            if any(s in query_lower for s in trx_signals):
                factors_formatted = ", ".join(f'"{f}"' for f,_ in top_factors[:3])
                return {
                    "reply": (
                        f"Based on today's data ({len(all_txns)} transactions total): "
                        f"{len(flagged)} flagged (avg risk {avg_flagged_risk:.0f}%), "
                        f"{len(blocked)} blocked (avg risk {avg_blocked_risk:.0f}%), "
                        f"{len(allowed)} allowed. "
                        f"Top flagged merchants: {', '.join(f'{m} ({c})' for m,c in top_flagged_merchants)}. "
                        f"Common risk factors: {factors_formatted}."
                    )
                }
            return {"reply": "I'm not able to help with that. Please ask me about your transactions."}


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

    if not GROQ_API_KEY:
        explanation = (
            f"{txn_id} at {merchant} (${amount:,.2f}) got a {risk_pct}% risk score. "
            f"Action: {decision_type}."
        )
        return {"explanation": explanation, "model_used": "fallback"}

    factors_str = ", ".join(key_factors) if key_factors else "none"
    prompt = f"""Transaction {txn_id} at {merchant} for ${amount:,.2f}. Risk score: {risk_pct}%. Decision: {decision_type}. Key factors: {factors_str}. Flagged neighbors: {neighbors_flagged}/{neighbors_total}. Cluster fraud rate: {comm_pct}%. Explain this decision in 1-2 sentences using the actual data values. Never mention models, machine learning, or knowledge graphs."""

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
                        {"role": "system", "content": "You are a fraud analyst. Explain decisions concisely using actual risk score, key factors, and peer comparison data. Never mention models, ML, KG, or technical internals."},

                        {"role": "user", "content": prompt}
                    ]
                }
            )
            response.raise_for_status()
            data = response.json()
            explanation = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not explanation:
                explanation = f"{txn_id} was {decision_type} with a {risk_pct}% risk score."
            return {"explanation": explanation, "model_used": "groq"}
        except Exception:
            explanation = (
                f"{txn_id} at {merchant} (${amount:,.2f}) got a {risk_pct}% risk score. "
                f"Action: {decision_type}."
            )
            return {"explanation": explanation, "model_used": "fallback"}