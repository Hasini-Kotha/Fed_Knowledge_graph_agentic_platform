"""TraceAI Backend — FastAPI server.

Connects the Next.js frontend to the 5-stage Federated FL + KG pipeline.

Frontend config (frontend/src/lib/config.ts):
    API_BASE_URL = "http://localhost:8000"
    USE_MOCK = true   → flip to false to hit these real endpoints
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import scan, batch, decisions, dashboard, system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="TraceAI API",
    description="Federated Knowledge Graph Enhanced Agentic AI Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router)
app.include_router(batch.router)
app.include_router(decisions.router)
app.include_router(dashboard.router)
app.include_router(system.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "traceai-backend"}
