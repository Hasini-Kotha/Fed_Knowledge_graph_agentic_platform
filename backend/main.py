"""TraceAI Backend — FastAPI server.

Connects the Next.js frontend to the 5-stage Federated FL + KG pipeline.

Frontend config (frontend/src/lib/config.ts):
    API_BASE_URL = "http://localhost:8000"
    USE_MOCK = true   → flip to false to hit these real endpoints
"""

# Load .env FIRST — before any other import reads os.environ
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()   # reads .env from CWD (project root)

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import scan, batch, decisions, dashboard, system, explain, admin
from backend.decision_store import DB_PATH

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router)
app.include_router(batch.router)
app.include_router(decisions.router)
app.include_router(dashboard.router)
app.include_router(system.router)
app.include_router(explain.router)
app.include_router(admin.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "traceai-backend"}


@app.get("/api/debug/db-path")
async def debug_db_path():
    return {
        "cwd": os.getcwd(),
        "db_path": str(DB_PATH),
        "db_abs": str(DB_PATH.resolve()),
        "db_exists": DB_PATH.exists(),
    }


@app.get("/api/debug/predictor")
async def debug_predictor():
    import backend.routers.scan as scan_mod
    info = {"predictor_loaded": scan_mod._predictor is not None}
    if scan_mod._predictor is None:
        try:
            scan_mod._load_pipeline()
            info["after_load"] = scan_mod._predictor is not None
            if scan_mod._predictor is not None:
                info["summary"] = scan_mod._predictor.get_summary()
        except Exception as e:
            info["error"] = str(e)
            import traceback
            info["traceback"] = traceback.format_exc()
    else:
        info["summary"] = scan_mod._predictor.get_summary()
    return info
