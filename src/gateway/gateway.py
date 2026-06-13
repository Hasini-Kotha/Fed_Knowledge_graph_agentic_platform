"""gateway.py — FastAPI application entry point.

Run with:
    uvicorn src.gateway.gateway:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""

# Load .env FIRST — before any other import reads os.environ
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()   # reads .env from CWD (project root)

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .database import init_db
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FL Gateway — FedProx + LiteFraudNet",
    description=(
        "Lightweight API gateway that authenticates bank clients via JWT, "
        "validates model weight updates, and forwards them to the FedProx "
        "aggregation layer."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for the hackathon demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    """Initialise the SQLite database when the gateway starts."""
    init_db()


# ---------------------------------------------------------------------------
# Mount routes
# ---------------------------------------------------------------------------

app.include_router(router, prefix="")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "FL Gateway"}
