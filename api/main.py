from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

HOST        = os.getenv("HOST", "0.0.0.0")
PORT        = int(os.getenv("PORT", "8000"))
APP_NAME    = os.getenv("APP_NAME", "Orion Alpha")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Loguru config ─────────────────────────────────────────────────────────────
logger.remove()  # Remove default handler
logger.add(
    sink=lambda msg: print(msg, end=""),  # stdout (container-friendly)
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="DEBUG" if ENVIRONMENT == "development" else "INFO",
    colorize=True,
)

# ── Lifespan: startup / shutdown events ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    logger.info("━" * 60)
    logger.info(f"  🚀  {APP_NAME} is running")
    logger.info(f"  ENV   : {ENVIRONMENT}")
    logger.info(f"  MODEL : {GROQ_MODEL}")
    logger.info(f"  HOST  : {HOST}:{PORT}")
    logger.info("━" * 60)
    print("\n  ✅  Orion Alpha is running\n")
    yield
    # ── SHUTDOWN ──
    logger.info(f"🛑  {APP_NAME} shutting down — goodbye.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    description=(
        "Agentic investment research system for pre-seed VCs. "
        "Autonomously researches startups and returns structured "
        "INVEST / PASS / WATCH recommendations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS — wide-open for prototype ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.routes.research import router as research_router   # noqa: E402

app.include_router(research_router, prefix="/api/v1")

# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="App status and metadata",
)
async def health() -> dict:
    logger.debug("Health check called")
    return {
        "app":         APP_NAME,
        "status":      "ok",
        "environment": ENVIRONMENT,
        "model":       GROQ_MODEL,
        "version":     "0.1.0",
        "docs":        "/docs",
    }


# ── Schema endpoint ───────────────────────────────────────────────────────────
@app.get(
    "/schema",
    tags=["System"],
    summary="Return the canonical ResearchReport output schema",
)
async def schema() -> dict:
    from api.models import ResearchReport
    return ResearchReport.model_json_schema()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        reload=ENVIRONMENT == "development",
        log_level="debug" if ENVIRONMENT == "development" else "info",
    )