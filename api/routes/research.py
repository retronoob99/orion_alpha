from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from loguru import logger

load_dotenv()

# ── Internal imports ──────────────────────────────────────────────────────────
from agent.orchestrator import run_research
from scorer.recommendation import generate_recommendation
from api.models import ErrorResponse, Recommendation, ResearchRequest, ResearchReport

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/research",
    tags=["research"],
)


def _confidence_label(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "LOW"

    if value >= 70:
        return "HIGH"
    if value >= 45:
        return "MEDIUM"
    return "LOW"


# ── POST / ────────────────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=ResearchReport,
    summary="Run full investment research on a company",
    responses={
        500: {"model": ErrorResponse, "description": "Internal pipeline error"},
    },
)
async def research_company(request: ResearchRequest) -> Dict[str, Any]:
    """
    Full Orion Alpha investment research pipeline.

    Accepts a company name (and optional URL / sector hint / extra context),
    then runs three sequential stages:

    1. **Research stage** — LangChain agent calls all 6 tools (founder,
       funding, financials, market, macro, news) and returns a raw
       research blob.
    2. **Scoring stage** — Groq LLM analyses the raw research against
       configurable weight factors and produces an INVEST / PASS / WATCH
       decision with a 0-100 confidence score.
    3. **Report stage** — Combines research + recommendation into a single
       structured JSON report matching the ResearchReport schema.

    Returns the fully assembled report as JSON.
    """
    company_name: str = request.company_name
    start_time = time.time()

    logger.info(
        f"[REQUEST START] company='{company_name}' | "
        f"url={request.url!r} | sector={request.sector_hint!r}"
    )

    # ── Stage 1: Research ─────────────────────────────────────────────────────
    try:
        research_result: Dict[str, Any] = run_research(
            company_name=company_name,
            url=str(request.url) if request.url else None,
            sector_hint=request.sector_hint,
        )
        raw_research: str = research_result.get("raw_agent_output", "")
        tools_called = research_result.get("tool_calls_made", [])
        research_id = research_result.get("research_id", "unknown")

        logger.info(
            f"[STAGE 1 ✓] research complete | id={research_id} | "
            f"tools_called={tools_called}"
        )

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        logger.error(
            f"[STAGE 1 ✗] Research stage failed for '{company_name}' "
            f"after {elapsed}s — {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Research stage failed for '{company_name}': {exc}. "
                "Check GROQ_API_KEY, TAVILY_API_KEY, and FRED_API_KEY."
            ),
        )

    # ── Stage 2: Scoring / Recommendation ─────────────────────────────────────
    try:
        recommendation: Dict[str, Any] = generate_recommendation(
            company_name=company_name,
            raw_research=raw_research,
        )
        logger.info(
            f"[STAGE 2 ✓] scoring complete | "
            f"decision={recommendation.get('decision')} | "
            f"confidence={recommendation.get('confidence_score')}"
        )

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        logger.error(
            f"[STAGE 2 ✗] Scoring stage failed for '{company_name}' "
            f"after {elapsed}s — {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Scoring stage failed for '{company_name}': {exc}. "
                "Raw research was collected but recommendation could not be generated."
            ),
        )

    # ── Stage 3: Report Building ───────────────────────────────────────────────
    try:
        confidence_score = recommendation.get("confidence_score", 0)
        report = ResearchReport(
            company_name=company_name,
            url=request.url,
            sector=request.sector_hint,
            research_id=research_id,
            generated_at=research_result.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            model_used=research_result.get("model_used", "unknown"),
            agent_steps=research_result.get("agent_steps", 0),
            raw_agent_output=raw_research,
            recommendation=Recommendation(
                verdict=recommendation.get("decision", "WATCH"),
                confidence=_confidence_label(confidence_score),
                composite_score=max(0.0, min(10.0, float(confidence_score) / 10.0 if confidence_score is not None else 0.0)),
                one_line=(recommendation.get("reasoning") or "").split(".")[0].strip() or "No summary available.",
                bull_case=list(recommendation.get("strengths", [])),
                bear_case=list(recommendation.get("risks", [])),
                watch_triggers=list(recommendation.get("watch_triggers", [])),
                full_reasoning=recommendation.get("reasoning", ""),
            ),
        )
        logger.info(f"[STAGE 3 ✓] report built | verdict={report.recommendation.verdict}")

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        logger.error(
            f"[STAGE 3 ✗] Report builder failed for '{company_name}' "
            f"after {elapsed}s — {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Report building stage failed for '{company_name}': {exc}. "
                "Research and scoring completed — only report assembly failed."
            ),
        )

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    logger.info(
        f"[REQUEST END] company='{company_name}' | "
        f"verdict={report.recommendation.verdict} | "
        f"confidence={report.recommendation.confidence} | "
        f"total_time={elapsed}s"
    )

    return report.model_dump()


# ── GET /health ────────────────────────────────────────────────────────────────
@router.get(
    "/health",
    summary="Research service health check",
)
async def research_health() -> Dict[str, str]:
    """
    Lightweight health check for the research router.

    Returns a static JSON payload confirming the research service
    is mounted and reachable. Does not test downstream dependencies
    (LLM, ChromaDB, Tavily) — use the root /health endpoint for
    full app status.
    """
    return {"status": "ok", "service": "research"}
