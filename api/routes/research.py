from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict
import re

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from loguru import logger

load_dotenv()

# ── Internal imports ──────────────────────────────────────────────────────────
from agent.orchestrator import run_research
from agent.tools.founder_tool import research_founder
from agent.tools.funding_tool import research_funding
from agent.tools.financials_tool import research_financials
from agent.tools.market_tool import research_market
from agent.tools.macro_tool import research_macro_conditions
from agent.tools.news_tool import research_news
from scorer.recommendation import generate_recommendation
from api.models import (
    ErrorResponse,
    FinancialsSignal,
    FounderSignal,
    FundingSignal,
    MacroSignal,
    MarketSignal,
    NewsSignal,
    Recommendation,
    ResearchRequest,
    ResearchReport,
    SignalScores,
)

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


def _safe_invoke(tool, argument: str) -> str:
    try:
        return str(tool.invoke(argument)).strip()
    except Exception as exc:
        logger.warning(f"Tool {getattr(tool, 'name', tool)} failed: {exc}")
        return f"Tool unavailable: {exc}"


def _signal_score(text: str) -> float:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()

    if not cleaned:
        return 0.0

    if any(marker in lowered for marker in ["unavailable", "no data available", "no relevant", "not found"]):
        return 0.0

    score = 1.5
    if re.search(r"\d", cleaned):
        score += 1.0
    if any(marker in lowered for marker in ["found", "raised", "investor", "funding", "market", "sec", "news", "cpi", "fed", "competitor", "founder"]):
        score += 2.0
    if len(cleaned) > 500:
        score += 0.5

    return round(min(10.0, score), 1)


def _derive_funding_signal(text: str) -> FundingSignal:
    lowered = text.lower()
    
    # Remove the tool's own description text to avoid false "pre-seed" matches
    cleaned = lowered.replace("pre-seed analyst", "").replace("pre-seed investor", "")
    
    # Search from most mature stage to earliest — pick the LATEST stage mentioned
    stage = None
    for candidate in ["ipo", "growth", "series-d", "series-c", "series-b", "series-a", "seed", "pre-seed"]:
        if candidate in cleaned or candidate.replace("-", " ") in cleaned:
            stage = candidate
            break

    amount_match = re.search(r"(?:\$|usd\s*)([\d,.]+)\s*([kmb])?", text, re.IGNORECASE)
    total_raised = None
    if amount_match:
        raw_amount = float(amount_match.group(1).replace(",", ""))
        suffix = (amount_match.group(2) or "").lower()
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
        total_raised = raw_amount * multiplier

    crunchbase_match = "crunchbase" in lowered and not any(marker in lowered for marker in ["no relevant", "unavailable"])
    
    # Extract investor names from text
    investors = []
    investor_patterns = re.findall(r"(?:investor|led by|backed by|from)\s+([A-Z][A-Za-z\s&]+?)(?:[,.\n]|and )", text)
    for inv in investor_patterns[:5]:
        cleaned_inv = inv.strip()
        if len(cleaned_inv) > 2 and len(cleaned_inv) < 50:
            investors.append(cleaned_inv)

    return FundingSignal(
        total_raised_usd=total_raised,
        last_round_stage=stage or "unknown",
        investors=investors,
        crunchbase_match=crunchbase_match,
        raw_summary=text or None,
        score=_signal_score(text),
    )


def _derive_founded_signal(text: str) -> FounderSignal:
    return FounderSignal(
        names=[],
        backgrounds=[],
        notable_exits=[],
        linkedin_urls=[],
        raw_summary=text or None,
        score=_signal_score(text),
    )


def _derive_financials_signal(text: str) -> FinancialsSignal:
    lowered = text.lower()
    return FinancialsSignal(
        is_public=any(marker in lowered for marker in ["10-k", "s-1", "public company", "sec filings"]),
        sec_filings_found="sec filings" in lowered and "no sec filings" not in lowered,
        raw_summary=text or None,
        score=_signal_score(text),
    )


def _derive_market_signal(text: str) -> MarketSignal:
    return MarketSignal(
        raw_summary=text or None,
        score=_signal_score(text),
    )


def _derive_macro_signal(text: str, sector: str | None) -> MacroSignal:
    lowered = text.lower()
    return MacroSignal(
        sector=sector,
        interest_rate_env=("tight / cautious" if "tight / cautious" in lowered else "loose / favorable" if "loose / favorable" in lowered else "neutral"),
        inflation_signal=("elevated" if "cpi" in lowered and "high" in lowered else None),
        raw_summary=text or None,
        score=_signal_score(text),
    )


def _derive_news_signal(text: str) -> NewsSignal:
    headlines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isdigit() and ". " in stripped:
            headlines.append(stripped.split(". ", 1)[1].strip())
    return NewsSignal(
        headlines=headlines[:5],
        raw_summary=text or None,
        score=_signal_score(text),
    )


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

    # ── Direct tool fallback / structured signal assembly ────────────────────
    founder_text = _safe_invoke(research_founder, company_name)
    funding_text = _safe_invoke(research_funding, company_name)
    financials_text = _safe_invoke(research_financials, company_name)
    market_text = _safe_invoke(research_market, company_name)
    macro_input = request.sector_hint or company_name
    macro_text = _safe_invoke(research_macro_conditions, macro_input)
    news_text = _safe_invoke(research_news, company_name)

    structured_research = "\n\n".join([
        founder_text,
        funding_text,
        financials_text,
        market_text,
        macro_text,
        news_text,
    ]).strip()

    if not raw_research.strip():
        raw_research = structured_research
    else:
        raw_research = f"{raw_research.strip()}\n\n{structured_research}"

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
        founder_signal = _derive_founded_signal(founder_text)
        funding_signal = _derive_funding_signal(funding_text)
        financials_signal = _derive_financials_signal(financials_text)
        market_signal = _derive_market_signal(market_text)
        macro_signal = _derive_macro_signal(macro_text, request.sector_hint)
        news_signal = _derive_news_signal(news_text)

        scores = SignalScores(
            founder=founder_signal.score,
            funding=funding_signal.score,
            financials=financials_signal.score,
            market=market_signal.score,
            macro=macro_signal.score,
            news=news_signal.score,
            composite=round(
                (
                    founder_signal.score * 0.25
                    + funding_signal.score * 0.20
                    + financials_signal.score * 0.15
                    + market_signal.score * 0.20
                    + macro_signal.score * 0.10
                    + news_signal.score * 0.10
                ),
                2,
            ),
        )

        report = ResearchReport(
            company_name=company_name,
            url=request.url,
            sector=request.sector_hint,
            research_id=research_id,
            generated_at=research_result.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            model_used=research_result.get("model_used", "unknown"),
            agent_steps=research_result.get("agent_steps", 0),
            founder=founder_signal,
            funding=funding_signal,
            financials=financials_signal,
            market=market_signal,
            macro=macro_signal,
            news=news_signal,
            scores=scores,
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
