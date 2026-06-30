from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Lazy singleton — one TavilyClient instance for the lifetime of the process
# ---------------------------------------------------------------------------
_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.debug("TavilyClient initialised for market_tool.")
    return _tavily_client


# ---------------------------------------------------------------------------
# Layer 1 — TAM / market size search
# ---------------------------------------------------------------------------
def _tavily_market_size(company_name: str) -> dict:
    """Search for market size / TAM information."""
    query = f"{company_name} market size TAM total addressable market industry"
    logger.info(f"[market_tool] TAM search → '{query}'")
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        return {
            "answer": response.get("answer", ""),
            "results": response.get("results", []),
        }
    except Exception as exc:
        logger.warning(f"[market_tool] TAM Tavily search failed: {exc}")
        return {"answer": "", "results": []}


# ---------------------------------------------------------------------------
# Layer 2 — Competitor / alternatives search
# ---------------------------------------------------------------------------
def _tavily_competitors(company_name: str) -> dict:
    """Search for competitors and similar companies."""
    query = f"{company_name} competitors alternatives similar companies"
    logger.info(f"[market_tool] Competitor search → '{query}'")
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=6,
        )
        return {
            "answer": response.get("answer", ""),
            "results": response.get("results", []),
        }
    except Exception as exc:
        logger.warning(f"[market_tool] Competitor Tavily search failed: {exc}")
        return {"answer": "", "results": []}


# ---------------------------------------------------------------------------
# Layer 3 — RAG lookup: similar companies already indexed from CSVs
# ---------------------------------------------------------------------------
def _rag_similar_companies(company_name: str) -> str:
    """Query ChromaDB for similar companies in the same sector."""
    query = (
        f"similar companies competitors same sector industry market "
        f"{company_name}"
    )
    logger.info(f"[market_tool] RAG similar-companies query → '{query}'")
    try:
        from rag.pipeline import run_rag_query
        result = run_rag_query(query)
        return result or "No similar companies found in indexed data."
    except Exception as exc:
        logger.warning(f"[market_tool] RAG lookup failed: {exc}")
        return "RAG lookup unavailable — run data/ingest.py first."


# ---------------------------------------------------------------------------
# Layer 4 — Combine all three sources into a structured summary
# ---------------------------------------------------------------------------
def _build_summary(
    company_name: str,
    market_data: dict,
    competitor_data: dict,
    rag_context: str,
) -> str:
    lines: list[str] = []
    sep = "─" * 60

    lines.append(f"MARKET RESEARCH — {company_name.upper()}")
    lines.append(sep)

    # ── [MARKET SIZE] ──────────────────────────────────────────
    lines.append("\n[MARKET SIZE]")
    if market_data["answer"]:
        lines.append(market_data["answer"])
    else:
        lines.append("No synthesized TAM answer available.")

    if market_data["results"]:
        lines.append("\nSources:")
        for i, r in enumerate(market_data["results"][:4], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", "")[:180].strip()
            lines.append(f"  {i}. {title}")
            lines.append(f"     {url}")
            if snippet:
                lines.append(f"     {snippet}…")

    lines.append(sep)

    # ── [COMPETITORS — WEB] ────────────────────────────────────
    lines.append("\n[COMPETITORS — WEB]")
    if competitor_data["answer"]:
        lines.append(competitor_data["answer"])
    else:
        lines.append("No synthesized competitor answer available.")

    if competitor_data["results"]:
        lines.append("\nSources:")
        for i, r in enumerate(competitor_data["results"][:5], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", "")[:180].strip()
            lines.append(f"  {i}. {title}")
            lines.append(f"     {url}")
            if snippet:
                lines.append(f"     {snippet}…")

    lines.append(sep)

    # ── [COMPETITORS — INDEXED DATA] ───────────────────────────
    lines.append("\n[COMPETITORS — INDEXED DATA]")
    lines.append(rag_context)

    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain tool — entry point for the agent
# ---------------------------------------------------------------------------
@tool
def research_market(company_name: str) -> str:
    """
    Research the market landscape for a given company.

    Use this tool when you need to find:
    - Total Addressable Market (TAM) and market size estimates
    - Industry growth rate and sector trends
    - Direct competitors and alternative solutions
    - Market positioning relative to similar companies
    - Comparable funded startups in the same space (from indexed Crunchbase / VC data)

    This tool runs TWO separate Tavily web searches:
      1. Market size / TAM / industry data
      2. Competitors and similar companies

    It also queries the RAG vector store for similar companies
    already indexed from the Crunchbase and VC funding CSVs.

    Results are returned in three clearly labelled sections:
    [MARKET SIZE], [COMPETITORS — WEB], [COMPETITORS — INDEXED DATA].

    Args:
        company_name: The name of the company or startup to research.

    Returns:
        A structured text summary covering market size, web competitors,
        and indexed comparable companies.
    """
    logger.info(f"[market_tool] Starting market research for: '{company_name}'")

    # Run both Tavily searches (independent — could be parallelised later)
    market_data = _tavily_market_size(company_name)
    competitor_data = _tavily_competitors(company_name)

    # Query RAG for similar indexed companies
    rag_context = _rag_similar_companies(company_name)

    # Build structured summary
    summary = _build_summary(company_name, market_data, competitor_data, rag_context)

    logger.success(f"[market_tool] Market research complete for '{company_name}'.")
    return summary


# ---------------------------------------------------------------------------
# CLI smoke-test: python -m agent.tools.market_tool "Notion"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "Figma"
    print(research_market.invoke(target))
