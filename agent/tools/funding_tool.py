from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

# ── Tavily singleton ────────────────────────────────────────────────────────

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _tavily_client = TavilyClient(api_key=api_key)
        logger.debug("Tavily client initialised for funding_tool.")
    return _tavily_client


# ── Layer 1: Tavily live web search ────────────────────────────────────────

def _tavily_search(company_name: str) -> str:
    """Search Tavily for funding rounds, valuation, and investor data."""
    query = f"{company_name} funding round valuation investors total raised"
    logger.info(f"[funding_tool] Tavily search → '{query}'")
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=6,
            include_answer=True,
        )

        parts: list[str] = []

        # Synthesised answer block
        answer = (response.get("answer") or "").strip()
        if answer:
            parts.append(f"Summary:\n{answer}")

        # Top source snippets
        results = response.get("results") or []
        if results:
            parts.append("Source Snippets:")
            for i, r in enumerate(results[:5], start=1):
                title   = (r.get("title")   or "").strip()
                content = (r.get("content") or "").strip()[:400]
                url     = (r.get("url")     or "").strip()
                parts.append(f"  {i}. {title}\n     {content}\n     Source: {url}")

        if not parts:
            return "No web funding data found via Tavily."

        return "\n\n".join(parts)

    except Exception as exc:
        logger.warning(f"[funding_tool] Tavily search failed: {exc}")
        return f"Tavily search unavailable: {exc}"


# ── Layer 2: RAG lookup from Crunchbase / VC CSV index ────────────────────

def _rag_funding_lookup(company_name: str) -> str:
    """Query the RAG pipeline for funding data indexed from Crunchbase/VC CSVs."""
    logger.info(f"[funding_tool] RAG lookup for funding data → '{company_name}'")
    try:
        from rag.pipeline import run_rag_query  # local import avoids circular deps
        rag_query = (
            f"funding rounds investment amount investors stage raised "
            f"valuation pre-seed seed Series A B C {company_name}"
        )
        result = run_rag_query(rag_query)
        return result.strip() if result else "No relevant funding data found in RAG index."
    except Exception as exc:
        logger.warning(f"[funding_tool] RAG lookup failed: {exc}")
        return f"RAG pipeline unavailable: {exc}"


# ── Layer 3: Combine both sources into a clean summary ────────────────────

def _build_summary(company_name: str, web_result: str, rag_result: str) -> str:
    """Merge Tavily web results and RAG index results into a structured summary."""
    return (
        f"=== FUNDING RESEARCH: {company_name.upper()} ===\n\n"
        f"── LIVE WEB DATA (Tavily) ──\n"
        f"{web_result}\n\n"
        f"── INDEXED FUNDING DATA (Crunchbase / VC CSV) ──\n"
        f"{rag_result}\n\n"
        f"=== END OF FUNDING RESEARCH ==="
    )


# ── LangChain Tool ─────────────────────────────────────────────────────────

@tool
def research_funding(company_name: str) -> str:
    """
    Research funding history, investment rounds, and valuation for a startup or company.

    Use this tool when you need to find:
    - Total funding raised (seed, Series A/B/C, etc.)
    - Individual funding round amounts and dates
    - Names of investors and venture capital firms involved
    - Company valuation at last known round
    - Funding stage (pre-seed, seed, growth, late-stage)
    - Any notable follow-on funding or recent raises

    This tool combines two data sources:
    1. Live web search via Tavily for the most recent funding announcements
    2. RAG-indexed data from Crunchbase and VC investment CSV datasets

    Args:
        company_name: The name of the startup or company to research funding for.

    Returns:
        A structured text summary of funding rounds, amounts, investors,
        and valuation data from both web and indexed sources.
    """
    logger.info(f"[funding_tool] Starting funding research for: '{company_name}'")

    web_result = _tavily_search(company_name)
    rag_result = _rag_funding_lookup(company_name)
    summary    = _build_summary(company_name, web_result, rag_result)

    logger.success(f"[funding_tool] Funding research complete for: '{company_name}'")
    return summary


# ── CLI smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "Stripe"
    print(research_funding.invoke(target))
