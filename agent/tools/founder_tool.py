from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from tavily import TavilyClient

from rag.pipeline import run_rag_query

load_dotenv()

# ---------------------------------------------------------------------------
# Tavily client — lazy init so import doesn't fail if key is missing at
# module load time (e.g. during unit tests or orchestrator dry-runs)
# ---------------------------------------------------------------------------
_tavily: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file: TAVILY_API_KEY=tvly-xxxx"
            )
        _tavily = TavilyClient(api_key=api_key)
    return _tavily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int = 5) -> str:
    """Run a Tavily web search and return concatenated result snippets."""
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )

        parts: list[str] = []

        # Top-level synthesized answer (when available)
        if response.get("answer"):
            parts.append(f"[Web Summary]\n{response['answer']}")

        # Individual result snippets
        for i, result in enumerate(response.get("results", []), start=1):
            title   = result.get("title", "").strip()
            url     = result.get("url", "").strip()
            content = result.get("content", "").strip()
            if content:
                parts.append(f"[Source {i}] {title}\n{url}\n{content}")

        return "\n\n".join(parts) if parts else "No web results found."

    except Exception as exc:
        logger.warning(f"Tavily search failed for query '{query}': {exc}")
        return f"Web search unavailable: {exc}"


def _rag_founder_lookup(company_name: str) -> str:
    """Query the RAG pipeline for any indexed founder/people data."""
    try:
        rag_query = (
            f"founder co-founder team background education experience {company_name}"
        )
        result = run_rag_query(rag_query)
        return result if result else "No founder data found in indexed documents."
    except Exception as exc:
        logger.warning(f"RAG lookup failed for '{company_name}': {exc}")
        return f"RAG lookup unavailable: {exc}"


def _build_summary(
    company_name: str,
    web_results: str,
    rag_results: str,
) -> str:
    """Combine web + RAG results into a clean, labelled summary string."""
    separator = "-" * 60
    return (
        f"=== FOUNDER & TEAM RESEARCH: {company_name.upper()} ===\n\n"
        f"{separator}\n"
        f"[LIVE WEB RESEARCH — Tavily]\n"
        f"{separator}\n"
        f"{web_results}\n\n"
        f"{separator}\n"
        f"[INDEXED DATA — RAG / people.csv]\n"
        f"{separator}\n"
        f"{rag_results}\n"
    )


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def research_founder(company_name: str) -> str:
    """
    Research the founders and leadership team of a startup or company.

    Use this tool when you need to find:
    - Founder names and co-founder names
    - Professional backgrounds, previous roles, and career history
    - Past startup experience, exits, or notable achievements
    - Educational background (universities, degrees)
    - LinkedIn profiles or public bios
    - Team composition and key hires

    The tool combines:
    1. Live web search via Tavily (real-time LinkedIn, news, Crunchbase profiles)
    2. Indexed people/founder data from the local RAG pipeline (people.csv)

    Input:  company_name — the name of the company to research (e.g. "Stripe")
    Output: a structured text summary of founders, backgrounds, and experience.
    """
    logger.info(f"[founder_tool] Researching founders for: '{company_name}'")

    # --- Step 1: Tavily web search ---
    search_query = (
        f"{company_name} founder co-founder CEO background experience startup"
    )
    logger.debug(f"[founder_tool] Tavily query: '{search_query}'")
    web_results = _tavily_search(search_query)
    logger.debug(f"[founder_tool] Tavily returned {len(web_results)} chars")

    # --- Step 2: RAG lookup from people.csv ---
    logger.debug(f"[founder_tool] Running RAG lookup for '{company_name}'")
    rag_results = _rag_founder_lookup(company_name)
    logger.debug(f"[founder_tool] RAG returned {len(rag_results)} chars")

    # --- Step 3: Combine and return ---
    summary = _build_summary(company_name, web_results, rag_results)
    logger.success(
        f"[founder_tool] Founder research complete for '{company_name}' "
        f"({len(summary)} chars)"
    )
    return summary