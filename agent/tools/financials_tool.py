from __future__ import annotations

import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
SEC_SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"
SEC_TIMEOUT      = 10          # seconds
MAX_FILINGS      = 5           # max filings to surface in summary
MAX_WEB_RESULTS  = 5

# ── Lazy Tavily client ────────────────────────────────────────────────────────
_tavily_client: TavilyClient | None = None

def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set in .env — required for financial web fallback."
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.debug("Tavily client initialised for financials_tool.")
    return _tavily_client


# ── SEC EDGAR full-text search ─────────────────────────────────────────────────
def _query_sec_edgar(company_name: str) -> list[dict]:
    """
    Queries SEC EDGAR full-text search for 10-K and S-1 filings.
    Returns a list of dicts: {filing_type, filing_date, company, accession, link}
    Returns an empty list on any error or timeout.
    """
    params = {
        "q":     company_name,
        "forms": "10-K,S-1",
    }
    try:
        logger.info(f"[SEC EDGAR] Querying filings for: '{company_name}'")
        resp = requests.get(
            SEC_SEARCH_URL,
            params=params,
            timeout=SEC_TIMEOUT,
            headers={"User-Agent": "OrionAlpha/1.0 research@orionalpha.ai"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"[SEC EDGAR] Request timed out after {SEC_TIMEOUT}s for '{company_name}'.")
        return []
    except requests.exceptions.ConnectionError as exc:
        logger.warning(f"[SEC EDGAR] Connection error for '{company_name}': {exc}")
        return []
    except requests.exceptions.HTTPError as exc:
        logger.warning(f"[SEC EDGAR] HTTP error {resp.status_code} for '{company_name}': {exc}")
        return []
    except Exception as exc:
        logger.warning(f"[SEC EDGAR] Unexpected error for '{company_name}': {exc}")
        return []

    # Parse hits — EDGAR returns {"hits": {"hits": [...]}}
    hits = data.get("hits", {}).get("hits", [])
    filings: list[dict] = []
    for hit in hits[:MAX_FILINGS]:
        src = hit.get("_source", {})
        accession = src.get("accession_no", "").replace("-", "")
        cik        = src.get("entity_id", "")
        link = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type={src.get('form_type', '')}&dateb=&owner=include&count=10"
            if cik else "https://www.sec.gov/cgi-bin/browse-edgar"
        )
        filings.append({
            "filing_type":  src.get("form_type", "N/A"),
            "filing_date":  src.get("file_date",  "N/A"),
            "company":      src.get("display_names", company_name),
            "accession":    accession or "N/A",
            "link":         link,
        })

    logger.info(f"[SEC EDGAR] Found {len(filings)} filings for '{company_name}'.")
    return filings


def _format_sec_section(company_name: str, filings: list[dict]) -> str:
    """Formats the SEC filings list into a clean labelled section string."""
    if not filings:
        return (
            "[SEC FILINGS]\n"
            f"No SEC filings found — company likely private/pre-IPO, "
            f"relying on web signals only.\n"
        )
    lines = [f"[SEC FILINGS] — {len(filings)} filing(s) found for '{company_name}':"]
    for i, f in enumerate(filings, 1):
        lines.append(
            f"  {i}. Type: {f['filing_type']} | Date: {f['filing_date']} "
            f"| Company: {f['company']}\n"
            f"     Link: {f['link']}"
        )
    return "\n".join(lines)


# ── Tavily financial web fallback ─────────────────────────────────────────────
def _tavily_financials_search(company_name: str) -> str:
    """
    Falls back to Tavily for revenue/ARR/growth signals when SEC has no filings.
    Always runs to supplement SEC data with live web signals.
    """
    query = f"{company_name} revenue ARR growth financial signals"
    logger.info(f"[Tavily] Searching financial signals: '{query}'")
    try:
        client = _get_tavily()
        resp   = client.search(
            query=query,
            search_depth="advanced",
            max_results=MAX_WEB_RESULTS,
            include_answer=True,
        )
        parts: list[str] = []
        if resp.get("answer"):
            parts.append(f"Summary: {resp['answer']}")
        for r in resp.get("results", [])[:MAX_WEB_RESULTS]:
            title   = r.get("title",   "No title")
            url     = r.get("url",     "")
            snippet = r.get("content", "")[:280]
            parts.append(f"• {title}\n  {url}\n  {snippet}")
        return "\n".join(parts) if parts else "No web financial signals found."
    except Exception as exc:
        logger.warning(f"[Tavily] Financial search failed for '{company_name}': {exc}")
        return "Web financial signal search unavailable."


# ── Final summary builder ──────────────────────────────────────────────────────
def _build_summary(
    company_name: str,
    sec_section: str,
    web_section: str,
) -> str:
    divider = "─" * 60
    return (
        f"FINANCIAL RESEARCH — {company_name.upper()}\n"
        f"{divider}\n\n"
        f"{sec_section}\n\n"
        f"{divider}\n\n"
        f"[WEB FINANCIAL SIGNALS]\n"
        f"{web_section}\n\n"
        f"{divider}\n"
        f"Note: SEC filings only exist for public companies. "
        f"Pre-seed / private companies will show web signals only."
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────
@tool
def research_financials(company_name: str) -> str:
    """
    Research the financial profile of a company for VC investment analysis.

    Use this tool when you need:
    - SEC 10-K or S-1 filings (public companies only)
    - Revenue, ARR, or MRR estimates
    - Growth rate signals
    - Profitability indicators
    - Public financial disclosures

    This tool first queries the SEC EDGAR full-text search API for any
    10-K or S-1 filings. If no filings are found (common for pre-seed /
    private companies), it falls back to a live Tavily web search for
    revenue signals, ARR, and financial growth indicators.

    Both SEC filing links and web signals are returned in clearly labelled
    sections: [SEC FILINGS] and [WEB FINANCIAL SIGNALS].

    Args:
        company_name: The name of the company to research (e.g. "Stripe")

    Returns:
        A structured text summary with SEC filing details and/or web-sourced
        financial signals, clearly labelled for the investment scoring agent.
    """
    logger.info(f"[financials_tool] Starting financial research for: '{company_name}'")

    # Step 1 — Query SEC EDGAR
    filings     = _query_sec_edgar(company_name)
    sec_section = _format_sec_section(company_name, filings)

    # Step 2 — Always run Tavily for live financial signals
    web_section = _tavily_financials_search(company_name)

    # Step 3 — Combine and return
    summary = _build_summary(company_name, sec_section, web_section)
    logger.success(f"[financials_tool] Financial research complete for '{company_name}'.")
    return summary


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "Stripe"
    print(research_financials.invoke(target))
