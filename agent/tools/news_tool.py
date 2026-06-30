from __future__ import annotations

import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Singleton ────────────────────────────────────────────────────────────────

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://tavily.com"
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.debug("TavilyClient initialised for news_tool.")
    return _tavily_client


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(raw: str | None) -> datetime | None:
    """Try to parse a date string into a datetime object."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _tavily_news_search(company_name: str) -> list[dict]:
    """
    Search Tavily with topic='news' for the most recent 5 articles
    about the company. Returns a list of article dicts.
    """
    query = f"{company_name} latest news announcement"
    logger.info(f"[news_tool] Tavily news search → '{query}'")

    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            topic="news",
            max_results=5,
        )
        results = response.get("results", [])
        logger.info(f"[news_tool] Tavily returned {len(results)} news result(s).")
        return results

    except Exception as exc:
        logger.warning(f"[news_tool] Tavily news search failed: {exc}")
        return []


def _sort_by_date(articles: list[dict]) -> list[dict]:
    """Sort articles by published_date descending. Articles without dates go last."""
    def _sort_key(article: dict):
        dt = _parse_date(article.get("published_date"))
        # Return a sortable tuple: (has_date, datetime) — nulls go to end
        return (0, datetime.min) if dt is None else (1, dt)

    return sorted(articles, key=_sort_key, reverse=True)


def _format_article(index: int, article: dict) -> str:
    """Format a single news article into a clean readable string."""
    title   = article.get("title", "No title").strip()
    url     = article.get("url", "No URL").strip()
    snippet = article.get("content", article.get("snippet", "No snippet available.")).strip()
    raw_date = article.get("published_date")
    dt = _parse_date(raw_date)
    date_str = dt.strftime("%B %d, %Y") if dt else (raw_date or "Date unknown")

    # Truncate long snippets
    if len(snippet) > 300:
        snippet = snippet[:297] + "..."

    return (
        f"  {index}. {title}\n"
        f"     Date    : {date_str}\n"
        f"     Source  : {url}\n"
        f"     Snippet : {snippet}"
    )


def _build_summary(company_name: str, articles: list[dict]) -> str:
    """Assemble the final clean text summary."""
    header = (
        f"NEWS RESEARCH — {company_name.upper()}\n"
        + "─" * 60
    )

    if not articles:
        return (
            f"{header}\n\n"
            f"No recent news found for {company_name}."
        )

    sorted_articles = _sort_by_date(articles)
    lines = [header, f"\n[RECENT NEWS] — {len(sorted_articles)} article(s) found:\n"]
    for i, article in enumerate(sorted_articles, start=1):
        lines.append(_format_article(i, article))
        lines.append("")  # blank line between articles

    return "\n".join(lines).strip()


# ── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def research_news(company_name: str) -> str:
    """
    Search for the latest news and announcements about a company.

    Use this tool when you need to find:
    - Recent press releases or product launches
    - Funding announcements or acquisition news
    - Leadership changes (new CEO, co-founder departure)
    - Regulatory or legal developments
    - Industry coverage or media sentiment
    - Any recent notable events that could affect investment decisions

    Input  : company name (e.g. "Notion", "OpenAI", "Stripe")
    Output : Up to 5 most recent news articles sorted by date,
             each with title, publication date, source URL, and snippet.
             Returns 'No recent news found' if nothing is available.

    Data source: Tavily News Search API (real-time web).
    """
    logger.info(f"[news_tool] research_news called for: '{company_name}'")

    articles = _tavily_news_search(company_name)
    summary  = _build_summary(company_name, articles)

    logger.info(
        f"[news_tool] research_news complete. "
        f"{len(articles)} article(s) returned for '{company_name}'."
    )
    return summary


# ── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    company = sys.argv[1] if len(sys.argv) > 1 else "Mistral AI"
    print(research_news.invoke(company))