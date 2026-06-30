from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger
from langchain.tools import tool
from fredapi import Fred

load_dotenv()

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ── FRED series to pull ────────────────────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "Federal Funds Rate":      "FEDFUNDS",
    "10-Year Treasury Yield":  "DGS10",
    "CPI Inflation Rate":      "CPIAUCSL",
    "Venture Capital Index":   "VENTUREX",   # may not exist — handled gracefully
}

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_fred_client: Fred | None = None


def _get_fred() -> Fred:
    global _fred_client
    if _fred_client is None:
        if not FRED_API_KEY:
            raise EnvironmentError(
                "FRED_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        _fred_client = Fred(api_key=FRED_API_KEY)
        logger.debug("FRED client initialised.")
    return _fred_client


# ── Fetch latest value for one FRED series ────────────────────────────────────
def _fetch_series_latest(series_name: str, series_id: str) -> dict:
    """
    Returns a dict with keys: series_name, series_id, value, date, status.
    Never raises — failed series return status='unavailable'.
    """
    try:
        fred = _get_fred()
        series = fred.get_series(series_id)
        # Drop NaN and get the last valid data point
        series_clean = series.dropna()
        if series_clean.empty:
            logger.warning(f"FRED series '{series_id}' returned no data.")
            return {
                "series_name": series_name,
                "series_id":   series_id,
                "value":       None,
                "date":        None,
                "status":      "no_data",
            }
        latest_date  = series_clean.index[-1]
        latest_value = float(series_clean.iloc[-1])
        logger.info(
            f"FRED | {series_name} ({series_id}): "
            f"{latest_value:.2f} as of {latest_date.strftime('%Y-%m-%d')}"
        )
        return {
            "series_name": series_name,
            "series_id":   series_id,
            "value":       latest_value,
            "date":        latest_date.strftime("%Y-%m-%d"),
            "status":      "ok",
        }
    except Exception as exc:
        logger.warning(f"FRED | Could not fetch '{series_id}': {exc}")
        return {
            "series_name": series_name,
            "series_id":   series_id,
            "value":       None,
            "date":        None,
            "status":      "unavailable",
        }


# ── Macro climate label logic ─────────────────────────────────────────────────
def _compute_climate_label(metrics: list[dict]) -> tuple[str, str]:
    """
    Returns (label, explanation) based on latest Fed Funds Rate and CPI.

    Rules:
      - Fed Funds > 5% AND CPI rising  → "Tight / Cautious"
      - Fed Funds < 3%                 → "Loose / Favorable"
      - Otherwise                      → "Neutral"
    """
    fed_funds_val: float | None = None
    cpi_val:       float | None = None

    for m in metrics:
        if m["series_id"] == "FEDFUNDS" and m["status"] == "ok":
            fed_funds_val = m["value"]
        if m["series_id"] == "CPIAUCSL" and m["status"] == "ok":
            cpi_val = m["value"]

    if fed_funds_val is None:
        return (
            "Neutral",
            "Fed Funds Rate data unavailable — defaulting to Neutral climate.",
        )

    # CPI "rising" heuristic: value above 300 index points (historical norm ~260–320)
    # This is a prototype-level heuristic; replace with YoY delta in production.
    cpi_elevated = (cpi_val is not None and cpi_val > 300)

    if fed_funds_val > 5.0 and cpi_elevated:
        label       = "🔴 Tight / Cautious"
        explanation = (
            f"Fed Funds Rate is high ({fed_funds_val:.2f}%) with elevated CPI "
            f"({cpi_val:.1f}). Borrowing costs are elevated — VCs typically "
            "become more selective and valuations compress in this environment."
        )
    elif fed_funds_val < 3.0:
        label       = "🟢 Loose / Favorable"
        explanation = (
            f"Fed Funds Rate is low ({fed_funds_val:.2f}%). Capital is cheap, "
            "risk appetite is typically higher, and VC deployment tends to "
            "accelerate in this macro environment."
        )
    else:
        label       = "🟡 Neutral"
        explanation = (
            f"Fed Funds Rate ({fed_funds_val:.2f}%) is in a moderate range. "
            "Macro conditions are neither strongly supportive nor restrictive "
            "for early-stage investment."
        )

    return label, explanation


# ── Build formatted output string ─────────────────────────────────────────────
def _build_summary(sector: str, metrics: list[dict], climate_label: str, climate_explanation: str) -> str:
    divider = "─" * 60
    lines   = [
        f"MACRO CONDITIONS RESEARCH — SECTOR: {sector.upper()}",
        divider,
        "",
        "[MACRO METRICS — FRED DATA]",
    ]

    for m in metrics:
        if m["status"] == "ok":
            lines.append(
                f"  • {m['series_name']} ({m['series_id']}): "
                f"{m['value']:.2f}  |  As of: {m['date']}"
            )
        else:
            lines.append(
                f"  • Data unavailable for {m['series_name']} ({m['series_id']})"
            )

    lines += [
        "",
        divider,
        "",
        "[MACRO CLIMATE ASSESSMENT]",
        f"  Overall Climate: {climate_label}",
        "",
        f"  {climate_explanation}",
        "",
        divider,
        "",
        "[SECTOR NOTE]",
        f"  Sector provided: '{sector}'",
        "  Note: In this prototype, macro conditions are reported at the macro-economy",
        "  level. Sector-specific overlays (e.g. SaaS vs. biotech vs. climate tech)",
        "  will be added in v2 using FRED sector-level indices and World Bank data.",
    ]

    return "\n".join(lines)


# ── LangChain Tool ─────────────────────────────────────────────────────────────
@tool
def research_macro_conditions(sector: str) -> str:
    """
    Research current macroeconomic conditions relevant to VC and startup investing.

    Use this tool when you need to:
    - Understand the current interest rate environment (Fed Funds Rate)
    - Assess inflation trends (CPI) and their impact on startup valuations
    - Get the 10-Year Treasury Yield as a risk-free rate benchmark
    - Determine the overall macro climate label: Tight/Cautious, Loose/Favorable, or Neutral
    - Understand whether macro conditions support or hinder early-stage investment

    Input:
        sector (str): The sector or industry of the company being researched
                      (e.g. 'fintech', 'healthtech', 'SaaS', 'climate tech').
                      In this prototype, macro data is economy-wide; sector is
                      logged for future sector-specific scoring.

    Returns:
        str: A structured summary of key FRED macro metrics, each with its
             latest value and date, plus an overall macro climate label
             and investment-context explanation.

    Data sources:
        - Federal Reserve Economic Data (FRED) via fredapi
        - Series: FEDFUNDS, DGS10, CPIAUCSL, VENTUREX (if available)
    """
    logger.info(f"research_macro_conditions called | sector='{sector}'")

    # Fetch all series
    metrics: list[dict] = []
    for series_name, series_id in FRED_SERIES.items():
        result = _fetch_series_latest(series_name, series_id)
        metrics.append(result)

    # Compute climate label
    climate_label, climate_explanation = _compute_climate_label(metrics)
    logger.info(f"Macro climate label: {climate_label}")

    # Build and return summary
    summary = _build_summary(sector, metrics, climate_label, climate_explanation)
    return summary


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sector_input = sys.argv[1] if len(sys.argv) > 1 else "fintech"
    print(research_macro_conditions.invoke(sector_input))
