from __future__ import annotations

"""
output/report_builder.py
────────────────────────
Assembles the final structured investment research report for Orion Alpha.

Takes the raw agent research output + parsed recommendation dict and
combines them into one clean, validated dictionary ready to be returned
by the FastAPI /research endpoint or serialised to JSON.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MAX_SUMMARY_WORDS: int = 500

REQUIRED_RECOMMENDATION_KEYS: Dict[str, Any] = {
    "verdict":          "WATCH",
    "confidence":       0,
    "summary":          "No summary available.",
    "signals":          {},
    "reasoning":        "Recommendation data was incomplete — defaulting to WATCH.",
    "strengths":        [],
    "risks":            [],
    "bull_case":        "Insufficient data to determine bull case.",
    "bear_case":        "Insufficient data to determine bear case.",
    "watch_triggers":   [],
}

# Map verdict aliases to canonical values
_VERDICT_ALIASES: Dict[str, str] = {
    "invest":   "INVEST",
    "pass":     "PASS",
    "watch":    "WATCH",
    "hold":     "WATCH",
    "maybe":    "WATCH",
    "skip":     "PASS",
    "no":       "PASS",
    "yes":      "INVEST",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _truncate_text(text: str, max_words: int = MAX_SUMMARY_WORDS) -> str:
    """
    Truncate *text* to at most *max_words* words.

    If truncation occurs, appends '... [truncated]' to the result so
    downstream consumers know the text was cut.

    Args:
        text:      The string to truncate.
        max_words: Maximum word count before truncation (default 500).

    Returns:
        The original string if within limit, otherwise a truncated version
        with '... [truncated]' appended.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return f"{truncated}... [truncated]"


def _normalise_verdict(raw: Any) -> str:
    """Normalise a verdict value to INVEST / PASS / WATCH."""
    if not isinstance(raw, str):
        return "WATCH"
    normalised = raw.strip().lower()
    return _VERDICT_ALIASES.get(normalised, raw.strip().upper())


def _safe_list(value: Any) -> List[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        # Handle single string that should be a list
        return [value.strip()]
    return []


def _validate_recommendation(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that *recommendation* has all required keys.

    Missing or null keys are filled with safe defaults and a warning is
    logged for each gap found. Returns a clean, fully-populated dict.
    """
    if not isinstance(recommendation, dict):
        logger.warning(
            "recommendation is not a dict (got {t}) — using all defaults",
            t=type(recommendation).__name__,
        )
        return dict(REQUIRED_RECOMMENDATION_KEYS)

    validated: Dict[str, Any] = {}
    missing_keys: List[str] = []

    for key, default in REQUIRED_RECOMMENDATION_KEYS.items():
        raw_value = recommendation.get(key)

        # Treat None, empty string, and empty dict as missing
        if raw_value is None or raw_value == "" or raw_value == {}:
            validated[key] = default
            missing_keys.append(key)
        else:
            validated[key] = raw_value

    if missing_keys:
        logger.warning(
            "Recommendation dict missing/null keys: {keys} — filled with safe defaults",
            keys=missing_keys,
        )

    # Always normalise the verdict
    validated["verdict"] = _normalise_verdict(validated["verdict"])

    # Always coerce confidence to int 0–100
    try:
        conf = int(float(str(validated["confidence"])))
        validated["confidence"] = max(0, min(100, conf))
    except (ValueError, TypeError):
        logger.warning("confidence value '{v}' is not numeric — defaulting to 0", v=validated["confidence"])
        validated["confidence"] = 0

    # Coerce list fields
    validated["strengths"]      = _safe_list(validated.get("strengths", []))
    validated["risks"]          = _safe_list(validated.get("risks", []))
    validated["watch_triggers"] = _safe_list(validated.get("watch_triggers", []))

    return validated


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def build_report(
    company_name:    str,
    raw_research:    str,
    recommendation:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble the final structured investment research report.

    Combines the agent's raw research output with the parsed recommendation
    dict into a single validated dictionary that matches the ResearchReport
    Pydantic schema defined in api/models.py.

    Steps:
        1. Validate and fill-default the recommendation dict.
        2. Truncate raw_research to a 500-word research_summary.
        3. Build and return the full report dict.

    Args:
        company_name:    The target company that was researched.
        raw_research:    The full text output produced by the agent
                         (all tool outputs concatenated).
        recommendation:  The parsed JSON recommendation dict extracted
                         from the agent's final output. May be partial —
                         missing keys are filled with safe defaults.

    Returns:
        A dict with keys:
            company_name      — str
            decision          — "INVEST" | "PASS" | "WATCH"
            confidence_score  — int 0–100
            reasoning         — str
            strengths         — list[str]
            risks             — list[str]
            bull_case         — str
            bear_case         — str
            watch_triggers    — list[str]
            research_summary  — str (max 500 words, truncated if longer)
            full_research     — str (complete untruncated raw_research)
            generated_at      — ISO 8601 UTC timestamp str
    """
    logger.info("Building report for '{company}'", company=company_name)

    # ── Step 1: validate recommendation ──────────────────────────────
    rec = _validate_recommendation(recommendation)

    # ── Step 2: truncate raw research to summary ──────────────────────
    clean_research  = (raw_research or "").strip()
    research_summary = _truncate_text(clean_research, MAX_SUMMARY_WORDS)

    if research_summary.endswith("... [truncated]"):
        logger.debug(
            "research_summary truncated to {n} words for '{company}'",
            n=MAX_SUMMARY_WORDS,
            company=company_name,
        )

    # ── Step 3: assemble report ───────────────────────────────────────
    report: Dict[str, Any] = {
        "company_name":     company_name.strip(),
        "decision":         rec["verdict"],
        "confidence_score": rec["confidence"],
        "reasoning":        str(rec.get("reasoning") or rec.get("summary") or "").strip(),
        "strengths":        rec["strengths"],
        "risks":            rec["risks"],
        "bull_case":        str(rec.get("bull_case", "")).strip(),
        "bear_case":        str(rec.get("bear_case", "")).strip(),
        "watch_triggers":   rec["watch_triggers"],
        "research_summary": research_summary,
        "full_research":    clean_research,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
    }

    logger.success(
        "Report built — company: '{company}' | decision: {decision} | confidence: {conf}%",
        company=report["company_name"],
        decision=report["decision"],
        conf=report["confidence_score"],
    )

    return report


# ─────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    _mock_raw = " ".join([f"Word{i}" for i in range(600)])  # 600 words → triggers truncation

    _mock_rec = {
        "verdict":        "INVEST",
        "confidence":     78,
        "summary":        "Strong founding team with prior exits. TAM > $10B. Seed round led by Sequoia.",
        "reasoning":      "The company shows strong signals across founder quality, market size, and early traction.",
        "strengths":      ["Experienced founding team", "Large and growing TAM", "Strong lead investor"],
        "risks":          ["Pre-revenue", "Crowded market", "Regulatory uncertainty"],
        "bull_case":      "Becomes category leader in a $10B+ market within 5 years.",
        "bear_case":      "Fails to differentiate from incumbents and burns through runway.",
        "watch_triggers": ["Revenue milestone", "Series A close", "Key hire announcement"],
        "signals":        {},
    }

    _report = build_report(
        company_name="Mistral AI",
        raw_research=_mock_raw,
        recommendation=_mock_rec,
    )

    print(json.dumps(_report, indent=2))

    # Test missing keys path
    print("\n── Testing missing keys path ──")
    _partial_rec = {"verdict": "pass", "confidence": "eighty"}
    _report2 = build_report(
        company_name="Unknown Startup",
        raw_research="Short research text.",
        recommendation=_partial_rec,
    )
    print(json.dumps(_report2, indent=2))