from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent          # LangChain 1.3+ unified API
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from loguru import logger

# ── 6 research tools ──────────────────────────────────────────────────────────
from agent.tools.founder_tool    import research_founder
from agent.tools.funding_tool    import research_funding
from agent.tools.financials_tool import research_financials
from agent.tools.market_tool     import research_market
from agent.tools.macro_tool      import research_macro_conditions
from agent.tools.news_tool       import research_news

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
PROMPTS_DIR  = Path(__file__).parent / "prompts"

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file.\n"
        "Free key: https://console.groq.com"
    )

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are Orion Alpha — an autonomous VC investment research analyst for pre-seed investors.

Your mission: given a company name (and optionally a URL or sector), run ALL 6 research
tools in order, then output a structured JSON investment recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY TOOL EXECUTION ORDER — call every tool, every time:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. research_founder(company_name)         → founder & team background
2. research_funding(company_name)         → funding rounds, investors, stage
3. research_financials(company_name)      → SEC filings, revenue, ARR signals
4. research_market(company_name)          → TAM/SAM, competitors, moat
5. research_macro_conditions(sector)      → FRED macro signals, climate label
6. research_news(company_name)            → recent news, sentiment

Do NOT skip any tool. Do NOT call a tool more than once.
Do NOT output the final JSON until all 6 tools have returned results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING WEIGHTS (composite_score = weighted sum, 0.0–10.0):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  founder:    0.25   (team quality is the #1 pre-seed signal)
  funding:    0.20   (investor validation and traction)
  market:     0.20   (TAM/SAM and competitive dynamics)
  financials: 0.15   (revenue signals and unit economics)
  macro:      0.10   (sector tailwinds and macro climate)
  news:       0.10   (recent momentum and sentiment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERDICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INVEST → composite_score ≥ 7.0  (HIGH or MEDIUM confidence)
  WATCH  → composite_score 4.5–6.9 (promising, needs maturity)
  PASS   → composite_score < 4.5   (clear red flags)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — after all 6 tools have run, return ONLY this JSON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "verdict":          "INVEST" | "PASS" | "WATCH",
  "confidence":       "HIGH" | "MEDIUM" | "LOW",
  "composite_score":  <float 0.0–10.0>,
  "one_line":         "<single sentence verdict>",
  "bull_case":        ["<reason 1>", "<reason 2>", "<reason 3>"],
  "bear_case":        ["<risk 1>",   "<risk 2>",   "<risk 3>"],
  "watch_triggers":   ["<trigger 1>", "<trigger 2>"],
  "full_reasoning":   "<2–3 paragraph reasoning grounded in tool outputs>",
  "signals": {
    "founder":    { "score": <0–10>, "summary": "<key findings>" },
    "funding":    { "score": <0–10>, "summary": "<key findings>" },
    "financials": { "score": <0–10>, "summary": "<key findings>" },
    "market":     { "score": <0–10>, "summary": "<key findings>" },
    "macro":      { "score": <0–10>, "summary": "<key findings>" },
    "news":       { "score": <0–10>, "summary": "<key findings>" }
  }
}

Rules:
- Never fabricate funding numbers or founder details.
- Cite specific data points from tool outputs in full_reasoning.
- If a tool returned no data, score that signal 0 and note "No data available."
- Return valid JSON only — no markdown, no extra text around it.
""".strip()


def _load_prompt_file(filename: str, fallback: str) -> str:
    """Load prompt from file; use inline fallback if file missing."""
    path = PROMPTS_DIR / filename
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        logger.debug(f"Loaded prompt: {path}")
        return content
    logger.warning(f"Prompt file not found ({path}) — using inline fallback.")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS LIST
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = [
    research_founder,
    research_funding,
    research_financials,
    research_market,
    research_macro_conditions,
    research_news,
]

_ALL_TOOL_NAMES = {t.name for t in _TOOLS}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT BUILDER  (LangChain 1.3 — create_agent)
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=4096,
    )


def _build_agent(llm: ChatGroq):
    """
    Build a LangChain 1.3 agent using create_agent.
    Returns a compiled LangGraph that accepts:
      {"messages": [{"role": "user", "content": "..."}]}
    """
    system_text = _load_prompt_file("system_prompt.txt", _SYSTEM_PROMPT)

    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=system_text,
    )


# ── Lazy singletons ───────────────────────────────────────────────────────────
_llm:   Optional[ChatGroq] = None
_agent                     = None   # compiled LangGraph


def _get_agent():
    global _llm, _agent
    if _agent is None:
        logger.info(f"Initialising Orion Alpha agent — model: {GROQ_MODEL}")
        _llm   = _build_llm()
        _agent = _build_agent(_llm)
        logger.success(f"Agent ready. {len(_TOOLS)} tools registered: {[t.name for t in _TOOLS]}")
    return _agent


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first valid JSON object from the agent's raw output.
    Tries three strategies:
      1. Fenced ```json … ``` block
      2. Largest { … } substring
      3. Whole text as JSON
    """
    # Strategy 1 — fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2 — largest braced block
    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3 — whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    logger.warning("Could not extract structured JSON from agent output.")
    return None


def _parse_messages(messages: list) -> tuple[str, List[str]]:
    """
    Extract (raw_output, tool_calls_made) from the LangGraph message list.

    - raw_output     : final AIMessage content string
    - tool_calls_made: deduplicated list of tool names that were invoked
    """
    raw_output      = ""
    tool_calls_made = []

    for msg in messages:
        # Collect tool names from AIMessage.tool_calls
        if isinstance(msg, AIMessage):
            raw_output = msg.content or ""          # last AIMessage wins
            for tc in getattr(msg, "tool_calls", []):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name and name not in tool_calls_made:
                    tool_calls_made.append(name)

    return raw_output, tool_calls_made


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_research(
    company_name:  str,
    url:           Optional[str] = None,
    sector_hint:   Optional[str] = None,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full Orion Alpha autonomous research pipeline for a target company.

    Calls all 6 tools (founder, funding, financials, market, macro, news),
    compiles findings, and returns a structured investment recommendation.

    Parameters
    ----------
    company_name  : Company name to research (required)
    url           : Optional company website URL
    sector_hint   : Optional sector/industry hint (e.g. "fintech", "healthtech")
    extra_context : Any additional context for the agent

    Returns
    -------
    dict with keys:
      research_id           — unique run UUID
      generated_at          — ISO 8601 UTC timestamp
      company_name          — input company name
      url                   — input URL (or None)
      sector                — input sector hint (or None)
      model_used            — Groq model string
      agent_steps           — total tool calls made
      tool_calls_made       — list of tool names actually invoked
      raw_agent_output      — final LLM text output
      parsed_recommendation — extracted JSON verdict (or None)
      all_messages          — full LangGraph message list
      error                 — error message if agent crashed (or None)
    """
    research_id  = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()

    # Build human message content
    parts = [f"Research target: {company_name}"]
    if url:
        parts.append(f"Website URL: {url}")
    if sector_hint:
        parts.append(f"Sector / industry: {sector_hint}")
    if extra_context:
        parts.append(f"Additional context: {extra_context}")
    parts.append(
        "\nRun all 6 research tools in order, then output the structured JSON recommendation."
    )
    human_input = "\n".join(parts)

    logger.info(f"[{research_id}] ── Orion Alpha research START: '{company_name}'")

    try:
        agent  = _get_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": human_input}]
        })

        all_messages              = result.get("messages", [])
        raw_output, tool_calls_made = _parse_messages(all_messages)
        agent_steps               = len(tool_calls_made)

        logger.info(
            f"[{research_id}] ── Research COMPLETE: {agent_steps} tool calls | "
            f"Tools: {tool_calls_made}"
        )

        # Warn on skipped tools
        skipped = _ALL_TOOL_NAMES - set(tool_calls_made)
        if skipped:
            logger.warning(
                f"[{research_id}] ⚠ Tools NOT called (agent skipped): {sorted(skipped)}"
            )

        parsed_recommendation = _extract_json(raw_output)
        if parsed_recommendation:
            logger.success(
                f"[{research_id}] Verdict: {parsed_recommendation.get('verdict')} | "
                f"Score: {parsed_recommendation.get('composite_score')}"
            )
        else:
            logger.warning(f"[{research_id}] Could not parse structured recommendation.")

        return {
            "research_id":           research_id,
            "generated_at":          generated_at,
            "company_name":          company_name,
            "url":                   url,
            "sector":                sector_hint,
            "model_used":            GROQ_MODEL,
            "agent_steps":           agent_steps,
            "tool_calls_made":       tool_calls_made,
            "raw_agent_output":      raw_output,
            "parsed_recommendation": parsed_recommendation,
            "all_messages":          all_messages,
            "error":                 None,
        }

    except Exception as exc:
        logger.error(f"[{research_id}] Agent crashed", exc_info=True)
        return {
            "research_id":           research_id,
            "generated_at":          generated_at,
            "company_name":          company_name,
            "url":                   url,
            "sector":                sector_hint,
            "model_used":            GROQ_MODEL,
            "agent_steps":           0,
            "tool_calls_made":       [],
            "raw_agent_output":      "",
            "parsed_recommendation": None,
            "all_messages":          [],
            "error":                 str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pprint import pprint

    _name   = sys.argv[1] if len(sys.argv) > 1 else "Mistral AI"
    _url    = sys.argv[2] if len(sys.argv) > 2 else None
    _sector = sys.argv[3] if len(sys.argv) > 3 else None

    out = run_research(company_name=_name, url=_url, sector_hint=_sector)

    print("\n" + "═" * 60)
    print(f"  ORION ALPHA — RESEARCH RESULT: {out['company_name']}")
    print("═" * 60)
    pprint({k: v for k, v in out.items() if k != "all_messages"})