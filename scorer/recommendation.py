from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

load_dotenv()

# ── Env vars ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:   str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Scoring weights (default to equal-ish if not set)
WEIGHT_FOUNDER:     float = float(os.getenv("WEIGHT_FOUNDER",     "0.25"))
WEIGHT_FUNDING:     float = float(os.getenv("WEIGHT_FUNDING",     "0.20"))
WEIGHT_MARKET:      float = float(os.getenv("WEIGHT_MARKET",      "0.20"))
WEIGHT_FINANCIALS:  float = float(os.getenv("WEIGHT_FINANCIALS",  "0.15"))
WEIGHT_MACRO:       float = float(os.getenv("WEIGHT_MACRO",       "0.10"))
WEIGHT_NEWS:        float = float(os.getenv("WEIGHT_NEWS",        "0.10"))

# ── Safe fallback ────────────────────────────────────────────────────────────
_FALLBACK: dict = {
    "decision":        "WATCH",
    "confidence_score": 0,
    "reasoning":       "Unable to generate recommendation due to parsing error.",
    "strengths":       [],
    "risks":           ["Recommendation engine failed to produce valid output"],
}

# ── LLM singleton ────────────────────────────────────────────────────────────
_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        logger.debug(f"Initialising ChatGroq for scorer — model: {GROQ_MODEL}")
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=2048,
        )
    return _llm


# ── Prompt builder ───────────────────────────────────────────────────────────
def _build_prompt(company_name: str, raw_research: str) -> str:
    """Build the structured scoring prompt with weights embedded."""
    w_founder    = round(WEIGHT_FOUNDER    * 100, 1)
    w_funding    = round(WEIGHT_FUNDING    * 100, 1)
    w_market     = round(WEIGHT_MARKET     * 100, 1)
    w_financials = round(WEIGHT_FINANCIALS * 100, 1)
    w_macro      = round(WEIGHT_MACRO      * 100, 1)
    w_news       = round(WEIGHT_NEWS       * 100, 1)

    return f"""You are a senior partner at a pre-seed venture capital fund evaluating whether to invest in a startup.

You have been given comprehensive research on the company: {company_name}

Your task is to analyze the research and produce a structured investment recommendation.

SCORING WEIGHTS — apply these priorities when forming your assessment:
  • Founder quality & team composition : {w_founder}% importance
  • Funding history & investor signals  : {w_funding}% importance
  • Market size & competitive landscape : {w_market}% importance
  • Financial signals & revenue data    : {w_financials}% importance
  • Macro-economic conditions           : {w_macro}% importance
  • Recent news & momentum              : {w_news}% importance

DECISION THRESHOLDS:
  • INVEST  → Strong founder + large market + positive signals across most categories
  • PASS    → Weak team, saturated market, poor financials, or major red flags
  • WATCH   → Interesting but insufficient data, early traction, or mixed signals

RESEARCH DATA:
{raw_research}

OUTPUT RULES — CRITICAL:
- Respond with ONLY a valid JSON object. No markdown. No code blocks. No extra text.
- Use EXACTLY these keys: "decision", "confidence_score", "reasoning", "strengths", "risks"
- "decision"        : exactly one of "INVEST", "PASS", or "WATCH" (uppercase, string)
- "confidence_score": integer from 0 to 100
- "reasoning"       : 2-3 sentence plain-English summary of your recommendation
- "strengths"       : JSON array of 2-4 short strings (key positive signals)
- "risks"           : JSON array of 2-4 short strings (key concerns or red flags)

Example format:
{{"decision":"INVEST","confidence_score":72,"reasoning":"Strong repeat founder with prior exit in a large, growing market. Funding signals are positive and macro conditions are favourable for this sector.","strengths":["Serial founder with prior exit","$40B TAM with 18% YoY growth"],"risks":["No SEC filings — pre-revenue","Crowded competitive landscape"]}}"""


# ── JSON extractor ───────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict:
    """
    Three-strategy JSON extraction:
    1. Fenced code block  ```json ... ```
    2. First {...} block in the text
    3. Parse the whole text as-is
    """
    # Strategy 1 — fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Strategy 2 — largest {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group())

    # Strategy 3 — raw parse
    return json.loads(text.strip())


# ── Core function ────────────────────────────────────────────────────────────
def generate_recommendation(company_name: str, raw_research: str) -> dict:
    """
    Analyze raw VC research text for a given company and produce a structured
    investment recommendation using a Groq LLM.

    The function:
    - Sends a weighted scoring prompt to the Groq LLM (model from GROQ_MODEL env var)
    - Instructs the LLM to act as a pre-seed VC investment committee member
    - Parses the JSON response with a single automatic retry on parse failure
    - Returns a safe fallback dict if both attempts fail

    Args:
        company_name : Display name of the company being evaluated
        raw_research : Full compiled research string from the agent orchestrator

    Returns:
        dict with keys: company_name, decision, confidence_score,
                        reasoning, strengths, risks
    """
    llm = _get_llm()

    prompt_text = _build_prompt(company_name, raw_research)
    logger.info(f"[{company_name}] Sending scoring prompt to {GROQ_MODEL} "
                f"(research length: {len(raw_research)} chars)")
    logger.debug(f"[{company_name}] Weights — founder:{WEIGHT_FOUNDER} "
                 f"funding:{WEIGHT_FUNDING} market:{WEIGHT_MARKET} "
                 f"financials:{WEIGHT_FINANCIALS} macro:{WEIGHT_MACRO} "
                 f"news:{WEIGHT_NEWS}")

    messages = [
        SystemMessage(content=(
            "You are a pre-seed VC investment committee member. "
            "You always respond with ONLY valid JSON. No markdown, no prose."
        )),
        HumanMessage(content=prompt_text),
    ]

    # ── Attempt 1 ───────────────────────────────────────────────────────────
    try:
        response = llm.invoke(messages)
        raw_output: str = response.content.strip()
        logger.debug(f"[{company_name}] LLM response received "
                     f"({len(raw_output)} chars)")
        logger.debug(f"[{company_name}] Raw LLM output:\n{raw_output[:500]}")

        result = _extract_json(raw_output)
        logger.success(f"[{company_name}] JSON parsed on first attempt — "
                       f"decision: {result.get('decision')} | "
                       f"confidence: {result.get('confidence_score')}")
        result["company_name"] = company_name
        return result

    except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
        logger.warning(f"[{company_name}] JSON parse failed (attempt 1): {parse_err}")

    # ── Attempt 2 — stricter retry ───────────────────────────────────────────
    logger.info(f"[{company_name}] Retrying with stricter JSON instruction...")
    retry_messages = messages + [
        HumanMessage(content=(
            "Your previous response was not valid JSON. "
            "Respond with ONLY valid JSON, no markdown code blocks, no extra text. "
            "Return exactly this structure:\n"
            '{"decision":"INVEST|PASS|WATCH","confidence_score":0-100,'
            '"reasoning":"string","strengths":["str"],"risks":["str"]}'
        )),
    ]

    try:
        retry_response = llm.invoke(retry_messages)
        raw_retry: str = retry_response.content.strip()
        logger.debug(f"[{company_name}] Retry LLM response "
                     f"({len(raw_retry)} chars):\n{raw_retry[:500]}")

        result = _extract_json(raw_retry)
        logger.success(f"[{company_name}] JSON parsed on retry — "
                       f"decision: {result.get('decision')} | "
                       f"confidence: {result.get('confidence_score')}")
        result["company_name"] = company_name
        return result

    except (json.JSONDecodeError, ValueError, KeyError) as retry_err:
        logger.error(f"[{company_name}] JSON parse failed on retry too: {retry_err}")

    # ── Fallback ─────────────────────────────────────────────────────────────
    logger.error(f"[{company_name}] Returning safe fallback recommendation.")
    fallback = dict(_FALLBACK)
    fallback["company_name"] = company_name
    return fallback


# ── CLI smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    _company = sys.argv[1] if len(sys.argv) > 1 else "Mistral AI"
    _research = (
        "Founders: Arthur Mensch (ex-DeepMind), Guillaume Lample (ex-Meta AI). "
        "Strong technical pedigree. Raised €105M seed round — largest in EU history. "
        "Market: LLM / generative AI, $1.3T TAM by 2032. "
        "Competitors: OpenAI, Anthropic, Cohere, Google. "
        "Macro: tight monetary conditions but AI sector remains a VC priority. "
        "News: Mistral Large model released Q1 2024, partnership with Microsoft Azure."
    )
    rec = generate_recommendation(_company, _research)
    import pprint
    pprint.pprint(rec)
