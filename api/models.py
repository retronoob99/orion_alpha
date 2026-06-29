from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator
import re


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    INVEST  = "INVEST"
    PASS    = "PASS"
    WATCH   = "WATCH"


class ConfidenceLevel(str, Enum):
    HIGH    = "HIGH"
    MEDIUM  = "MEDIUM"
    LOW     = "LOW"


class FundingStage(str, Enum):
    PRE_SEED    = "pre-seed"
    SEED        = "seed"
    SERIES_A    = "series-a"
    SERIES_B    = "series-b"
    SERIES_C    = "series-c"
    SERIES_D    = "series-d"
    GROWTH      = "growth"
    IPO         = "ipo"
    UNKNOWN     = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """
    Input payload for POST /research.
    At minimum, company_name is required.
    """
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the company to research.",
        examples=["Stripe", "OpenAI", "Notion"],
    )
    url: Optional[str] = Field(
        default=None,
        description="Company website or LinkedIn URL (optional but improves research quality).",
        examples=["https://stripe.com"],
    )
    extra_context: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Any additional context the user wants the agent to consider.",
        examples=["Focus on their expansion into LATAM markets."],
    )
    sector_hint: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Industry / sector hint to guide macro scoring (e.g. 'fintech', 'health tech').",
        examples=["fintech", "climate tech", "B2B SaaS"],
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not re.match(r"^https?://", v):
            v = "https://" + v
        return v

    @field_validator("company_name")
    @classmethod
    def strip_company_name(cls, v: str) -> str:
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "company_name": "Stripe",
                "url": "https://stripe.com",
                "sector_hint": "fintech",
                "extra_context": "Interested in their recent expansion into Africa.",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-REPORT SCHEMAS  (one per agent tool)
# ─────────────────────────────────────────────────────────────────────────────

class FounderSignal(BaseModel):
    """Output from founder_tool."""
    names: List[str] = Field(default_factory=list, description="Founder / co-founder names.")
    backgrounds: List[str] = Field(default_factory=list, description="Brief background per founder.")
    team_size_estimate: Optional[str] = Field(default=None, description="Estimated team headcount.")
    notable_exits: List[str] = Field(default_factory=list, description="Prior exits or notable companies.")
    linkedin_urls: List[str] = Field(default_factory=list)
    raw_summary: Optional[str] = Field(default=None, description="LLM-generated founder summary.")
    score: float = Field(default=0.0, ge=0.0, le=10.0, description="Founder signal score 0–10.")


class FundingSignal(BaseModel):
    """Output from funding_tool."""
    total_raised_usd: Optional[float] = Field(default=None, description="Total capital raised in USD.")
    last_round_stage: Optional[FundingStage] = Field(default=FundingStage.UNKNOWN)
    last_round_amount_usd: Optional[float] = Field(default=None)
    last_round_date: Optional[str] = Field(default=None, description="ISO date string e.g. 2023-06-01.")
    investors: List[str] = Field(default_factory=list, description="Known investor names.")
    num_rounds: Optional[int] = Field(default=None)
    crunchbase_match: bool = Field(default=False, description="Was a Crunchbase CSV match found?")
    raw_summary: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


class FinancialsSignal(BaseModel):
    """Output from financials_tool (SEC EDGAR + web revenue signals)."""
    is_public: bool = Field(default=False)
    sec_filings_found: bool = Field(default=False)
    revenue_estimate: Optional[str] = Field(default=None, description="Revenue range or estimate if available.")
    profitability_signal: Optional[str] = Field(default=None, description="Profitable / burning / unknown.")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Any extracted KPIs.")
    raw_summary: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


class MarketSignal(BaseModel):
    """Output from market_tool."""
    market_size_estimate: Optional[str] = Field(default=None, description="TAM/SAM estimate if found.")
    growth_rate_estimate: Optional[str] = Field(default=None)
    top_competitors: List[str] = Field(default_factory=list)
    competitive_moat: Optional[str] = Field(default=None, description="Described competitive advantage.")
    raw_summary: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


class MacroSignal(BaseModel):
    """Output from macro_tool (FRED API + sector conditions)."""
    sector: Optional[str] = Field(default=None)
    interest_rate_env: Optional[str] = Field(default=None, description="e.g. 'high / rising / stable'.")
    inflation_signal: Optional[str] = Field(default=None)
    sector_health_score: float = Field(default=5.0, ge=0.0, le=10.0)
    fred_series_used: List[str] = Field(default_factory=list, description="FRED series IDs pulled.")
    raw_summary: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


class NewsSignal(BaseModel):
    """Output from news_tool."""
    headlines: List[str] = Field(default_factory=list, description="Top recent headlines.")
    sentiment: Optional[str] = Field(default=None, description="positive / neutral / negative.")
    sentiment_score: float = Field(default=5.0, ge=0.0, le=10.0, description="0=very negative, 10=very positive.")
    key_events: List[str] = Field(default_factory=list, description="Notable recent events.")
    raw_summary: Optional[str] = Field(default=None)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING + RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

class SignalScores(BaseModel):
    """Weighted component scores feeding the final recommendation."""
    founder:    float = Field(default=0.0, ge=0.0, le=10.0)
    funding:    float = Field(default=0.0, ge=0.0, le=10.0)
    financials: float = Field(default=0.0, ge=0.0, le=10.0)
    market:     float = Field(default=0.0, ge=0.0, le=10.0)
    macro:      float = Field(default=0.0, ge=0.0, le=10.0)
    news:       float = Field(default=0.0, ge=0.0, le=10.0)
    composite:  float = Field(default=0.0, ge=0.0, le=10.0, description="Weighted composite 0–10.")


class Recommendation(BaseModel):
    """Final Invest / Pass / Watch decision with full reasoning."""
    verdict:        Verdict         = Field(..., description="INVEST | PASS | WATCH")
    confidence:     ConfidenceLevel = Field(..., description="HIGH | MEDIUM | LOW")
    composite_score: float          = Field(..., ge=0.0, le=10.0)
    one_line:       str             = Field(..., description="Single-sentence verdict summary.")
    bull_case:      List[str]       = Field(default_factory=list, description="Top reasons to invest.")
    bear_case:      List[str]       = Field(default_factory=list, description="Top risks / reasons to pass.")
    watch_triggers: List[str]       = Field(
        default_factory=list,
        description="Conditions that would change the verdict (for WATCH or borderline cases).",
    )
    full_reasoning: str             = Field(..., description="Full LLM-generated reasoning paragraph.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RESPONSE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class ResearchReport(BaseModel):
    """
    Complete structured output returned by POST /research.
    Assembles all agent signals + final recommendation.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    company_name:   str             = Field(..., description="Company name as provided in request.")
    url:            Optional[str]   = Field(default=None)
    sector:         Optional[str]   = Field(default=None)
    research_id:    str             = Field(..., description="Unique UUID for this research run.")
    generated_at:   str             = Field(..., description="ISO 8601 UTC timestamp.")

    # ── LLM / Model metadata ─────────────────────────────────────────────────
    model_used:     str             = Field(..., description="Groq model used, e.g. llama3-70b-8192.")
    agent_steps:    int             = Field(default=0, description="Number of agent tool calls made.")

    # ── Per-signal sub-reports ────────────────────────────────────────────────
    founder:        FounderSignal       = Field(default_factory=FounderSignal)
    funding:        FundingSignal       = Field(default_factory=FundingSignal)
    financials:     FinancialsSignal    = Field(default_factory=FinancialsSignal)
    market:         MarketSignal        = Field(default_factory=MarketSignal)
    macro:          MacroSignal         = Field(default_factory=MacroSignal)
    news:           NewsSignal          = Field(default_factory=NewsSignal)

    # ── Scores + final verdict ────────────────────────────────────────────────
    scores:             SignalScores    = Field(default_factory=SignalScores)
    recommendation:     Recommendation

    # ── Raw agent output (debug) ──────────────────────────────────────────────
    raw_agent_output:   Optional[str]  = Field(
        default=None,
        description="Raw LLM output before JSON extraction (debug use only).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "company_name": "Stripe",
                "url": "https://stripe.com",
                "sector": "fintech",
                "research_id": "a1b2c3d4-0000-0000-0000-111122223333",
                "generated_at": "2025-01-15T10:00:00Z",
                "model_used": "llama3-70b-8192",
                "agent_steps": 8,
                "scores": {
                    "founder": 9.2, "funding": 9.5, "financials": 8.8,
                    "market": 9.0, "macro": 6.5, "news": 7.5, "composite": 8.6,
                },
                "recommendation": {
                    "verdict": "INVEST",
                    "confidence": "HIGH",
                    "composite_score": 8.6,
                    "one_line": "Stripe shows exceptional founder pedigree, strong funding trajectory, and dominant market position.",
                    "bull_case": ["Top-tier founders", "Global payments moat", "$95B last valuation"],
                    "bear_case": ["High competition from Adyen", "Macro headwinds on fintech multiples"],
                    "watch_triggers": [],
                    "full_reasoning": "...",
                },
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# API UTILITY SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    groq_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    research_id: Optional[str] = None


class SchemaResponse(BaseModel):
    """Returns the output JSON schema for tooling / frontend consumption."""
    schema_version: str = "1.0.0"
    report_schema: Dict[str, Any]