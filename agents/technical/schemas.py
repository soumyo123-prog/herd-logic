"""
Pydantic schemas — the contracts between the LLM, the levels post-processor,
and downstream agents.

Two layers:
  - LLM-facing (LLMSetupProposal, LLMTechnicalReport) — what we ask the
    model to produce. Narrow. No numeric entry/stop/target — the LLM is
    not trusted to compute those correctly.
  - Agent output (SwingSetup, TechnicalReport) — what the agent emits to
    downstream consumers. All numeric fields are computed deterministically
    from the LLM proposal plus the feature snapshot.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMSetupProposal(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    confidence: int = Field(ge=0, le=100)
    key_signals: list[str]
    holding_period_days: int = Field(ge=1, le=30)


class LLMTechnicalReport(BaseModel):
    market_trend: Literal["uptrend", "downtrend", "sideways"]
    breadth_reasoning: str
    setups: list[LLMSetupProposal]
    reasoning: str


class SwingSetup(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    confidence: int = Field(ge=0, le=100)
    timeframe: Literal["swing"] = "swing"
    holding_period_days: int
    key_signals: list[str]


class TechnicalReport(BaseModel):
    as_of: str  # ISO date
    market_trend: Literal["uptrend", "downtrend", "sideways"]
    breadth: dict
    setups: list[SwingSetup]
    reasoning: str
