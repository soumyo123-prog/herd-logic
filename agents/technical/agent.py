"""
TechnicalAgent — the first concrete BaseAgent subclass.

Pipeline:
  1. gather_features pulls daily + weekly OHLCV for every ticker via
     YFinanceProvider, drops tickers with insufficient history, builds
     per-ticker snapshots, and computes a market breadth block.
  2. build_system_prompt returns the constant prompt from prompts.py.
  3. The LLM emits an LLMTechnicalReport (qualitative proposals only).
  4. post_process calls levels.compute_levels for each proposal — the
     only place concrete entry/stop/target numbers are produced — and
     wraps everything into a final TechnicalReport dict sorted by
     confidence descending.
"""

from __future__ import annotations

from datetime import date

from agents.base_agent import BaseAgent
from agents.data.yfinance_provider import YFinanceProvider
from agents.technical.features import build_snapshot, insufficient_history
from agents.technical.levels import compute_levels
from agents.technical.prompts import SYSTEM_PROMPT
from agents.technical.schemas import (
    LLMSetupProposal,
    LLMTechnicalReport,
    TechnicalReport,
)
from agents.technical.universe import DEFAULT_UNIVERSE


class TechnicalAgent(BaseAgent):
    name = "technical"
    response_schema = LLMTechnicalReport

    def __init__(self, llm_client, yf_provider: YFinanceProvider | None = None):
        super().__init__(llm_client=llm_client)
        self.yf = yf_provider or YFinanceProvider()

    # ── Feature gathering ────────────────────────────────────────────

    def gather_features(
        self,
        tickers: list[str] | None = None,
        **_: object,
    ) -> dict:
        tickers = tickers or DEFAULT_UNIVERSE
        as_of = date.today().isoformat()

        snapshots: list[dict] = []
        skipped: list[str] = []

        for ticker in tickers:
            try:
                daily = self.yf.get_ohlcv(ticker, period="1y", interval="1d")
                weekly = self.yf.get_ohlcv(ticker, period="2y", interval="1wk")
            except Exception as e:
                print(f"[technical] skip {ticker}: fetch error {e}")
                skipped.append(ticker)
                continue

            if not daily or not weekly or insufficient_history(daily):
                print(f"[technical] skip {ticker}: insufficient history")
                skipped.append(ticker)
                continue

            snapshots.append(build_snapshot(ticker, daily, weekly, as_of))

        market = _market_block(snapshots)
        return {"as_of": as_of, "market": market, "tickers": snapshots, "skipped": skipped}

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ── Post-processing: LLM proposals → deterministic levels ────────

    def post_process(self, raw: dict, features: dict) -> dict:
        llm_report = LLMTechnicalReport.model_validate(raw)
        snapshot_by_ticker = {s["ticker"]: s for s in features["tickers"]}

        setups = []
        for prop in llm_report.setups:
            snap = snapshot_by_ticker.get(prop.ticker)
            if snap is None:
                # LLM hallucinated a ticker not in our input — drop it.
                continue
            if not _passes_rules(prop, snap):
                continue
            setups.append(compute_levels(prop, snap))

        setups.sort(key=lambda s: s.confidence, reverse=True)

        report = TechnicalReport(
            as_of=features["as_of"],
            market_trend=llm_report.market_trend,
            breadth=features["market"],
            setups=setups,
            reasoning=llm_report.reasoning,
        )
        return report.model_dump()


# ── Helpers ──────────────────────────────────────────────────────────

def _market_block(snapshots: list[dict]) -> dict:
    if not snapshots:
        return {
            "universe_size": 0, "advancing": 0, "declining": 0,
            "avg_rsi": None, "tickers_above_sma200": 0,
        }
    advancing = 0
    declining = 0
    above_200 = 0
    rsis: list[float] = []
    for s in snapshots:
        recent = s["recent_candles_5d"]
        if len(recent) >= 2 and recent[-1]["close"] > recent[-2]["close"]:
            advancing += 1
        else:
            declining += 1
        if s["daily"]["trend_label"] == "above_sma200":
            above_200 += 1
        rsi_v = s["daily"]["rsi14"]
        if rsi_v is not None:
            rsis.append(rsi_v)
    return {
        "universe_size": len(snapshots),
        "advancing": advancing,
        "declining": declining,
        "avg_rsi": round(sum(rsis) / len(rsis), 2) if rsis else None,
        "tickers_above_sma200": above_200,
    }


def _passes_rules(proposal: LLMSetupProposal, snapshot: dict) -> bool:
    """Enforce the hard rules from the system prompt as a belt-and-suspenders check."""
    daily = snapshot["daily"]
    trend = daily.get("trend_label")
    rsi_v = daily.get("rsi14")
    if rsi_v is None or trend is None:
        return False
    if proposal.direction == "long":
        return trend == "above_sma200" and rsi_v >= 40
    return trend == "below_sma200" and rsi_v <= 60
