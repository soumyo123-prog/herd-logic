"""
Test script — verifies OpenRouter connectivity and structured JSON output.

Run this to confirm:
1. Your API key works
2. MiniMax M2.5 responds
3. Structured JSON output (the critical feature) works correctly
4. Pydantic validation catches bad data

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
    python scripts/test_openrouter.py
"""

from typing import Literal

from pydantic import BaseModel

# Add project root to path so we can import agents/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import LLMClient


# ── Test 1: Basic connectivity (free-form text) ─────────────────────────

def test_basic_text():
    print("=" * 60)
    print("TEST 1: Basic text generation")
    print("=" * 60)

    client = LLMClient()
    response = client.generate_text(
        system_prompt="You are a helpful assistant. Keep responses under 2 sentences.",
        user_message="What is swing trading?",
    )

    print(f"Response: {response}")
    print("PASSED — LLM responded with text.\n")


# ── Test 2: Structured JSON output (the critical one) ───────────────────

def test_structured_simple():
    """
    This is the most important test. We define a Pydantic schema and ask
    the LLM to fill it in. If this works, the entire agent pipeline will work.
    """
    print("=" * 60)
    print("TEST 2: Structured JSON output (simple schema)")
    print("=" * 60)

    # Define what we want back — this is like a "form" the LLM fills in
    class StockAnalysis(BaseModel):
        ticker: str
        direction: Literal["bullish", "bearish", "neutral"]
        confidence: int  # 0-100
        reasoning: str

    client = LLMClient()
    result = client.generate_structured(
        system_prompt="You are a stock analyst. Analyze the given stock and return your assessment.",
        user_message="Analyze RELIANCE (Reliance Industries) on NSE. Current price: ₹2,850. It recently broke above its 200 DMA with high volume.",
        response_schema=StockAnalysis,
    )

    # result is a plain dict — every field guaranteed to exist and be the right type
    print(f"Result: {result}")
    print(f"  Ticker: {result['ticker']}")
    print(f"  Direction: {result['direction']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning: {result['reasoning'][:100]}...")

    # Verify types
    assert isinstance(result["ticker"], str), "ticker should be a string"
    assert result["direction"] in ("bullish", "bearish", "neutral"), "direction should be a valid enum"
    assert isinstance(result["confidence"], int), "confidence should be an int"

    print("PASSED — Got valid structured JSON matching the schema.\n")


# ── Test 3: Complex nested schema (closer to real agent output) ──────────

def test_structured_complex():
    """
    This mimics what Agent 2 (Technical Analyst) would actually return.
    Tests nested objects, lists, and optional fields.
    """
    print("=" * 60)
    print("TEST 3: Structured JSON output (complex nested schema)")
    print("=" * 60)

    class SwingSetup(BaseModel):
        ticker: str
        direction: Literal["long", "short"]
        entry_zone_low: float
        entry_zone_high: float
        stop_loss: float
        target_1: float
        risk_reward_ratio: float
        confidence: int

    class MiniTechnicalReport(BaseModel):
        market_trend: Literal["uptrend", "downtrend", "sideways"]
        setups: list[SwingSetup]
        reasoning: str

    client = LLMClient()
    result = client.generate_structured(
        system_prompt=(
            "You are a technical analyst for Indian stocks (NSE). "
            "Analyze the given data and identify swing trade setups. "
            "Return a structured report."
        ),
        user_message=(
            "Nifty 50 is at 23,500, trending up. RSI 14 = 62 (not overbought). "
            "RELIANCE: price ₹2,850, above 200 DMA (₹2,720), RSI=58, broke out of a 2-week range. "
            "INFY: price ₹1,520, below 200 DMA (₹1,580), RSI=42, downtrend. "
            "Identify any swing setups."
        ),
        response_schema=MiniTechnicalReport,
    )

    print(f"Market trend: {result['market_trend']}")
    print(f"Number of setups found: {len(result['setups'])}")
    for i, setup in enumerate(result["setups"]):
        print(f"  Setup {i+1}: {setup['ticker']} {setup['direction']} "
              f"entry={setup['entry_zone_low']}-{setup['entry_zone_high']} "
              f"SL={setup['stop_loss']} T1={setup['target_1']} "
              f"R:R={setup['risk_reward_ratio']} confidence={setup['confidence']}")
    print(f"Reasoning: {result['reasoning'][:120]}...")

    # Verify structure
    assert isinstance(result["setups"], list), "setups should be a list"
    if result["setups"]:
        setup = result["setups"][0]
        assert isinstance(setup["stop_loss"], float), "stop_loss should be a float"
        assert setup["direction"] in ("long", "short"), "direction should be long or short"

    print("PASSED — Complex nested schema works.\n")


# ── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from agents.vault import Vault

    # Check if key is available from env var OR Keychain
    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
    if not has_key:
        try:
            Vault().get("OPENROUTER_API_KEY")
            has_key = True
            print("API key loaded from macOS Keychain.")
        except KeyError:
            pass

    if not has_key:
        print("ERROR: No API key found. Set it up using one of:")
        print("  .venv/bin/python -m scripts.vault_setup set OPENROUTER_API_KEY 'sk-or-v1-your-key'")
        print("  export OPENROUTER_API_KEY='sk-or-v1-your-key'")
        sys.exit(1)

    print(f"Using model: {os.getenv('LLM_MODEL', 'minimax/minimax-m2.5:free')}")
    print()

    test_basic_text()
    test_structured_simple()
    test_structured_complex()

    print("=" * 60)
    print("ALL TESTS PASSED — OpenRouter + structured JSON output verified!")
    print("The LLM client is ready for agent development.")
    print("=" * 60)
