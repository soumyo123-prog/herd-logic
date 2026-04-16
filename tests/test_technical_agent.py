"""
End-to-end TechnicalAgent test — real yfinance + real OpenRouter LLM.

We assert *invariants* on the output, not exact values. LLM responses
aren't deterministic, but the deterministic post-processor ensures
every numeric level is reproducible given the same LLM proposals.

Requires OPENROUTER_API_KEY in env or macOS Keychain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data.cache import Cache
from agents.data.yfinance_provider import YFinanceProvider
from agents.llm_client import LLMClient
from agents.technical.agent import TechnicalAgent
from agents.technical.levels import RR_TARGET_1
from agents.technical.universe import DEFAULT_UNIVERSE


def test_technical_agent_full_run():
    print("TEST: TechnicalAgent.run() on 15-ticker universe")

    llm = LLMClient()
    yf = YFinanceProvider(cache=Cache(namespace="tech-e2e"))
    agent = TechnicalAgent(llm_client=llm, yf_provider=yf)

    report = agent.run()

    # Shape assertions
    assert report["market_trend"] in ("uptrend", "downtrend", "sideways")
    assert isinstance(report["setups"], list)
    assert "breadth" in report
    print(f"  market_trend = {report['market_trend']}")
    print(f"  breadth      = {report['breadth']}")
    print(f"  setups       = {len(report['setups'])}")

    for i, setup in enumerate(report["setups"], start=1):
        # Invariant 1: risk-reward ratio is approximately RR_TARGET_1.
        rr = setup["risk_reward_ratio"]
        assert abs(rr - RR_TARGET_1) < 0.1, f"setup {i}: R:R = {rr}"

        # Invariant 2: confidence in [0, 100]
        assert 0 <= setup["confidence"] <= 100

        # Invariant 3: ticker is from the input universe
        assert setup["ticker"] in DEFAULT_UNIVERSE, (
            f"setup {i}: unexpected ticker {setup['ticker']}"
        )

        # Invariant 4: direction / target ordering
        if setup["direction"] == "long":
            assert setup["target_1"] > setup["entry_zone_high"], (
                f"setup {i}: long T1 {setup['target_1']} below entry"
            )
            assert setup["stop_loss"] < setup["entry_zone_low"]
            assert setup["target_2"] > setup["target_1"]
        else:
            assert setup["target_1"] < setup["entry_zone_low"]
            assert setup["stop_loss"] > setup["entry_zone_high"]
            assert setup["target_2"] < setup["target_1"]

        print(
            f"  #{i} {setup['ticker']} {setup['direction']} "
            f"entry={setup['entry_zone_low']}-{setup['entry_zone_high']} "
            f"SL={setup['stop_loss']} T1={setup['target_1']} T2={setup['target_2']} "
            f"conf={setup['confidence']}"
        )

    # Invariant 5: setups sorted by confidence descending
    confs = [s["confidence"] for s in report["setups"]]
    assert confs == sorted(confs, reverse=True), f"not sorted: {confs}"

    print("  PASSED")


if __name__ == "__main__":
    test_technical_agent_full_run()
    print("\nEnd-to-end Technical Agent test passed.")
