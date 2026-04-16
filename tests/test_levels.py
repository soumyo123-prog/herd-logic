"""
Tests for deterministic entry/SL/target computation.

No network, no LLM — we hand-craft LLMSetupProposal instances and feature
snapshots, then assert that compute_levels produces the exact numeric
levels our risk rules dictate. These are the most load-bearing numbers
in the whole swing system, so the tests are explicit to the decimal.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.technical.levels import (
    ATR_MULTIPLIER_STOP,
    ENTRY_ZONE_PCT,
    RR_TARGET_1,
    RR_TARGET_2,
    compute_levels,
)
from agents.technical.schemas import LLMSetupProposal


def _snapshot(last_close: float, atr: float) -> dict:
    return {
        "ticker": "TEST",
        "last_close": last_close,
        "daily": {"atr14": atr},
    }


def test_long_setup_levels():
    print("TEST: Long setup with last=100, atr=2 → SL=97, T1=106, T2=109")
    prop = LLMSetupProposal(
        ticker="TEST", direction="long", confidence=70,
        key_signals=["above SMA200"], holding_period_days=7,
    )
    setup = compute_levels(prop, _snapshot(100.0, 2.0))
    assert math.isclose(setup.entry_zone_low, 99.5)
    assert math.isclose(setup.entry_zone_high, 100.5)
    assert math.isclose(setup.stop_loss, 97.0)
    assert math.isclose(setup.target_1, 106.0)  # 100 + 2 × (100 - 97) = 106
    assert math.isclose(setup.target_2, 109.0)  # 100 + 3 × 3 = 109
    assert math.isclose(setup.risk_reward_ratio, RR_TARGET_1)
    print("  PASSED")


def test_short_setup_levels():
    print("TEST: Short setup with last=100, atr=2 → SL=103, T1=94, T2=91")
    prop = LLMSetupProposal(
        ticker="TEST", direction="short", confidence=60,
        key_signals=["below SMA200"], holding_period_days=7,
    )
    setup = compute_levels(prop, _snapshot(100.0, 2.0))
    assert math.isclose(setup.stop_loss, 103.0)
    assert math.isclose(setup.target_1, 94.0)  # 100 - 2 × (103 - 100) = 94
    assert math.isclose(setup.target_2, 91.0)
    print("  PASSED")


def test_levels_propagate_llm_fields():
    print("TEST: confidence, key_signals, holding_period_days copied from proposal")
    prop = LLMSetupProposal(
        ticker="TEST", direction="long", confidence=55,
        key_signals=["a", "b"], holding_period_days=10,
    )
    setup = compute_levels(prop, _snapshot(100.0, 2.0))
    assert setup.confidence == 55
    assert setup.key_signals == ["a", "b"]
    assert setup.holding_period_days == 10
    assert setup.timeframe == "swing"
    print("  PASSED")


def test_constants_are_exposed_for_tuning():
    print("TEST: tunable constants are exported")
    assert ATR_MULTIPLIER_STOP == 1.5
    assert RR_TARGET_1 == 2.0
    assert RR_TARGET_2 == 3.0
    assert ENTRY_ZONE_PCT == 0.005
    print("  PASSED")


if __name__ == "__main__":
    test_long_setup_levels()
    test_short_setup_levels()
    test_levels_propagate_llm_fields()
    test_constants_are_exposed_for_tuning()
    print("\nAll level-computation tests passed.")
