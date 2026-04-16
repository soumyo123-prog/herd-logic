"""
Tests for the feature-snapshot builder.

No network. We synthesize ~250 daily candles and ~60 weekly candles,
feed them through build_snapshot(), and verify shape + derived labels.
The LLM never sees raw OHLCV — only the snapshot — so its structure
is part of the agent's public contract and worth explicit tests.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.technical.features import (
    MIN_DAILY_CANDLES,
    build_snapshot,
    insufficient_history,
)


def _synth_candles(n: int, start_price: float = 100.0, step: float = 0.5):
    """Generate `n` synthetic daily candles with monotonically rising closes."""
    return [
        {
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "open": start_price + i * step - 0.2,
            "high": start_price + i * step + 0.3,
            "low": start_price + i * step - 0.4,
            "close": start_price + i * step,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


def test_snapshot_has_required_daily_keys():
    print("TEST: snapshot daily block has every required key")
    daily = _synth_candles(250)
    weekly = _synth_candles(60, start_price=100, step=2.5)
    snap = build_snapshot("TEST", daily, weekly, as_of="2026-04-16")

    required = {
        "sma20", "sma50", "sma200",
        "ema9", "ema21",
        "rsi14", "rsi_state",
        "macd", "macd_signal", "macd_hist", "macd_state",
        "bb_upper", "bb_lower", "bb_position",
        "atr14", "atr_pct",
        "volume_ratio_20d",
        "trend_label", "pct_from_sma200",
    }
    missing = required - set(snap["daily"].keys())
    assert not missing, f"missing daily keys: {missing}"
    print("  PASSED")


def test_snapshot_trend_label_on_rising_series():
    print("TEST: rising series → trend_label 'above_sma200'")
    daily = _synth_candles(250)
    weekly = _synth_candles(60, step=2.5)
    snap = build_snapshot("TEST", daily, weekly, as_of="2026-04-16")
    assert snap["daily"]["trend_label"] == "above_sma200", (
        f"got {snap['daily']['trend_label']}"
    )
    assert snap["daily"]["rsi_state"] in ("oversold", "neutral", "overbought")
    print("  PASSED")


def test_recent_candles_5d_included():
    print("TEST: snapshot includes last 5 raw candles")
    daily = _synth_candles(250)
    weekly = _synth_candles(60, step=2.5)
    snap = build_snapshot("TEST", daily, weekly, as_of="2026-04-16")
    assert len(snap["recent_candles_5d"]) == 5
    for row in snap["recent_candles_5d"]:
        assert set(row.keys()) >= {"date", "open", "high", "low", "close", "volume"}
    print("  PASSED")


def test_insufficient_history_detection():
    print("TEST: insufficient_history returns True for < MIN_DAILY_CANDLES")
    daily = _synth_candles(MIN_DAILY_CANDLES - 1)
    assert insufficient_history(daily) is True
    daily = _synth_candles(MIN_DAILY_CANDLES)
    assert insufficient_history(daily) is False
    print("  PASSED")


def test_trailing_nan_candle_is_dropped():
    print("TEST: trailing NaN candle (partial live session) is filtered")
    daily = _synth_candles(250)
    # Append a partial-session candle the way yfinance emits it mid-day.
    daily.append({
        "date": "2026-04-17",
        "open": float("nan"), "high": float("nan"),
        "low": float("nan"), "close": float("nan"),
        "volume": 123456,
    })
    weekly = _synth_candles(60, step=2.5)
    snap = build_snapshot("TEST", daily, weekly, as_of="2026-04-17")
    # last_close must come from the last VALID candle, not the NaN one.
    expected_last = 100.0 + 249 * 0.5  # 250th synthetic candle's close
    assert math.isclose(snap["last_close"], expected_last), (
        f"got {snap['last_close']}, expected {expected_last}"
    )
    # sma200 must be a real number, not NaN.
    sma200 = snap["daily"]["sma200"]
    assert sma200 is not None and not math.isnan(sma200), (
        f"sma200 corrupted by NaN: {sma200}"
    )
    print("  PASSED")


if __name__ == "__main__":
    test_snapshot_has_required_daily_keys()
    test_snapshot_trend_label_on_rising_series()
    test_recent_candles_5d_included()
    test_insufficient_history_detection()
    test_trailing_nan_candle_is_dropped()
    print("\nAll feature-snapshot tests passed.")
