"""
Pipeline test — hits real Yahoo Finance, skips the LLM.

Verifies that for a handful of tickers we can:
  1. Fetch OHLCV from YFinanceProvider.
  2. Build snapshots for those with enough history.
  3. Produce a well-formed market breadth block.

Flakiness: Yahoo is usually available; occasional first-run slowness is
normal. If this fails, verify network + rerun before suspecting code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data.cache import Cache
from agents.data.yfinance_provider import YFinanceProvider
from agents.technical.features import build_snapshot, insufficient_history


def test_pipeline_snapshot_for_five_tickers():
    print("TEST: build snapshots for 5 real tickers via YFinanceProvider")
    cache = Cache(namespace="tech-pipe-1")
    yf = YFinanceProvider(cache=cache)

    tickers = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "SBIN"]
    built = 0
    for t in tickers:
        daily = yf.get_ohlcv(t, period="1y", interval="1d")
        weekly = yf.get_ohlcv(t, period="2y", interval="1wk")
        assert isinstance(daily, list) and isinstance(weekly, list), (
            f"{t}: expected lists, got {type(daily)} / {type(weekly)}"
        )
        if insufficient_history(daily):
            print(f"  skip {t} (insufficient history)")
            continue
        snap = build_snapshot(t, daily, weekly, as_of="2026-04-16")
        assert snap["ticker"] == t
        for key in ("sma20", "sma200", "rsi14", "atr14", "trend_label"):
            assert key in snap["daily"], f"{t} missing {key}"
        built += 1
        print(f"  {t}: trend={snap['daily']['trend_label']}, "
              f"rsi={snap['daily']['rsi14']:.1f}, "
              f"atr={snap['daily']['atr14']:.2f}")
    assert built >= 3, f"expected >=3 snapshots built, got {built}"
    print("  PASSED")


if __name__ == "__main__":
    test_pipeline_snapshot_for_five_tickers()
    print("\nPipeline test passed.")
