"""
YFinanceProvider tests — real network calls against Yahoo Finance.

These are integration tests, not pure unit tests. They hit Yahoo's servers,
so they're subject to:
  - Network latency (first call per ticker ~1-3s)
  - Occasional rate limiting / transient 5xx (base class retry handles these)
  - Yahoo occasionally delisting or renaming symbols

We use a fresh namespaced cache per test so runs are deterministic, and
we verify cache behavior by timing the second call.

Run with:
    make test-yfinance
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data.cache import Cache
from agents.data.yfinance_provider import INDEX_MAP, YFinanceProvider


# ── Individual tests ────────────────────────────────────────────────────

def test_health_check():
    print("=" * 60)
    print("TEST 1: health_check() hits Nifty and returns True")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t1"))
    assert provider.health_check() is True
    print("  Health check PASSED — Yahoo reachable, Nifty has data\n")


def test_ticker_qualification():
    print("=" * 60)
    print("TEST 2: Ticker normalization rules")
    print("=" * 60)

    p = YFinanceProvider(cache=Cache(namespace="yf-t2"))
    cases = {
        "RELIANCE": "RELIANCE.NS",        # bare → .NS
        "reliance": "RELIANCE.NS",        # case-insensitive
        " INFY ": "INFY.NS",              # trims whitespace
        "TCS.NS": "TCS.NS",               # already suffixed → unchanged
        "HDFC.BO": "HDFC.BO",             # BSE suffix preserved
        "^NSEI": "^NSEI",                 # index → unchanged
        "CL=F": "CL=F",                   # commodity → unchanged
        "DX-Y.NYB": "DX-Y.NYB",           # DXY → unchanged
    }
    for raw, expected in cases.items():
        actual = p._qualify(raw)
        assert actual == expected, f"{raw!r} → {actual!r}, expected {expected!r}"
        print(f"  {raw!r:<15} → {actual}")
    print("PASSED\n")


def test_get_ohlcv_basic():
    print("=" * 60)
    print("TEST 3: get_ohlcv returns normalized candles for RELIANCE")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t3"))
    candles = provider.get_ohlcv("RELIANCE", period="1mo", interval="1d")

    assert isinstance(candles, list) and len(candles) > 0, "Expected non-empty list"
    first = candles[0]
    for key in ("date", "open", "high", "low", "close", "volume"):
        assert key in first, f"Missing key {key!r} in candle: {first}"
    print(f"  Got {len(candles)} candles")
    print(f"  First: {first}")
    print(f"  Last:  {candles[-1]}")
    print("PASSED\n")


def test_ohlcv_cache_hit_is_fast():
    print("=" * 60)
    print("TEST 4: Second get_ohlcv call hits cache (fast)")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t4"))

    t0 = time.time()
    first = provider.get_ohlcv("INFY", period="1mo", interval="1d")
    cold_ms = (time.time() - t0) * 1000

    t1 = time.time()
    second = provider.get_ohlcv("INFY", period="1mo", interval="1d")
    warm_ms = (time.time() - t1) * 1000

    assert first == second, "Cached payload should equal original"
    print(f"  Cold call: {cold_ms:.1f} ms ({len(first)} candles)")
    print(f"  Warm call: {warm_ms:.1f} ms")
    # Cache should be orders of magnitude faster. 50ms is a loose bound.
    assert warm_ms < max(50, cold_ms / 5), (
        f"Warm call too slow ({warm_ms:.1f} ms) — cache may not be working"
    )
    print("PASSED\n")


def test_get_quote():
    print("=" * 60)
    print("TEST 5: get_quote returns flat snapshot dict")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t5"))
    quote = provider.get_quote("RELIANCE")

    assert quote["symbol"] == "RELIANCE.NS"
    # Market may be closed, so some fields can be None; but structure is stable.
    expected_keys = {
        "symbol", "last_price", "open", "day_high", "day_low",
        "previous_close", "volume", "market_cap", "fifty_day_avg",
        "two_hundred_day_avg", "currency", "exchange",
    }
    assert expected_keys.issubset(quote.keys()), (
        f"Missing keys: {expected_keys - quote.keys()}"
    )
    print(f"  Quote: {quote}")
    print("PASSED\n")


def test_get_fundamentals():
    print("=" * 60)
    print("TEST 6: get_fundamentals returns curated PE/EPS/sector dict")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t6"))
    fundies = provider.get_fundamentals("RELIANCE")

    assert fundies["symbol"] == "RELIANCE.NS"
    # These are nearly always populated for large caps.
    for key in ("sector", "marketCap", "trailingPE"):
        assert fundies.get(key) is not None, f"Expected non-None {key}, got {fundies.get(key)}"
    print(f"  Sector: {fundies['sector']}")
    print(f"  Industry: {fundies['industry']}")
    print(f"  Market cap: {fundies['marketCap']:,}")
    print(f"  Trailing PE: {fundies['trailingPE']}")
    print(f"  52w range: {fundies['fiftyTwoWeekLow']} - {fundies['fiftyTwoWeekHigh']}")
    print("PASSED\n")


def test_get_index_by_friendly_name():
    print("=" * 60)
    print("TEST 7: get_index('nifty') resolves to ^NSEI")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t7"))
    candles = provider.get_ohlcv.__self__  # just to confirm method binding
    _ = candles

    nifty = provider.get_index("nifty", period="5d", interval="1d")
    assert isinstance(nifty, list) and nifty, "Expected Nifty candles"
    print(f"  Nifty last 5d: {len(nifty)} candles")
    print(f"  Latest close: {nifty[-1]['close']:.2f}")

    # Unknown name should raise ValueError with helpful message
    try:
        provider.get_index("nonsense")
    except ValueError as e:
        print(f"  Unknown index correctly rejected: {e}")
    else:
        raise AssertionError("Expected ValueError for unknown index")
    print("PASSED\n")


def test_get_global_snapshot():
    print("=" * 60)
    print("TEST 8: get_global_snapshot returns all key global markets")
    print("=" * 60)

    provider = YFinanceProvider(cache=Cache(namespace="yf-t8"))
    snap = provider.get_global_snapshot()

    for friendly in ("sp500", "dow", "nikkei", "crude", "gold", "dxy", "usd_inr"):
        assert friendly in snap, f"Missing {friendly} in snapshot"
        entry = snap[friendly]
        if "error" in entry:
            print(f"  {friendly:10}: ERROR ({entry['error']})")
        else:
            print(f"  {friendly:10}: {entry['last_close']:>12.2f}  "
                  f"({entry['pct_change']:+.2f}%)")
    # At least the major US indices should resolve cleanly
    assert "error" not in snap["sp500"], f"S&P 500 failed: {snap['sp500']}"
    print("PASSED\n")


def test_invalidate_forces_refetch():
    print("=" * 60)
    print("TEST 9: invalidate() drops the cache entry")
    print("=" * 60)

    cache = Cache(namespace="yf-t9")
    provider = YFinanceProvider(cache=cache)

    key = provider._cache_key(
        "get_ohlcv", ticker="TCS.NS", period="1mo", interval="1d"
    )

    provider.get_ohlcv("TCS", period="1mo", interval="1d")
    assert cache.get(key) is not None, "Expected cache entry after first call"

    provider.invalidate("get_ohlcv", ticker="TCS.NS", period="1mo", interval="1d")
    assert cache.get(key) is None, "Cache entry should be gone after invalidate"

    print("  Cache entry correctly removed after invalidate()")
    print("PASSED\n")


# ── Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("YFinanceProvider integration tests — hitting Yahoo Finance\n")
    print(f"Known indices: {sorted(INDEX_MAP.keys())}\n")

    test_health_check()
    test_ticker_qualification()
    test_get_ohlcv_basic()
    test_ohlcv_cache_hit_is_fast()
    test_get_quote()
    test_get_fundamentals()
    test_get_index_by_friendly_name()
    test_get_global_snapshot()
    test_invalidate_forces_refetch()

    print("=" * 60)
    print("ALL TESTS PASSED — YFinanceProvider ready for agents to use.")
    print("=" * 60)
