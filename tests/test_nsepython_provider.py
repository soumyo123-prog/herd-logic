"""
NSEPythonProvider integration tests — real network calls to NSE India.

NSE is much flakier than Yahoo. We expect occasional transient failures
from anti-scraping protections; the base class's retry logic absorbs
most of them. If an individual test fails it's usually NSE being moody,
not our code.

What we verify:
  - health_check returns True
  - Each method returns the documented JSON-serializable shape
  - Cache dedupes a second call
  - invalidate() drops the cache entry
  - Friendly index names resolve correctly
  - _to_float handles Indian number formatting ("25,696.05")

Run with:
    make test-nsepython
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data.cache import Cache
from agents.data.nsepython_provider import (
    INDEX_NAME_MAP,
    NSEPythonProvider,
    _resolve_index,
    _to_float,
)


# ── Pure helpers (offline, always run) ──────────────────────────────────

def test_index_name_resolution():
    print("=" * 60)
    print("TEST 1: Friendly index names resolve to NSE format")
    print("=" * 60)

    cases = {
        "nifty": "NIFTY 50",
        "NIFTY": "NIFTY 50",
        "nifty50": "NIFTY 50",
        "banknifty": "NIFTY BANK",
        "finnifty": "NIFTY FIN SERVICE",
        "midcap": "NIFTY MIDCAP 100",
        "NIFTY MIDCAP 150": "NIFTY MIDCAP 150",  # pass-through for qualified names
    }
    for key, expected in cases.items():
        got = _resolve_index(key)
        assert got == expected, f"{key!r} → {got!r}, expected {expected!r}"
        print(f"  {key!r:<22} → {got!r}")

    try:
        _resolve_index("nonsense")
    except ValueError as e:
        print(f"  Unknown key correctly rejected: {e}")
    else:
        raise AssertionError("Expected ValueError for unknown key")
    print("PASSED\n")


def test_to_float_parsing():
    print("=" * 60)
    print("TEST 2: _to_float handles Indian number formatting")
    print("=" * 60)

    cases = [
        ("25,696.05", 25696.05),
        ("0.07", 0.07),
        ("1,23,456.78", 123456.78),  # Indian lakhs grouping
        (42, 42.0),
        (3.14, 3.14),
        (None, None),
        ("", None),
        ("not-a-number", None),
    ]
    for raw, expected in cases:
        got = _to_float(raw)
        assert got == expected, f"_to_float({raw!r}) → {got!r}, expected {expected!r}"
        print(f"  {raw!r:<20} → {got!r}")
    print("PASSED\n")


# ── Network-dependent tests ─────────────────────────────────────────────

def test_health_check():
    print("=" * 60)
    print("TEST 3: health_check() pings NSE marketStatus")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t3"))
    assert provider.health_check() is True
    print("  PASSED\n")


def test_market_status():
    print("=" * 60)
    print("TEST 4: get_market_status returns normalized dict")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t4"))
    status = provider.get_market_status()

    assert "markets" in status, "Expected 'markets' key"
    assert "Capital Market" in status["markets"], (
        f"Expected Capital Market entry, got {list(status['markets'].keys())}"
    )
    cap = status["markets"]["Capital Market"]
    assert cap.get("status") in ("Open", "Closed", "Close")
    print(f"  Capital Market: {cap['status']}")
    print(f"  Nifty close: {status.get('nifty_close')}")
    print("PASSED\n")


def test_fii_dii_flow():
    print("=" * 60)
    print("TEST 5: get_fii_dii returns FII and DII records")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t5"))
    flow = provider.get_fii_dii()

    assert isinstance(flow, list) and flow, "Expected non-empty list"
    categories = {row.get("category") for row in flow}
    print(f"  Categories present: {categories}")
    for row in flow:
        print(f"  {row}")
    # NSE publishes both DII and FII/FPI each trading day
    assert any("DII" in c for c in categories if c), "Expected DII row"
    assert any("FII" in c for c in categories if c), "Expected FII row"
    print("PASSED\n")


def test_top_gainers_and_losers():
    print("=" * 60)
    print("TEST 6: top gainers and losers (broad market)")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t6"))
    gainers = provider.get_top_gainers()
    losers = provider.get_top_losers()

    assert isinstance(gainers, list) and len(gainers) > 0
    assert isinstance(losers, list) and len(losers) > 0

    g = gainers[0]
    assert "symbol" in g, f"Expected 'symbol' key in gainer row, got {list(g.keys())}"
    print(f"  Top gainer: {g.get('symbol')} ({g.get('perChange')}%)")
    print(f"  Top loser:  {losers[0].get('symbol')} ({losers[0].get('perChange')}%)")

    # Keys should be trimmed to the allow-list, no bloat
    allowed = {
        "symbol", "ltp", "open_price", "high_price", "low_price",
        "prev_price", "net_price", "perChange", "trade_quantity", "turnover",
    }
    extra_keys = set(g.keys()) - allowed
    assert not extra_keys, f"Unexpected keys leaked through: {extra_keys}"
    print("PASSED\n")


def test_bulk_and_block_deals():
    print("=" * 60)
    print("TEST 7: bulk and block deals")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t7"))
    bulk = provider.get_bulk_deals()
    block = provider.get_block_deals()

    # On some days the block deals list is legitimately empty. Bulk deals
    # almost always have something. Both should be lists regardless.
    assert isinstance(bulk, list)
    assert isinstance(block, list)
    print(f"  Bulk deals:  {len(bulk)} rows")
    print(f"  Block deals: {len(block)} rows")
    if bulk:
        print(f"  First bulk deal: {bulk[0]}")
    print("PASSED\n")


def test_option_chain():
    print("=" * 60)
    print("TEST 8: option chain for NIFTY")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t8"))
    oc = provider.get_option_chain("NIFTY")

    # Shape must be stable even when market is closed (empty rows OK).
    for key in ("symbol", "underlying_value", "expiries", "rows"):
        assert key in oc, f"Missing key {key!r}"
    assert oc["symbol"] == "NIFTY"
    print(f"  Underlying: {oc['underlying_value']}")
    print(f"  Expiries available: {len(oc['expiries'])}")
    print(f"  Strike rows: {len(oc['rows'])}")
    if oc["rows"]:
        print(f"  Sample row: {oc['rows'][0]}")
    else:
        print("  (market closed — rows empty, expected)")
    print("PASSED\n")


def test_index_quote():
    print("=" * 60)
    print("TEST 9: index quote numeric parsing")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t9"))
    nifty = provider.get_index_quote("nifty")

    for key in ("last", "open", "high", "low", "previous_close",
                "percent_change", "year_high", "year_low"):
        assert key in nifty, f"Missing {key!r}"
        # All should be float or None, never string with commas
        v = nifty[key]
        assert v is None or isinstance(v, float), (
            f"{key!r} = {v!r} (type {type(v).__name__}) — expected float or None"
        )
    print(f"  Nifty last: {nifty['last']}, "
          f"year range: {nifty['year_low']} – {nifty['year_high']}")
    print("PASSED\n")


def test_cache_hit_is_fast():
    print("=" * 60)
    print("TEST 10: Second call hits cache (fast)")
    print("=" * 60)

    provider = NSEPythonProvider(cache=Cache(namespace="nse-t10"))

    t0 = time.time()
    first = provider.get_market_status()
    cold_ms = (time.time() - t0) * 1000

    t1 = time.time()
    second = provider.get_market_status()
    warm_ms = (time.time() - t1) * 1000

    assert first == second
    print(f"  Cold call: {cold_ms:.1f} ms")
    print(f"  Warm call: {warm_ms:.1f} ms")
    assert warm_ms < max(50, cold_ms / 5), "Cache may not be active"
    print("PASSED\n")


def test_invalidate():
    print("=" * 60)
    print("TEST 11: invalidate() drops cached entry")
    print("=" * 60)

    cache = Cache(namespace="nse-t11")
    provider = NSEPythonProvider(cache=cache)

    provider.get_market_status()
    key = provider._cache_key("get_market_status")
    assert cache.get(key) is not None, "Expected cache entry after first call"

    provider.invalidate("get_market_status")
    assert cache.get(key) is None, "Entry should be gone after invalidate"

    print("  Cache entry correctly removed after invalidate()")
    print("PASSED\n")


# ── Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NSEPythonProvider integration tests — hitting NSE India")
    print(f"Known index keys: {sorted(INDEX_NAME_MAP.keys())}\n")

    # Offline tests first — fastest feedback, and they validate the pure
    # helpers regardless of NSE availability.
    test_index_name_resolution()
    test_to_float_parsing()

    # Network tests — each hits NSE.
    test_health_check()
    test_market_status()
    test_fii_dii_flow()
    test_top_gainers_and_losers()
    test_bulk_and_block_deals()
    test_option_chain()
    test_index_quote()
    test_cache_hit_is_fast()
    test_invalidate()

    print("=" * 60)
    print("ALL TESTS PASSED — NSEPythonProvider ready for agents to use.")
    print("=" * 60)
