"""
Tests for BaseDataProvider and Cache.

We use a `FakeProvider` that simulates an external source — we can control
exactly when it succeeds, fails, or hangs. This exercises caching, retry,
and graceful degradation without hitting a real network.

Run with:
    make test-data
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data.base_provider import BaseDataProvider, ProviderError
from agents.data.cache import Cache


# ── Fake provider used throughout the tests ─────────────────────────────

class FakeProvider(BaseDataProvider):
    """
    A provider that wraps a controllable `fetch` function. Tests inject
    the function to simulate success, flakiness, or permanent failure.
    """

    name = "fake"

    def __init__(self, fetch_fn, **kwargs):
        super().__init__(**kwargs)
        self._fetch_fn = fetch_fn
        self.call_count = 0

    def get_thing(self, ticker: str, force_refresh: bool = False) -> dict:
        def _fetch():
            self.call_count += 1
            return self._fetch_fn(ticker)

        return self._cached_call(
            method="get_thing",
            fetch_fn=_fetch,
            ttl=2,  # short TTL so we can test expiry without sleeping long
            force_refresh=force_refresh,
            ticker=ticker,
        )

    def health_check(self) -> bool:
        return True


# ── Tests ───────────────────────────────────────────────────────────────

def test_cache_hit_avoids_second_fetch():
    print("=" * 60)
    print("TEST 1: Cache hit avoids a second upstream fetch")
    print("=" * 60)

    cache = Cache(namespace="test1")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=lambda t: {"ticker": t, "price": 100.0},
        cache=cache,
    )

    first = provider.get_thing("RELIANCE")
    second = provider.get_thing("RELIANCE")

    assert first == second, "Cached value should equal original"
    assert provider.call_count == 1, (
        f"Upstream should be called once, not {provider.call_count}"
    )
    print(f"  Result: {first}")
    print(f"  Upstream calls: {provider.call_count} (expected 1)")
    print("PASSED\n")


def test_cache_key_includes_params():
    print("=" * 60)
    print("TEST 2: Different params → different cache entries")
    print("=" * 60)

    cache = Cache(namespace="test2")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=lambda t: {"ticker": t, "price": hash(t) % 1000},
        cache=cache,
    )

    a = provider.get_thing("RELIANCE")
    b = provider.get_thing("INFY")
    a_again = provider.get_thing("RELIANCE")

    assert a["ticker"] == "RELIANCE" and b["ticker"] == "INFY"
    assert a == a_again, "Same ticker should hit cache"
    assert provider.call_count == 2, (
        f"Should have fetched exactly twice (once per ticker), got {provider.call_count}"
    )
    print(f"  RELIANCE: {a}")
    print(f"  INFY: {b}")
    print(f"  Upstream calls: {provider.call_count} (expected 2)")
    print("PASSED\n")


def test_ttl_expiry():
    print("=" * 60)
    print("TEST 3: TTL expiry forces a re-fetch")
    print("=" * 60)

    cache = Cache(namespace="test3")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=lambda t: {"ticker": t, "ts": time.time()},
        cache=cache,
    )

    first = provider.get_thing("TCS")
    print(f"  First fetch at ts={first['ts']:.2f}")

    # TTL in FakeProvider.get_thing is 2s. Wait it out.
    time.sleep(2.2)

    second = provider.get_thing("TCS")
    print(f"  Second fetch at ts={second['ts']:.2f}")

    assert second["ts"] > first["ts"], "Should have re-fetched after TTL"
    assert provider.call_count == 2, (
        f"Expected 2 upstream calls after TTL, got {provider.call_count}"
    )
    print("PASSED\n")


def test_force_refresh():
    print("=" * 60)
    print("TEST 4: force_refresh=True bypasses cache")
    print("=" * 60)

    cache = Cache(namespace="test4")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=lambda t: {"ticker": t, "ts": time.time()},
        cache=cache,
    )

    provider.get_thing("HDFC")
    provider.get_thing("HDFC", force_refresh=True)

    assert provider.call_count == 2, (
        f"force_refresh should skip cache, expected 2 calls, got {provider.call_count}"
    )
    print(f"  Upstream calls: {provider.call_count} (expected 2)")
    print("PASSED\n")


def test_retry_recovers_from_transient_error():
    print("=" * 60)
    print("TEST 5: Retry logic recovers from transient errors")
    print("=" * 60)

    attempts = {"n": 0}

    def flaky(ticker):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError(f"simulated transient failure #{attempts['n']}")
        return {"ticker": ticker, "price": 42}

    cache = Cache(namespace="test5")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=flaky,
        cache=cache,
        max_retries=5,
        base_delay=0.05,  # keep the test fast
    )

    result = provider.get_thing("BHEL")
    print(f"  Result after retries: {result}")
    print(f"  Total upstream attempts: {attempts['n']}")
    assert result == {"ticker": "BHEL", "price": 42}
    assert attempts["n"] == 3
    print("PASSED\n")


def test_retry_exhaustion_raises():
    print("=" * 60)
    print("TEST 6: Persistent failures raise ProviderError")
    print("=" * 60)

    def always_fails(_ticker):
        raise ConnectionError("upstream permanently down")

    cache = Cache(namespace="test6")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=always_fails,
        cache=cache,
        max_retries=2,
        base_delay=0.05,
    )

    try:
        provider.get_thing("DEAD")
        raise AssertionError("Should have raised ProviderError")
    except ProviderError as e:
        print(f"  Correctly raised ProviderError: {e}")
    print("PASSED\n")


def test_non_retryable_error_propagates_immediately():
    print("=" * 60)
    print("TEST 7: Non-retryable errors bypass the retry loop")
    print("=" * 60)

    attempts = {"n": 0}

    def auth_error(_ticker):
        attempts["n"] += 1
        raise ValueError("401 Unauthorized")  # not in retryable_exceptions

    cache = Cache(namespace="test7")
    cache.clear_namespace()

    provider = FakeProvider(
        fetch_fn=auth_error,
        cache=cache,
        max_retries=5,
        base_delay=0.05,
    )

    try:
        provider.get_thing("X")
    except ValueError:
        pass
    print(f"  Upstream attempts: {attempts['n']} (expected 1 — no retries)")
    assert attempts["n"] == 1, (
        f"Non-retryable errors should not retry, got {attempts['n']} attempts"
    )
    print("PASSED\n")


def test_cache_survives_when_redis_down():
    print("=" * 60)
    print("TEST 8: Cache falls back to in-memory when Redis unreachable")
    print("=" * 60)

    # Point at a definitely-dead port so Redis connection fails
    cache = Cache(host="127.0.0.1", port=1, namespace="test8")
    assert cache.backend == "memory", "Should have fallen back to memory"

    provider = FakeProvider(
        fetch_fn=lambda t: {"ticker": t, "ok": True},
        cache=cache,
    )
    first = provider.get_thing("CACHE_DOWN")
    second = provider.get_thing("CACHE_DOWN")

    assert first == second
    assert provider.call_count == 1, "In-memory cache should still dedupe fetches"
    print(f"  Backend: {cache.health()['backend']}")
    print(f"  Calls: {provider.call_count} (expected 1)")
    print("PASSED\n")


def test_cache_key_order_independence():
    print("=" * 60)
    print("TEST 9: Cache key is stable regardless of param order")
    print("=" * 60)

    provider = FakeProvider(fetch_fn=lambda _t: {}, cache=Cache(namespace="test9"))

    k1 = provider._cache_key("m", ticker="A", period="1y")
    k2 = provider._cache_key("m", period="1y", ticker="A")

    assert k1 == k2, f"Keys differ: {k1} vs {k2}"
    print(f"  Both keys: {k1}")
    print("PASSED\n")


# ── Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cache_hit_avoids_second_fetch()
    test_cache_key_includes_params()
    test_ttl_expiry()
    test_force_refresh()
    test_retry_recovers_from_transient_error()
    test_retry_exhaustion_raises()
    test_non_retryable_error_propagates_immediately()
    test_cache_survives_when_redis_down()
    test_cache_key_order_independence()

    print("=" * 60)
    print("ALL TESTS PASSED — BaseDataProvider is ready for concrete providers.")
    print("=" * 60)
