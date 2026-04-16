"""
BaseDataProvider — the scaffolding every market-data source inherits from.

Concrete providers (YFinanceProvider, NSEPythonProvider, NewsDataProvider)
subclass this and implement source-specific fetch methods. They don't
re-implement caching or retry — those live here.

Why this layer?
  - Every external data source fails sometimes (rate limits, transient
    network errors, upstream 5xx). Retry logic belongs in one place, not
    sprinkled across every provider.
  - Every source benefits from caching (yfinance is slow; NSE rate-limits).
    Concrete providers just call `self._cached_call(...)` instead of
    managing cache keys themselves.
  - Uniform health_check() lets the dashboard show which sources are live.

Design pattern: Template Method — the base class defines the skeleton
(cache lookup → fetch → retry → cache write), and subclasses fill in the
`fetch` callable and the cache key.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Callable

from agents.data.cache import Cache


class CacheTTL:
    """
    Time-to-live constants, in seconds, tuned to how quickly each data
    class becomes stale in swing-trading context.

    Intraday prices move continuously, so a tight TTL. Fundamentals change
    quarterly, so a long TTL. These are tuned for the POC; adjust when
    profiling shows hot paths.
    """

    QUOTE_LIVE = 60                 # 1 min — live tick price
    OHLCV_INTRADAY = 300            # 5 min — intraday candles
    OHLCV_DAILY = 3600              # 1 hr — daily candles (refresh after close)
    INDEX_SNAPSHOT = 300            # 5 min — Nifty/Sensex snapshot
    FII_DII = 3600                  # 1 hr — FII/DII flow (released daily)
    OPTION_CHAIN = 300              # 5 min — option chain data
    FUNDAMENTALS = 86400            # 24 hr — PE, EPS, etc.
    NEWS = 600                      # 10 min — news headlines
    TICKER_UNIVERSE = 604800        # 1 week — list of NSE tickers
    SECTOR_META = 86400             # 24 hr — sector/industry mapping


class ProviderError(Exception):
    """Raised when a provider exhausts retries or hits a non-retryable error."""


class BaseDataProvider(ABC):
    """
    Abstract base for market-data providers.

    Subclasses must set `name` and implement `health_check()`. Everything
    else (caching, retry, key construction) is inherited.

    Retry strategy: exponential backoff with jitter. Doubling delay
    (1s, 2s, 4s) avoids hammering a struggling upstream; jitter (random
    0-50% added) prevents "thundering herd" when many processes retry at
    the same synchronized moment.
    """

    # Override in each subclass — used as the cache key prefix and for logs.
    name: str = "base"

    # Retryable error signatures. Subclasses can extend this tuple to
    # include source-specific exceptions (e.g. yfinance.exceptions.YFRateLimitError).
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)

    def __init__(
        self,
        cache: Cache | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self.cache = cache if cache is not None else Cache()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    # ── Public helpers for subclasses ──────────────────────────────────────

    def _cache_key(self, method: str, **params) -> str:
        """
        Build a deterministic, collision-free cache key.

        Format: `<provider>:<method>:<sorted k=v pairs>`
        Sorting ensures that get_ohlcv(ticker=A, period=6mo) and
        get_ohlcv(period=6mo, ticker=A) produce the same key.
        """
        parts = [f"{k}={params[k]}" for k in sorted(params)]
        suffix = ":".join(parts) if parts else "_"
        return f"{self.name}:{method}:{suffix}"

    def _cached_call(
        self,
        method: str,
        fetch_fn: Callable,
        ttl: int,
        force_refresh: bool = False,
        **params,
    ):
        """
        Cache-aside pattern with retry.

        Flow:
          1. Build the cache key from the method name + params.
          2. If not force_refresh, check cache → return on hit.
          3. On miss, call fetch_fn() with retry.
          4. Store the result (if non-None) and return it.

        `fetch_fn` is a zero-arg callable the subclass provides (usually a
        lambda capturing the real arguments). Keeping it zero-arg lets the
        base class call it inside the retry loop without knowing the
        signature of every concrete method.
        """
        key = self._cache_key(method, **params)

        if not force_refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        value = self._call_with_retry(fetch_fn, context=key)

        if value is not None:
            self.cache.set(key, value, ttl=ttl)
        return value

    def _call_with_retry(self, fn: Callable, context: str = ""):
        """
        Execute `fn` with retries and exponential backoff + jitter.

        Retries only on `self.retryable_exceptions`. Non-retryable errors
        (e.g., a 401 auth failure) propagate immediately — retrying won't
        help and only wastes budget.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return fn()
            except self.retryable_exceptions as e:
                last_error = e

                if attempt == self.max_retries - 1:
                    break

                # Exponential backoff: 1s, 2s, 4s, 8s... capped at max_delay.
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                # Jitter: +/- up to 50% so retries desynchronize across callers.
                delay = delay * (0.5 + random.random())

                print(
                    f"[{self.name}] {type(e).__name__} on {context or 'fetch'}, "
                    f"retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)

        raise ProviderError(
            f"[{self.name}] exhausted {self.max_retries} retries on {context or 'fetch'}: {last_error}"
        ) from last_error

    def invalidate(self, method: str, **params) -> None:
        """Drop a specific cached entry — useful when the agent detects stale data."""
        self.cache.delete(self._cache_key(method, **params))

    # ── Abstract surface ───────────────────────────────────────────────────

    @abstractmethod
    def health_check(self) -> bool:
        """
        Return True if the underlying source is reachable. Each provider
        implements a cheap probe — typically fetching a single well-known
        value. The dashboard and startup sequence use this to fail early.
        """
