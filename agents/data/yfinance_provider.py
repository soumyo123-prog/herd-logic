"""
YFinanceProvider — Tier 1 data source for OHLCV, fundamentals, and global markets.

Why yfinance?
  - Free, no API key, no rate-limit enforcement (gentle self-rate-limiting).
  - Covers Indian stocks (NSE: `.NS` suffix, BSE: `.BO`), indices, global
    equities (S&P 500, Nikkei), commodities (crude, gold), and FX.
  - Well-maintained Python wrapper around Yahoo Finance's public endpoints.

Caveats:
  - Yahoo scrapes data from exchanges and can lag real-time by ~15 minutes.
    For swing trading (multi-day holds) that's fine; for intraday it isn't.
  - Data quality is "mostly good" — occasional missing candles, weekend gaps,
    and the "lastPrice" field can be stale when markets are closed.

Ticker conventions used throughout this module:
  - NSE equities: "RELIANCE.NS", "INFY.NS"
  - BSE equities: "RELIANCE.BO"
  - Nifty 50: "^NSEI"
  - Sensex: "^BSESN"
  - India VIX: "^INDIAVIX"
  - S&P 500: "^GSPC", Dow: "^DJI", Nasdaq: "^IXIC", Nikkei: "^N225"
  - Crude: "CL=F", Gold: "GC=F", DXY (dollar index): "DX-Y.NYB"

All methods return JSON-serializable structures (dicts, lists of dicts) so
the cache layer can store them without custom serializers.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from agents.data.base_provider import BaseDataProvider, CacheTTL
from agents.data.cache import Cache


# Friendly index names → yfinance symbols. Keeps callers from having to
# remember Yahoo's quirky `^NSEI` / `^BSESN` / etc.
INDEX_MAP = {
    "nifty": "^NSEI",
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "banknifty": "^NSEBANK",
    "vix": "^INDIAVIX",
    "india_vix": "^INDIAVIX",
}

# Tickers for the macro agent's "global pulse" snapshot.
# Cues that move Indian markets overnight: US indices, Japan, commodities, FX.
GLOBAL_MARKETS = {
    "sp500": "^GSPC",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "nikkei": "^N225",
    "hang_seng": "^HSI",
    "ftse": "^FTSE",
    "crude": "CL=F",
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "usd_inr": "INR=X",
}

# Subset of yfinance's `info` dict we actually care about for fundamentals.
# The full dict has 100+ keys; most are noise. Picking a stable subset also
# means our cache entries stay small.
FUNDAMENTALS_KEYS = [
    "symbol",
    "shortName",
    "longName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "trailingEps",
    "forwardEps",
    "priceToBook",
    "dividendYield",
    "beta",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "averageVolume",
    "sharesOutstanding",
    "bookValue",
    "debtToEquity",
    "returnOnEquity",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "currency",
    "exchange",
]


class YFinanceProvider(BaseDataProvider):
    """
    Free market-data provider backed by Yahoo Finance.

    Usage:
        yf_provider = YFinanceProvider()
        candles = yf_provider.get_ohlcv("RELIANCE")          # NSE default
        quote = yf_provider.get_quote("INFY")
        fundies = yf_provider.get_fundamentals("TCS")
        nifty = yf_provider.get_index("nifty", period="1mo")
        globals_ = yf_provider.get_global_snapshot()
    """

    name = "yfinance"

    # yfinance raises YFRateLimitError and generic YFException under load or
    # when Yahoo's edge is flaky. ConnectionError / TimeoutError are already
    # retryable via the base class.
    retryable_exceptions = BaseDataProvider.retryable_exceptions + (
        YFRateLimitError,
        YFException,
    )

    def __init__(self, default_exchange: str = "NS", cache: Cache | None = None, **kwargs):
        """
        default_exchange: "NS" for NSE (default), "BO" for BSE. Concrete
        tickers can always override via fully-qualified symbols like
        "RELIANCE.BO" — this default just governs what `_qualify` does to
        unsuffixed inputs.
        """
        super().__init__(cache=cache, **kwargs)
        self.default_exchange = default_exchange

    # ── Ticker normalization ───────────────────────────────────────────────

    def _qualify(self, ticker: str) -> str:
        """
        Turn a raw ticker into a yfinance-ready symbol.

        Rules:
          - Starts with "^" (index) → leave alone.
          - Contains "=" or "-" (commodity/FX like "CL=F", "DX-Y.NYB") → leave alone.
          - Already has a "." suffix (RELIANCE.NS, RELIANCE.BO) → leave alone.
          - Otherwise, append ".{default_exchange}" (usually ".NS").

        This is a helper, not an enforcement — callers can still pass fully
        qualified tickers and they flow through unchanged.
        """
        ticker = ticker.strip().upper()
        if ticker.startswith("^") or "=" in ticker or "-" in ticker:
            return ticker
        if "." in ticker:
            return ticker
        return f"{ticker}.{self.default_exchange}"

    # ── Public API ─────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Return historical candles as a list of dicts.

        Each dict: {date, open, high, low, close, volume}.
        Dividends and stock splits columns are dropped — they're rarely useful
        for technical analysis and they bloat the cache.

        period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                  "1d", "5d", "1wk", "1mo", "3mo"

        TTL picks:
          - Intraday intervals (m, h) → OHLCV_INTRADAY (5 min)
          - Daily and above → OHLCV_DAILY (1 hr)
        """
        symbol = self._qualify(ticker)
        ttl = CacheTTL.OHLCV_INTRADAY if self._is_intraday(interval) else CacheTTL.OHLCV_DAILY

        def _fetch():
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df.empty:
                return []
            # Normalize: reset the DatetimeIndex into a column, lowercase
            # column names, drop useless cols, stringify dates (JSON-safe).
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            date_col = "date" if "date" in df.columns else "datetime"
            df[date_col] = df[date_col].astype(str)
            keep = [date_col, "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            return df.to_dict(orient="records")

        return self._cached_call(
            method="get_ohlcv",
            fetch_fn=_fetch,
            ttl=ttl,
            force_refresh=force_refresh,
            ticker=symbol,
            period=period,
            interval=interval,
        )

    def get_quote(self, ticker: str, force_refresh: bool = False) -> dict:
        """
        Snapshot quote via yfinance's `fast_info` — cheaper than full `info`.

        Returns a flat dict with last price, open/high/low, prev close, volume,
        and 50/200-day averages. This is the hot-path call that intraday
        monitoring would hit most frequently, so we give it a short TTL.
        """
        symbol = self._qualify(ticker)

        def _fetch():
            fi = yf.Ticker(symbol).fast_info
            # fast_info behaves like a dict-ish lazy loader. Pull what we need.
            return {
                "symbol": symbol,
                "last_price": fi.get("lastPrice"),
                "open": fi.get("open"),
                "day_high": fi.get("dayHigh"),
                "day_low": fi.get("dayLow"),
                "previous_close": fi.get("previousClose"),
                "volume": fi.get("lastVolume"),
                "market_cap": fi.get("marketCap"),
                "fifty_day_avg": fi.get("fiftyDayAverage"),
                "two_hundred_day_avg": fi.get("twoHundredDayAverage"),
                "currency": fi.get("currency"),
                "exchange": fi.get("exchange"),
            }

        return self._cached_call(
            method="get_quote",
            fetch_fn=_fetch,
            ttl=CacheTTL.QUOTE_LIVE,
            force_refresh=force_refresh,
            ticker=symbol,
        )

    def get_fundamentals(self, ticker: str, force_refresh: bool = False) -> dict:
        """
        Fundamentals snapshot — PE, EPS, margins, 52w range, etc.

        Uses yfinance's `info` dict but narrows to FUNDAMENTALS_KEYS so we
        don't cache 50 KB of trivia per ticker. Changes slowly, so a 24h TTL
        is comfortable.
        """
        symbol = self._qualify(ticker)

        def _fetch():
            info: dict[str, Any] = yf.Ticker(symbol).info or {}
            return {k: info.get(k) for k in FUNDAMENTALS_KEYS}

        return self._cached_call(
            method="get_fundamentals",
            fetch_fn=_fetch,
            ttl=CacheTTL.FUNDAMENTALS,
            force_refresh=force_refresh,
            ticker=symbol,
        )

    def get_index(
        self,
        index_key: str,
        period: str = "6mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Fetch an Indian index by friendly name (nifty, sensex, banknifty, vix).
        Falls through to `get_ohlcv` for the underlying Yahoo symbol.
        """
        key = index_key.lower().replace(" ", "_")
        symbol = INDEX_MAP.get(key)
        if symbol is None:
            raise ValueError(
                f"Unknown index '{index_key}'. Known: {sorted(INDEX_MAP.keys())}"
            )
        return self.get_ohlcv(symbol, period=period, interval=interval, force_refresh=force_refresh)

    def get_global_snapshot(self, force_refresh: bool = False) -> dict:
        """
        One-shot snapshot for the macro agent: latest close + 1-day %
        change for every symbol in GLOBAL_MARKETS.

        Cache TTL matches INDEX_SNAPSHOT (5 min). The macro agent runs once
        per morning, so a single call populates the cache and subsequent
        agent runs within the window are free.
        """

        def _fetch():
            out: dict[str, dict] = {}
            for friendly, symbol in GLOBAL_MARKETS.items():
                try:
                    # 5d gives us last close + prior close even on Monday
                    # (weekend gap) or long weekends.
                    df = yf.Ticker(symbol).history(period="5d", interval="1d")
                    if df.empty or len(df) < 2:
                        out[friendly] = {"symbol": symbol, "error": "no data"}
                        continue
                    last_close = float(df["Close"].iloc[-1])
                    prev_close = float(df["Close"].iloc[-2])
                    pct = ((last_close - prev_close) / prev_close) * 100.0
                    out[friendly] = {
                        "symbol": symbol,
                        "last_close": round(last_close, 4),
                        "prev_close": round(prev_close, 4),
                        "pct_change": round(pct, 3),
                    }
                except Exception as e:
                    # One failing symbol shouldn't kill the whole snapshot.
                    out[friendly] = {"symbol": symbol, "error": str(e)}
            return out

        return self._cached_call(
            method="get_global_snapshot",
            fetch_fn=_fetch,
            ttl=CacheTTL.INDEX_SNAPSHOT,
            force_refresh=force_refresh,
        )

    # ── Base class contract ────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Cheap liveness probe: fetch Nifty's latest daily candle."""
        try:
            df = yf.Ticker("^NSEI").history(period="1d", interval="1d")
            return not df.empty
        except Exception:
            return False

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_intraday(interval: str) -> bool:
        return interval.endswith(("m", "h"))
