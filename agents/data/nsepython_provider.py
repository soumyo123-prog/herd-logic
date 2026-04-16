"""
NSEPythonProvider — Tier 1 source for NSE India data that yfinance doesn't have.

yfinance gives us OHLCV and fundamentals but not the India-specific signals
that matter for swing trading:

  - **FII/DII flow** — daily net buy/sell by foreign and domestic institutions.
    The single strongest macro signal for the Indian market.
  - **Option chain** — strike-level OI and PE/CE flow → PCR, max-pain, support/
    resistance from open interest walls.
  - **Top gainers / losers** — sector rotation signals.
  - **Bulk and block deals** — institutional buying/selling footprints.
  - **Market status** — is the market open? (needed for TTL decisions)
  - **Delivery %** — quality of volume (high delivery = less speculative).

Why a separate provider instead of extending YFinanceProvider?

  - Different upstream (NSE's JSON endpoints vs Yahoo's).
  - Different failure modes (NSE uses cookie-based anti-scraping, so it
    rejects more requests; its retryable exception set is larger).
  - Different data shapes (mostly DataFrames, some nested dicts).

Each provider stays single-purpose — one library, one set of quirks. If NSE
changes their API tomorrow, we patch here without touching YFinanceProvider.

Caveat: NSE actively fights scrapers. Expect periodic failures that no amount
of retry fixes until nsepython pushes an update. When a method fails
permanently we surface `ProviderError` and the caller decides whether to
degrade (skip this signal for this cycle) or abort.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import nsepython as nse

from agents.data.base_provider import BaseDataProvider, CacheTTL
from agents.data.cache import Cache


# Friendly → NSE name. NSE uses spaces in index names ("NIFTY 50") which is
# awkward to pass around. Callers use snake-case keys.
INDEX_NAME_MAP = {
    "nifty": "NIFTY 50",
    "nifty50": "NIFTY 50",
    "nifty_next50": "NIFTY NEXT 50",
    "banknifty": "NIFTY BANK",
    "finnifty": "NIFTY FIN SERVICE",
    "midcap": "NIFTY MIDCAP 100",
    "smallcap": "NIFTY SMLCAP 100",
}

# Subset of nsepython's nse_eq() priceInfo we actually care about. The full
# response includes ~30 nested sections; most aren't useful day-to-day.
EQUITY_PRICE_KEYS = [
    "lastPrice",
    "change",
    "pChange",
    "previousClose",
    "open",
    "close",
    "vwap",
    "lowerCP",
    "upperCP",
]


class NSEPythonProvider(BaseDataProvider):
    """
    Wraps nsepython with caching and retry. Methods below return plain
    JSON-serializable structures (list of dicts, dict) so they slot into
    the shared cache layer without custom serializers.
    """

    name = "nsepython"

    # nsepython itself wraps `requests`. Failures surface as:
    #   - ConnectionError / TimeoutError (handled by base class)
    #   - json.JSONDecodeError when NSE returns an HTML "access denied" page
    #   - KeyError / IndexError when the response is a partial dict
    # We retry those because they're almost always transient cookie/token issues.
    retryable_exceptions = BaseDataProvider.retryable_exceptions + (
        json.JSONDecodeError,
        KeyError,
        IndexError,
    )

    def __init__(self, cache: Cache | None = None, **kwargs):
        # NSE is much flakier than Yahoo — bump default retries / give
        # upstream more time to recover.
        kwargs.setdefault("max_retries", 4)
        kwargs.setdefault("base_delay", 2.0)
        super().__init__(cache=cache, **kwargs)

    # ── Market status ──────────────────────────────────────────────────────

    def get_market_status(self, force_refresh: bool = False) -> dict:
        """
        Whether capital / currency / commodity markets are open right now.
        Used by the orchestrator to decide cadence: no point running the
        technical agent at midnight against stale mid-day candles.

        Short TTL because the state flips at 9:15 and 15:30 IST sharp.
        """

        def _fetch():
            raw = nse.nse_marketStatus()
            return {
                "updated_at": raw.get("marketcap", {}).get("timeStamp"),
                "markets": {
                    m.get("market", "?"): {
                        "status": m.get("marketStatus"),
                        "trade_date": m.get("tradeDate"),
                        "message": m.get("marketStatusMessage"),
                    }
                    for m in raw.get("marketState", [])
                },
                "nifty_close": raw.get("indicativenifty50", {}).get("finalClosingValue"),
            }

        return self._cached_call(
            method="get_market_status",
            fetch_fn=_fetch,
            ttl=CacheTTL.QUOTE_LIVE,
            force_refresh=force_refresh,
        )

    # ── FII/DII flow ───────────────────────────────────────────────────────

    def get_fii_dii(self, force_refresh: bool = False) -> list[dict]:
        """
        Daily net FII and DII activity (crores INR, buy/sell/net).

        Released once per trading day around 6 PM IST. 1h TTL is generous
        enough to avoid hammering NSE but still refreshes multiple times
        post-release.
        """

        def _fetch():
            df = nse.nse_fiidii()
            return _df_to_records(df)

        return self._cached_call(
            method="get_fii_dii",
            fetch_fn=_fetch,
            ttl=CacheTTL.FII_DII,
            force_refresh=force_refresh,
        )

    # ── Market movers ──────────────────────────────────────────────────────

    def get_top_gainers(self, force_refresh: bool = False) -> list[dict]:
        """
        Top gainers across all of NSE (nsepython doesn't parametrize by
        index — it returns the broad-market leaders).
        """

        def _fetch():
            df = nse.nse_get_top_gainers()
            return _trim_movers(df)

        return self._cached_call(
            method="get_top_gainers",
            fetch_fn=_fetch,
            ttl=CacheTTL.INDEX_SNAPSHOT,
            force_refresh=force_refresh,
        )

    def get_top_losers(self, force_refresh: bool = False) -> list[dict]:
        """Top losers across all of NSE. See `get_top_gainers` note."""

        def _fetch():
            df = nse.nse_get_top_losers()
            return _trim_movers(df)

        return self._cached_call(
            method="get_top_losers",
            fetch_fn=_fetch,
            ttl=CacheTTL.INDEX_SNAPSHOT,
            force_refresh=force_refresh,
        )

    # ── Deal activity (institutional footprints) ───────────────────────────

    def get_bulk_deals(self, force_refresh: bool = False) -> list[dict]:
        """
        Bulk deals (any trade > 0.5% of a company's equity).
        Reported end-of-day. TTL matches FII/DII (1h).
        """

        def _fetch():
            df = nse.get_bulkdeals()
            return _df_to_records(df)

        return self._cached_call(
            method="get_bulk_deals",
            fetch_fn=_fetch,
            ttl=CacheTTL.FII_DII,
            force_refresh=force_refresh,
        )

    def get_block_deals(self, force_refresh: bool = False) -> list[dict]:
        """
        Block deals (negotiated trades at a minimum 5 lakh shares / ₹5 Cr).
        Stronger institutional signal than bulk deals.
        """

        def _fetch():
            df = nse.get_blockdeals()
            return _df_to_records(df)

        return self._cached_call(
            method="get_block_deals",
            fetch_fn=_fetch,
            ttl=CacheTTL.FII_DII,
            force_refresh=force_refresh,
        )

    # ── Option chain ───────────────────────────────────────────────────────

    def get_option_chain(self, symbol: str, force_refresh: bool = False) -> dict:
        """
        Narrowed option chain: expiry list + strike-level CE/PE OI and LTP.
        Returns `{"expiries": [...], "rows": [...]}`. Each row:
            {strike, ce_oi, ce_change_oi, ce_ltp, pe_oi, pe_change_oi, pe_ltp}

        Off-market hours NSE returns an empty `records.data` — we pass that
        through as `rows: []` rather than raising, so the sentiment agent
        can render "option chain unavailable" gracefully.
        """
        symbol = symbol.upper().strip()

        def _fetch():
            raw = nse.nse_optionchain_scrapper(symbol)
            records = raw.get("records", {}) or {}
            expiries = records.get("expiryDates", []) or []
            data = records.get("data", []) or []

            rows = []
            for item in data:
                ce = item.get("CE", {}) or {}
                pe = item.get("PE", {}) or {}
                rows.append({
                    "strike": item.get("strikePrice"),
                    "expiry": item.get("expiryDate"),
                    "ce_oi": ce.get("openInterest"),
                    "ce_change_oi": ce.get("changeinOpenInterest"),
                    "ce_ltp": ce.get("lastPrice"),
                    "ce_volume": ce.get("totalTradedVolume"),
                    "pe_oi": pe.get("openInterest"),
                    "pe_change_oi": pe.get("changeinOpenInterest"),
                    "pe_ltp": pe.get("lastPrice"),
                    "pe_volume": pe.get("totalTradedVolume"),
                })
            return {
                "symbol": symbol,
                "underlying_value": records.get("underlyingValue"),
                "expiries": expiries,
                "rows": rows,
            }

        return self._cached_call(
            method="get_option_chain",
            fetch_fn=_fetch,
            ttl=CacheTTL.OPTION_CHAIN,
            force_refresh=force_refresh,
            symbol=symbol,
        )

    # ── Equity and index snapshots ─────────────────────────────────────────

    def get_equity_snapshot(self, ticker: str, force_refresh: bool = False) -> dict:
        """
        Curated price snapshot for an NSE equity. Mirrors yfinance's quote
        but from NSE's own endpoint — useful when Yahoo lags or disagrees.
        """
        ticker = ticker.upper().strip()

        def _fetch():
            raw = nse.nse_eq(ticker)
            price_info = (raw or {}).get("priceInfo", {}) or {}
            return {
                "symbol": ticker,
                **{k: price_info.get(k) for k in EQUITY_PRICE_KEYS},
            }

        return self._cached_call(
            method="get_equity_snapshot",
            fetch_fn=_fetch,
            ttl=CacheTTL.QUOTE_LIVE,
            force_refresh=force_refresh,
            ticker=ticker,
        )

    def get_index_quote(self, index_key: str, force_refresh: bool = False) -> dict:
        """
        Lightweight index quote (open/high/low/last/yearHigh/yearLow).
        """
        nse_index = _resolve_index(index_key)

        def _fetch():
            raw = nse.nse_get_index_quote(nse_index)
            return {
                "index": nse_index,
                "last": _to_float(raw.get("last")),
                "open": _to_float(raw.get("open")),
                "high": _to_float(raw.get("high")),
                "low": _to_float(raw.get("low")),
                "previous_close": _to_float(raw.get("previousClose")),
                "percent_change": _to_float(raw.get("percChange")),
                "year_high": _to_float(raw.get("yearHigh")),
                "year_low": _to_float(raw.get("yearLow")),
                "time_val": raw.get("timeVal"),
            }

        return self._cached_call(
            method="get_index_quote",
            fetch_fn=_fetch,
            ttl=CacheTTL.INDEX_SNAPSHOT,
            force_refresh=force_refresh,
            index=nse_index,
        )

    # ── Base class contract ────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Cheap liveness probe — fetch market status."""
        try:
            status = nse.nse_marketStatus()
            return "marketState" in (status or {})
        except Exception:
            return False


# ── Module-level helpers ────────────────────────────────────────────────

def _resolve_index(key: str) -> str:
    """Map friendly key (nifty/banknifty) → NSE's verbose name ('NIFTY 50')."""
    k = key.lower().replace(" ", "_")
    if k in INDEX_NAME_MAP:
        return INDEX_NAME_MAP[k]
    # If caller already passed a qualified NSE name, let it through.
    if " " in key or key.isupper():
        return key
    raise ValueError(
        f"Unknown index '{key}'. Known keys: {sorted(INDEX_NAME_MAP.keys())}"
    )


def _df_to_records(df: Any) -> list[dict]:
    """
    Convert a nsepython DataFrame (or None / empty) to a JSON-safe list.
    Drops the `meta` column when present — it holds huge nested company
    records that bloat the cache.
    """
    if df is None:
        return []
    if not isinstance(df, pd.DataFrame):
        # Some nsepython functions occasionally return lists already.
        return df if isinstance(df, list) else []
    if df.empty:
        return []
    drop_cols = [c for c in ("meta",) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # Timestamps → strings so JSON serialization doesn't fail.
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    return df.to_dict(orient="records")


def _trim_movers(df: Any) -> list[dict]:
    """
    Top gainers/losers DataFrames include 24 columns; most are heavy
    metadata. Keep only the fields an agent needs to reason about movers.
    """
    keep = [
        "symbol", "ltp", "open_price", "high_price", "low_price",
        "prev_price", "net_price", "perChange", "trade_quantity", "turnover",
    ]
    records = _df_to_records(df)
    trimmed = []
    for r in records:
        trimmed.append({k: r.get(k) for k in keep if k in r})
    return trimmed


def _to_float(val: Any) -> float | None:
    """Safely coerce '25,696.05' or '0.07' to float; return None on failure."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None
