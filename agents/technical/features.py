"""
Feature-snapshot builder — the bridge between raw OHLCV and the LLM.

The LLM is never shown 250 candles directly. Instead, we run the indicator
library over the series and produce one small dict per ticker containing
the current indicator values plus derived categorical labels
("above_sma200", "oversold", "bearish_cross_5d_ago"). That dict is what
the LLM sees; interpretation and decision-making happens on a compact,
pre-computed summary.

Tickers with fewer than MIN_DAILY_CANDLES days of history are unusable
(no SMA 200). Callers should filter them out via `insufficient_history`
before calling `build_snapshot`.
"""

from __future__ import annotations

import math

from agents.technical.indicators import (
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
    volume_ratio,
)

MIN_DAILY_CANDLES = 200


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _drop_invalid(candles: list[dict]) -> list[dict]:
    """
    Drop candles where any OHLC field is None or NaN.
    yfinance emits a partial candle for the current session before it closes;
    those rows have NaN OHLC and would poison every downstream indicator.
    """
    return [
        c for c in candles
        if all(
            c.get(k) is not None and not _is_nan(c.get(k))
            for k in ("open", "high", "low", "close")
        )
    ]


def insufficient_history(daily_candles: list[dict]) -> bool:
    return len(daily_candles) < MIN_DAILY_CANDLES


def _rsi_state(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 30:
        return "oversold"
    if value >= 70:
        return "overbought"
    return "neutral"


def _macd_state(hist_series: list[float | None]) -> str:
    """Classify the most recent MACD histogram crossover, if any."""
    recent = [h for h in hist_series[-20:] if h is not None]
    if len(recent) < 2:
        return "unknown"
    last = recent[-1]
    # Find the most recent sign change.
    for lookback in range(1, len(recent)):
        prev = recent[-1 - lookback]
        if (last > 0 and prev <= 0) or (last < 0 and prev >= 0):
            direction = "bullish_cross" if last > 0 else "bearish_cross"
            return f"{direction}_{lookback}d_ago"
    return "bullish" if last > 0 else "bearish"


def _trend_label(last_close: float, sma200: float | None) -> str:
    if sma200 is None:
        return "unknown"
    if last_close > sma200 * 1.02:
        return "above_sma200"
    if last_close < sma200 * 0.98:
        return "below_sma200"
    return "between"


def _block(candles: list[dict]) -> dict:
    """Compute the indicator block for one timeframe (daily or weekly)."""
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    last = closes[-1]

    sma20_v = sma(closes, 20)[-1]
    sma50_v = sma(closes, 50)[-1]
    sma200_v = sma(closes, 200)[-1] if len(closes) >= 200 else None
    ema9_v = ema(closes, 9)[-1]
    ema21_v = ema(closes, 21)[-1]
    rsi14_v = rsi(closes, 14)[-1]
    macd_v = macd(closes)
    bb_v = bollinger(closes, 20, 2.0)
    atr14_series = atr(candles, 14)
    atr14_v = atr14_series[-1]
    vol_ratio_v = volume_ratio(volumes, 20)[-1]

    bb_upper = bb_v["upper"][-1]
    bb_lower = bb_v["lower"][-1]
    bb_position: float | None = None
    if bb_upper is not None and bb_lower is not None and bb_upper > bb_lower:
        bb_position = (last - bb_lower) / (bb_upper - bb_lower)

    return {
        "sma20": sma20_v,
        "sma50": sma50_v,
        "sma200": sma200_v,
        "ema9": ema9_v,
        "ema21": ema21_v,
        "rsi14": rsi14_v,
        "rsi_state": _rsi_state(rsi14_v),
        "macd": macd_v["macd"][-1],
        "macd_signal": macd_v["signal"][-1],
        "macd_hist": macd_v["hist"][-1],
        "macd_state": _macd_state(macd_v["hist"]),
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "atr14": atr14_v,
        "atr_pct": (atr14_v / last * 100) if atr14_v is not None else None,
        "volume_ratio_20d": vol_ratio_v,
        "trend_label": _trend_label(last, sma200_v),
        "pct_from_sma200": (
            ((last - sma200_v) / sma200_v * 100) if sma200_v else None
        ),
    }


def build_snapshot(
    ticker: str,
    daily_candles: list[dict],
    weekly_candles: list[dict],
    as_of: str,
) -> dict:
    """
    Build the compact feature snapshot for one ticker. Input candles are
    JSON-serializable dicts (as produced by YFinanceProvider.get_ohlcv).
    """
    daily_candles = _drop_invalid(daily_candles)
    weekly_candles = _drop_invalid(weekly_candles)
    return {
        "ticker": ticker,
        "last_close": daily_candles[-1]["close"],
        "as_of": as_of,
        "daily": _block(daily_candles),
        "weekly": _block(weekly_candles),
        "recent_candles_5d": daily_candles[-5:],
    }
