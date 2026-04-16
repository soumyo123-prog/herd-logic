"""
Manually-implemented technical indicators.

Pure functions — input lists, return lists aligned by index, with `None`
for warm-up positions where insufficient data exists. No pandas, no numpy
beyond what the standard library and pure Python provide — keeps behavior
predictable and inspectable.

Every indicator has unit tests in tests/test_indicators.py covering
property-level invariants (SMA of constant → constant, RSI monotonic →
100, etc.).
"""

from __future__ import annotations


def sma(values: list[float], window: int) -> list[float | None]:
    """Simple moving average. First `window - 1` positions are None."""
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float | None] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)
            continue
        window_values = values[i - window + 1 : i + 1]
        result.append(sum(window_values) / window)
    return result


def ema(values: list[float], window: int) -> list[float | None]:
    """
    Exponential moving average. Seeded with the SMA of the first `window`
    values, then EMA_t = α·V_t + (1−α)·EMA_{t-1} with α = 2/(window+1).

    Positions 0..window-2 are None; position window-1 holds the SMA seed.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(values) < window:
        return [None] * len(values)

    alpha = 2.0 / (window + 1)
    result: list[float | None] = [None] * (window - 1)
    seed = sum(values[:window]) / window
    result.append(seed)
    prev = seed
    for v in values[window:]:
        prev = alpha * v + (1 - alpha) * prev
        result.append(prev)
    return result


def rsi(values: list[float], window: int = 14) -> list[float | None]:
    """
    Relative Strength Index using Wilder's smoothing.

    RSI = 100 − 100 / (1 + RS), where RS = avg_gain / avg_loss.
    First `window` positions are None. The first RSI value lands at
    index `window` (requires `window` price changes to compute).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(values) <= window:
        return [None] * len(values)

    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    def _rsi_from(ag: float, al: float) -> float | None:
        if al == 0:
            return 100.0 if ag > 0 else None
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    result: list[float | None] = [None] * len(values)
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    result[window] = _rsi_from(avg_gain, avg_loss)

    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        result[i + 1] = _rsi_from(avg_gain, avg_loss)

    return result


def macd(
    values: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[float | None]]:
    """
    MACD: difference of fast and slow EMAs, plus a signal-line EMA and
    the histogram (macd − signal). All three returned lists align with
    the input length; None-padded wherever inputs are warming up.
    """
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)

    macd_line: list[float | None] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_ema, slow_ema)
    ]

    # Signal EMA is computed on the defined tail of macd_line, then left-padded
    # back to the full length with None.
    tail = [m for m in macd_line if m is not None]
    tail_signal = ema(tail, signal)
    pad = [None] * (len(values) - len(tail))
    signal_line: list[float | None] = pad + tail_signal

    hist: list[float | None] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]

    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def bollinger(
    values: list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> dict[str, list[float | None]]:
    """
    Bollinger bands around the SMA. Standard deviation uses the population
    formula (divide by window, not window-1) to match most charting tools.
    """
    middle = sma(values, window)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)

    for i in range(window - 1, len(values)):
        m = middle[i]
        assert m is not None  # sma guarantees non-None from position window-1
        window_values = values[i - window + 1 : i + 1]
        variance = sum((x - m) ** 2 for x in window_values) / window
        std = variance ** 0.5
        upper[i] = m + num_std * std
        lower[i] = m - num_std * std

    return {"middle": middle, "upper": upper, "lower": lower}


def atr(
    ohlc: list[dict],
    window: int = 14,
) -> list[float | None]:
    """
    Average True Range using Wilder's smoothing.

    Expects `ohlc` as a list of dicts with keys `high`, `low`, `close` in
    chronological order. The first TR is undefined (no previous close);
    positions 0..window-1 are None. First ATR lands at index `window`,
    seeded with the simple mean of TR[1..window].
    """
    n = len(ohlc)
    if n < 2 or window < 1:
        return [None] * n

    tr_list: list[float | None] = [None]
    for i in range(1, n):
        h = ohlc[i]["high"]
        l = ohlc[i]["low"]
        pc = ohlc[i - 1]["close"]
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

    if n <= window:
        return [None] * n

    result: list[float | None] = [None] * window
    seed = sum(tr_list[1 : window + 1]) / window  # type: ignore[arg-type]
    result.append(seed)
    prev = seed
    for i in range(window + 1, n):
        prev = (prev * (window - 1) + tr_list[i]) / window  # type: ignore[operator]
        result.append(prev)
    return result


def volume_ratio(
    volumes: list[float],
    window: int = 20,
) -> list[float | None]:
    """
    Today's volume divided by the SMA of volume over `window`.
    Values > 1 mean above-average volume (useful breakout confirmation).
    """
    vol_sma = sma(volumes, window)
    return [
        (v / s) if (s is not None and s > 0) else None
        for v, s in zip(volumes, vol_sma)
    ]
