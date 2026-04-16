"""
Unit tests for manually-implemented indicators.

No network, no LLM — every test uses a deterministic synthetic series so we
can assert exact numeric outputs. Property tests (e.g., "SMA of a constant
series equals that constant after warm-up") catch the class of bugs that
silently hide in floating-point indicator math.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.technical.indicators import sma


def test_sma_constant_series():
    print("TEST: SMA of constant series equals that constant (after warm-up)")
    result = sma([5.0, 5.0, 5.0, 5.0, 5.0], 3)
    assert result == [None, None, 5.0, 5.0, 5.0], f"got {result}"
    print("  PASSED")


def test_sma_rolling_window():
    print("TEST: SMA of [1..5] window=3")
    result = sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    # avg([1,2,3])=2, avg([2,3,4])=3, avg([3,4,5])=4
    assert result == [None, None, 2.0, 3.0, 4.0], f"got {result}"
    print("  PASSED")


def test_sma_window_larger_than_series():
    print("TEST: SMA with window > len returns all None")
    result = sma([1.0, 2.0], 5)
    assert result == [None, None], f"got {result}"
    print("  PASSED")


def test_sma_invalid_window():
    print("TEST: SMA raises ValueError on window < 1")
    try:
        sma([1.0, 2.0], 0)
    except ValueError:
        print("  PASSED")
        return
    raise AssertionError("expected ValueError")


from agents.technical.indicators import ema


def test_ema_constant_series():
    print("TEST: EMA of constant series equals that constant after warm-up")
    result = ema([7.0] * 10, 3)
    # Warm-up: first 2 None, seed at index 2 = SMA of first 3 = 7.0
    assert result[0] is None and result[1] is None, f"got {result}"
    for v in result[2:]:
        assert math.isclose(v, 7.0, abs_tol=1e-9), f"got {v}"
    print("  PASSED")


def test_ema_converges_above_sma_on_rising_series():
    print("TEST: EMA tracks rising series faster than SMA")
    # Use an accelerating (non-linear) series where recent values are higher
    values = [float(i ** 1.5) for i in range(1, 21)]  # 1^1.5, 2^1.5, ..., 20^1.5
    ema_result = ema(values, 5)
    sma_result = sma(values, 5)
    # On the last value, EMA should be closer to the latest values than SMA.
    assert ema_result[-1] > sma_result[-1], (
        f"ema_last={ema_result[-1]}, sma_last={sma_result[-1]}"
    )
    print("  PASSED")


def test_ema_invalid_window():
    print("TEST: EMA raises ValueError on window < 1")
    try:
        ema([1.0, 2.0], 0)
    except ValueError:
        print("  PASSED")
        return
    raise AssertionError("expected ValueError")


from agents.technical.indicators import rsi


def test_rsi_monotonic_rising_equals_100():
    print("TEST: RSI of monotonically rising series → 100")
    values = [float(i) for i in range(30)]
    result = rsi(values, 14)
    # First 14 positions should be None.
    assert all(v is None for v in result[:14]), f"warm-up has non-None: {result[:14]}"
    # Values from position 14 onwards should be 100 (no losses).
    for i, v in enumerate(result[14:], start=14):
        assert v == 100.0, f"position {i}: expected 100.0, got {v}"
    print("  PASSED")


def test_rsi_monotonic_falling_equals_0():
    print("TEST: RSI of monotonically falling series → 0")
    values = [float(i) for i in range(30, 0, -1)]
    result = rsi(values, 14)
    for i, v in enumerate(result[14:], start=14):
        assert v == 0.0, f"position {i}: expected 0.0, got {v}"
    print("  PASSED")


def test_rsi_constant_series_is_none():
    print("TEST: RSI of constant series is None (no gains, no losses)")
    values = [5.0] * 30
    result = rsi(values, 14)
    for v in result[14:]:
        assert v is None, f"expected None, got {v}"
    print("  PASSED")


def test_rsi_warm_up_length():
    print("TEST: RSI warm-up length equals window")
    result = rsi([float(i) for i in range(30)], 14)
    nones = [v for v in result if v is None]
    assert len(nones) == 14, f"expected 14 None values, got {len(nones)}"
    print("  PASSED")


from agents.technical.indicators import macd


def test_macd_returns_three_aligned_series():
    print("TEST: MACD returns aligned macd/signal/hist lists")
    values = [float(i) + (i % 3) for i in range(60)]  # gentle zig-zag rising
    result = macd(values)
    assert set(result.keys()) == {"macd", "signal", "hist"}, f"keys={result.keys()}"
    n = len(values)
    for key in ("macd", "signal", "hist"):
        assert len(result[key]) == n, f"{key} length {len(result[key])} != {n}"
    print("  PASSED")


def test_macd_hist_equals_macd_minus_signal():
    print("TEST: MACD hist = macd − signal where both defined")
    values = [float(i) + (i % 3) for i in range(60)]
    result = macd(values)
    for m, s, h in zip(result["macd"], result["signal"], result["hist"]):
        if m is None or s is None:
            assert h is None, f"hist should be None when macd or signal is None"
        else:
            assert math.isclose(h, m - s, abs_tol=1e-9), f"hist {h} != {m - s}"
    print("  PASSED")


from agents.technical.indicators import bollinger


def test_bollinger_band_width_on_constant_is_zero():
    print("TEST: Bollinger bands collapse on constant series")
    values = [10.0] * 30
    result = bollinger(values, window=20, num_std=2)
    for i in range(19, 30):
        assert result["middle"][i] == 10.0
        # Std of constant is 0, so upper == middle == lower.
        assert math.isclose(result["upper"][i], 10.0, abs_tol=1e-9)
        assert math.isclose(result["lower"][i], 10.0, abs_tol=1e-9)
    print("  PASSED")


def test_bollinger_upper_above_middle_above_lower_on_variable():
    print("TEST: Bollinger upper > middle > lower on variable series")
    values = [float(i % 10) for i in range(40)]
    result = bollinger(values, window=20, num_std=2)
    for i in range(19, 40):
        u, m, l = result["upper"][i], result["middle"][i], result["lower"][i]
        assert u > m > l, f"at {i}: upper={u}, middle={m}, lower={l}"
    print("  PASSED")


from agents.technical.indicators import atr


def _fake_ohlc_constant_range(n: int, rng: float, close: float):
    """Generate n candles with constant range `rng` and constant close."""
    return [
        {"high": close + rng / 2, "low": close - rng / 2, "close": close}
        for _ in range(n)
    ]


def test_atr_constant_range_equals_range():
    print("TEST: ATR on constant-range candles equals that range")
    ohlc = _fake_ohlc_constant_range(30, rng=4.0, close=100.0)
    result = atr(ohlc, window=14)
    # First 14 positions None; from index 14 onwards ATR should equal 4.0.
    assert all(v is None for v in result[:14]), f"warm-up non-None: {result[:14]}"
    for i, v in enumerate(result[14:], start=14):
        assert math.isclose(v, 4.0, abs_tol=1e-9), f"at {i}: {v}"
    print("  PASSED")


def test_atr_length_matches_input():
    print("TEST: ATR output length equals input length")
    ohlc = _fake_ohlc_constant_range(50, 2.0, 100.0)
    result = atr(ohlc, window=14)
    assert len(result) == 50, f"got {len(result)}"
    print("  PASSED")


from agents.technical.indicators import volume_ratio


def test_volume_ratio_constant_is_one():
    print("TEST: volume_ratio of constant volume equals 1.0")
    vols = [1000.0] * 30
    result = volume_ratio(vols, window=20)
    for i, v in enumerate(result[19:], start=19):
        assert math.isclose(v, 1.0, abs_tol=1e-9), f"at {i}: {v}"
    print("  PASSED")


def test_volume_ratio_spike():
    print("TEST: volume_ratio detects spike above average")
    vols = [100.0] * 19 + [200.0] + [100.0] * 10
    result = volume_ratio(vols, window=20)
    # At index 19, window is [100×19, 200] → mean 105, today 200 → ratio ≈ 1.905
    assert result[19] is not None and result[19] > 1.8, f"got {result[19]}"
    print("  PASSED")


if __name__ == "__main__":
    test_sma_constant_series()
    test_sma_rolling_window()
    test_sma_window_larger_than_series()
    test_sma_invalid_window()
    test_ema_constant_series()
    test_ema_converges_above_sma_on_rising_series()
    test_ema_invalid_window()
    test_rsi_monotonic_rising_equals_100()
    test_rsi_monotonic_falling_equals_0()
    test_rsi_constant_series_is_none()
    test_rsi_warm_up_length()
    test_macd_returns_three_aligned_series()
    test_macd_hist_equals_macd_minus_signal()
    test_bollinger_band_width_on_constant_is_zero()
    test_bollinger_upper_above_middle_above_lower_on_variable()
    test_atr_constant_range_equals_range()
    test_atr_length_matches_input()
    test_volume_ratio_constant_is_one()
    test_volume_ratio_spike()
    print("\nAll indicator tests passed.")
