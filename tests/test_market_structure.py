from risk.market_structure import (
    PATTERN_BIAS_CONTINUATION,
    PATTERN_BIAS_NEUTRAL,
    PATTERN_BIAS_RECLAIM,
    RETEST_BULL_HOLD,
    STRUCTURE_TREND_DOWN,
    STRUCTURE_TREND_UP,
    SWING_HH_HL,
    SWING_LH_LL,
    analyze_market_structure,
)


def _candles_from_closes(closes, start_ts=0, step_ms=900000, wick=1.0):
    candles = []
    for idx, close in enumerate(closes):
        close = float(close)
        open_price = close
        high = max(open_price, close) + wick
        low = min(open_price, close) - wick
        candles.append([start_ts + (idx * step_ms), open_price, high, low, close, 1000.0])
    return candles


def test_analyze_market_structure_detects_hh_hl_uptrend():
    ohlcv_15m = _candles_from_closes([100, 103, 101, 105, 102, 107, 103, 109, 105, 111, 107, 113, 109, 115, 111, 117])
    ohlcv_1h = _candles_from_closes([100, 104, 102, 107, 105, 110, 108, 113, 111, 116], step_ms=3600000, wick=1.5)

    structure = analyze_market_structure("TESTUSDT", ohlcv_15m, ohlcv_1h, side="LONG")

    assert structure["swingState"] == SWING_HH_HL
    assert structure["structureTrend"] == STRUCTURE_TREND_UP


def test_analyze_market_structure_detects_lh_ll_downtrend():
    ohlcv_15m = _candles_from_closes([117, 114, 116, 112, 114, 110, 112, 108, 110, 106, 108, 104, 106, 102, 104, 100])
    ohlcv_1h = _candles_from_closes([116, 112, 114, 109, 111, 106, 108, 103, 105, 100], step_ms=3600000, wick=1.5)

    structure = analyze_market_structure("TESTUSDT", ohlcv_15m, ohlcv_1h, side="SHORT")

    assert structure["swingState"] == SWING_LH_LL
    assert structure["structureTrend"] == STRUCTURE_TREND_DOWN


def test_analyze_market_structure_detects_bull_retest_continuation_bias():
    ohlcv_15m = _candles_from_closes(
        [100.0, 101.1, 100.4, 102.0, 101.2, 102.9, 102.1, 103.8, 103.0, 104.8, 104.0, 105.6, 105.4, 105.8, 105.7, 105.9],
        wick=0.05,
    )
    ohlcv_1h = _candles_from_closes([99.5, 100.9, 100.3, 101.8, 101.1, 103.0, 102.2, 104.2, 103.7, 105.3], step_ms=3600000, wick=0.08)

    structure = analyze_market_structure("TESTUSDT", ohlcv_15m, ohlcv_1h, side="LONG")

    assert structure["breakoutRetestState"] == RETEST_BULL_HOLD
    assert structure["patternBias"] == PATTERN_BIAS_CONTINUATION
    assert structure["patternConfidence"] >= 0.6


def test_analyze_market_structure_detects_reclaim_bias():
    ohlcv_15m = _candles_from_closes(
        [102.0, 100.8, 101.3, 99.9, 100.5, 99.1, 99.8, 99.0, 99.4, 98.95, 99.2, 98.9, 99.05, 98.98, 99.1, 99.02],
        wick=0.15,
    )
    ohlcv_1h = _candles_from_closes([101.8, 100.9, 101.1, 100.1, 100.4, 99.7, 99.9, 99.5, 99.7, 99.4], step_ms=3600000, wick=0.2)

    structure = analyze_market_structure("TESTUSDT", ohlcv_15m, ohlcv_1h, side="LONG")

    assert structure["patternBias"] == PATTERN_BIAS_RECLAIM
    assert structure["patternConfidence"] >= 0.6


def test_analyze_market_structure_returns_neutral_on_noisy_series():
    ohlcv_15m = _candles_from_closes([100.0, 100.2, 99.9, 100.25, 100.0, 100.22, 100.18, 100.12, 100.19, 100.23, 100.1, 100.16], wick=0.05)
    ohlcv_1h = _candles_from_closes([100.0, 100.1, 100.0, 100.12, 100.04, 100.1, 100.03, 100.08], step_ms=3600000, wick=0.08)

    structure = analyze_market_structure("TESTUSDT", ohlcv_15m, ohlcv_1h, side="LONG")

    assert structure["patternBias"] == PATTERN_BIAS_NEUTRAL
    assert structure["patternConfidence"] < 0.5
