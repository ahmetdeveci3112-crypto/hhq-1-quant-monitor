from __future__ import annotations

from typing import Iterable, List, Sequence


PATTERN_SOURCE = "TRADING_PATTERN_SCANNER_INSPIRED"
STRUCTURE_VERSION = "v1"

STRUCTURE_TREND_UP = "UP"
STRUCTURE_TREND_DOWN = "DOWN"
STRUCTURE_TREND_RANGE = "RANGE"
STRUCTURE_TREND_MIXED = "MIXED"

SWING_HH_HL = "HH_HL"
SWING_LH_LL = "LH_LL"
SWING_MIXED = "MIXED"
SWING_FLAT = "FLAT"

COMPRESSION_NONE = "NONE"
COMPRESSION_MILD = "MILD"
COMPRESSION_TIGHT = "TIGHT"
COMPRESSION_EXPANDING = "EXPANDING"

RETEST_NONE = "NONE"
RETEST_BULL_HOLD = "BULL_RETEST_HOLD"
RETEST_BEAR_HOLD = "BEAR_RETEST_HOLD"
RETEST_FAILED_BREAKOUT = "FAILED_BREAKOUT"
RETEST_FAILED_BREAKDOWN = "FAILED_BREAKDOWN"

SR_ABOVE_SUPPORT = "ABOVE_SUPPORT"
SR_BELOW_RESISTANCE = "BELOW_RESISTANCE"
SR_INSIDE_RANGE = "INSIDE_RANGE"
SR_AT_BREAKOUT_LEVEL = "AT_BREAKOUT_LEVEL"
SR_UNKNOWN = "UNKNOWN"

PATTERN_BIAS_CONTINUATION = "CONTINUATION"
PATTERN_BIAS_RECLAIM = "RECLAIM"
PATTERN_BIAS_NEUTRAL = "NEUTRAL"


def _default_structure_context() -> dict:
    return {
        "structureTrend": STRUCTURE_TREND_MIXED,
        "swingState": SWING_MIXED,
        "compressionState": COMPRESSION_NONE,
        "breakoutRetestState": RETEST_NONE,
        "srContext": SR_UNKNOWN,
        "patternBias": PATTERN_BIAS_NEUTRAL,
        "patternConfidence": 0.0,
        "patternSource": PATTERN_SOURCE,
        "structureVersion": STRUCTURE_VERSION,
    }


def _to_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _safe_close(candle) -> float:
    if not isinstance(candle, (list, tuple)) or len(candle) < 5:
        return 0.0
    return _to_float(candle[4], 0.0)


def _safe_high(candle) -> float:
    if not isinstance(candle, (list, tuple)) or len(candle) < 3:
        return 0.0
    return _to_float(candle[2], 0.0)


def _safe_low(candle) -> float:
    if not isinstance(candle, (list, tuple)) or len(candle) < 4:
        return 0.0
    return _to_float(candle[3], 0.0)


def _safe_ts(candle) -> int:
    if not isinstance(candle, (list, tuple)) or not candle:
        return 0
    try:
        return int(candle[0])
    except Exception:
        return 0


def _candle_range(candle) -> float:
    high = _safe_high(candle)
    low = _safe_low(candle)
    if high <= 0 or low <= 0:
        return 0.0
    return max(0.0, high - low)


def _mean(values: Iterable[float]) -> float:
    safe_values = [abs(_to_float(value, 0.0)) for value in values if _to_float(value, 0.0) > 0]
    if not safe_values:
        return 0.0
    return sum(safe_values) / len(safe_values)


def _recent_atr(candles: Sequence[Sequence], length: int = 14) -> float:
    if not candles:
        return 0.0
    sample = list(candles)[-length:]
    return _mean(_candle_range(candle) for candle in sample)


def _range_span(candles: Sequence[Sequence]) -> float:
    sample = list(candles)
    if not sample:
        return 0.0
    highs = [_safe_high(candle) for candle in sample if _safe_high(candle) > 0]
    lows = [_safe_low(candle) for candle in sample if _safe_low(candle) > 0]
    if not highs or not lows:
        return 0.0
    return max(0.0, max(highs) - min(lows))


def _extract_pivots(candles: Sequence[Sequence], left: int, right: int) -> tuple[list[dict], list[dict]]:
    highs: list[dict] = []
    lows: list[dict] = []
    sample = list(candles)
    if len(sample) < (left + right + 3):
        return highs, lows

    for idx in range(left, len(sample) - right):
        current = sample[idx]
        high = _safe_high(current)
        low = _safe_low(current)
        if high <= 0 or low <= 0:
            continue
        left_window = sample[idx - left:idx]
        right_window = sample[idx + 1:idx + 1 + right]
        if all(high >= _safe_high(candle) for candle in left_window + right_window):
            highs.append({"idx": idx, "price": high, "ts": _safe_ts(current)})
        if all(low <= _safe_low(candle) for candle in left_window + right_window):
            lows.append({"idx": idx, "price": low, "ts": _safe_ts(current)})

    return highs, lows


def _derive_swing_state(highs: Sequence[dict], lows: Sequence[dict], tolerance: float) -> str:
    if len(highs) < 2 or len(lows) < 2:
        return SWING_MIXED

    last_high = _to_float(highs[-1].get("price"), 0.0)
    prev_high = _to_float(highs[-2].get("price"), 0.0)
    last_low = _to_float(lows[-1].get("price"), 0.0)
    prev_low = _to_float(lows[-2].get("price"), 0.0)
    if min(last_high, prev_high, last_low, prev_low) <= 0:
        return SWING_MIXED

    high_delta = last_high - prev_high
    low_delta = last_low - prev_low
    tol = max(abs(tolerance), min(last_high, last_low) * 0.0005)
    if abs(high_delta) <= tol and abs(low_delta) <= tol:
        return SWING_FLAT
    if high_delta > tol and low_delta > tol:
        return SWING_HH_HL
    if high_delta < -tol and low_delta < -tol:
        return SWING_LH_LL
    return SWING_MIXED


def _derive_structure_trend(swing_state: str, candles_1h: Sequence[Sequence], tolerance: float) -> str:
    if swing_state == SWING_HH_HL:
        return STRUCTURE_TREND_UP
    if swing_state == SWING_LH_LL:
        return STRUCTURE_TREND_DOWN
    if swing_state == SWING_FLAT:
        return STRUCTURE_TREND_RANGE

    sample = list(candles_1h)[-8:]
    if len(sample) < 4:
        return STRUCTURE_TREND_MIXED
    last_close = _safe_close(sample[-1])
    first_close = _safe_close(sample[0])
    if min(last_close, first_close) <= 0:
        return STRUCTURE_TREND_MIXED
    delta = last_close - first_close
    if abs(delta) <= max(abs(tolerance), first_close * 0.001):
        return STRUCTURE_TREND_RANGE
    return STRUCTURE_TREND_UP if delta > 0 else STRUCTURE_TREND_DOWN


def _nearest_sr_levels(
    last_close: float,
    highs: Sequence[dict],
    lows: Sequence[dict],
    tolerance: float,
) -> tuple[float, float]:
    tol = max(abs(tolerance), max(last_close, 1.0) * 0.0005)
    supports = sorted(
        {
            round(_to_float(item.get("price"), 0.0), 8)
            for item in lows
            if _to_float(item.get("price"), 0.0) > 0 and _to_float(item.get("price"), 0.0) <= (last_close + tol)
        }
    )
    resistances = sorted(
        {
            round(_to_float(item.get("price"), 0.0), 8)
            for item in highs
            if _to_float(item.get("price"), 0.0) > 0 and _to_float(item.get("price"), 0.0) >= (last_close - tol)
        }
    )
    support = supports[-1] if supports else 0.0
    resistance = resistances[0] if resistances else 0.0
    return support, resistance


def _derive_sr_context(last_close: float, support: float, resistance: float, tolerance: float) -> str:
    tol = max(abs(tolerance), max(last_close, 1.0) * 0.0005)
    if support <= 0 and resistance <= 0:
        return SR_UNKNOWN
    if support > 0 and abs(last_close - support) <= tol:
        return SR_AT_BREAKOUT_LEVEL
    if resistance > 0 and abs(last_close - resistance) <= tol:
        return SR_AT_BREAKOUT_LEVEL
    if support > 0 and resistance > 0 and support < last_close < resistance:
        return SR_INSIDE_RANGE
    if support > 0 and last_close > support:
        return SR_ABOVE_SUPPORT
    if resistance > 0 and last_close < resistance:
        return SR_BELOW_RESISTANCE
    return SR_UNKNOWN


def _derive_compression_state(candles_15m: Sequence[Sequence], atr_distance: float) -> str:
    sample = list(candles_15m)
    if len(sample) < 24:
        return COMPRESSION_NONE
    recent = sample[-12:]
    prev = sample[-24:-12]
    recent_span = _range_span(recent)
    prev_span = _range_span(prev)
    recent_atr = _recent_atr(recent, length=12)
    prev_atr = _recent_atr(prev, length=12)
    baseline = max(prev_span, atr_distance, 1e-9)
    if recent_span <= baseline * 0.65 and recent_atr <= max(prev_atr, atr_distance) * 0.80:
        return COMPRESSION_TIGHT
    if recent_span <= baseline * 0.85 and recent_atr <= max(prev_atr, atr_distance) * 0.95:
        return COMPRESSION_MILD
    if recent_span >= baseline * 1.20 and recent_atr >= max(prev_atr, atr_distance) * 1.10:
        return COMPRESSION_EXPANDING
    return COMPRESSION_NONE


def _derive_swing_state_fallback(candles_15m: Sequence[Sequence], tolerance: float) -> str:
    sample = list(candles_15m)
    if len(sample) < 12:
        return SWING_MIXED
    prev = [_safe_close(candle) for candle in sample[-12:-6] if _safe_close(candle) > 0]
    recent = [_safe_close(candle) for candle in sample[-6:] if _safe_close(candle) > 0]
    if len(prev) < 4 or len(recent) < 4:
        return SWING_MIXED
    prev_high = max(prev)
    prev_low = min(prev)
    recent_high = max(recent)
    recent_low = min(recent)
    tol = max(abs(tolerance), max(recent_high, prev_high, 1.0) * 0.0005)
    if recent_high > prev_high + tol and recent_low > prev_low + tol:
        return SWING_HH_HL
    if recent_high < prev_high - tol and recent_low < prev_low - tol:
        return SWING_LH_LL
    return SWING_MIXED


def _recent_closes(candles: Sequence[Sequence], count: int = 3) -> list[float]:
    sample = list(candles)[-count:]
    return [_safe_close(candle) for candle in sample if _safe_close(candle) > 0]


def _derive_breakout_retest_state(
    last_close: float,
    recent_closes: Sequence[float],
    support: float,
    resistance: float,
    tolerance: float,
    side: str,
) -> str:
    tol = max(abs(tolerance), max(last_close, 1.0) * 0.0005)
    closes = [value for value in recent_closes if value > 0]
    if not closes:
        return RETEST_NONE

    above_res = resistance > 0 and max(closes) >= resistance + (tol * 0.30)
    below_res = resistance > 0 and last_close < resistance - tol
    hold_above_res = resistance > 0 and last_close >= resistance - tol and min(closes) >= resistance - (tol * 1.2)

    below_sup = support > 0 and min(closes) <= support - (tol * 0.30)
    above_sup = support > 0 and last_close > support + tol
    hold_below_sup = support > 0 and last_close <= support + tol and max(closes) <= support + (tol * 1.2)

    safe_side = str(side or "").upper()
    if safe_side == "LONG":
        if above_res and hold_above_res:
            return RETEST_BULL_HOLD
        if below_sup and above_sup:
            return RETEST_FAILED_BREAKDOWN
        if above_res and below_res:
            return RETEST_FAILED_BREAKOUT
        return RETEST_NONE
    if safe_side == "SHORT":
        if below_sup and hold_below_sup:
            return RETEST_BEAR_HOLD
        if above_res and below_res:
            return RETEST_FAILED_BREAKOUT
        if below_sup and above_sup:
            return RETEST_FAILED_BREAKDOWN
        return RETEST_NONE

    if above_res and hold_above_res:
        return RETEST_BULL_HOLD
    if below_sup and hold_below_sup:
        return RETEST_BEAR_HOLD
    if above_res and below_res:
        return RETEST_FAILED_BREAKOUT
    if below_sup and above_sup:
        return RETEST_FAILED_BREAKDOWN
    return RETEST_NONE


def _fallback_sr_levels(candles_15m: Sequence[Sequence], support: float, resistance: float) -> tuple[float, float]:
    if support > 0 and resistance > 0:
        return support, resistance
    sample = list(candles_15m)
    if len(sample) < 12:
        return support, resistance
    core = sample[-12:-3]
    highs = [_safe_high(candle) for candle in core if _safe_high(candle) > 0]
    lows = [_safe_low(candle) for candle in core if _safe_low(candle) > 0]
    if support <= 0 and lows:
        support = min(lows)
    if resistance <= 0 and highs:
        resistance = max(highs)
    return support, resistance


def analyze_market_structure(symbol, ohlcv_15m, ohlcv_1h, *, side: str = "") -> dict:
    context = _default_structure_context()
    candles_15m = list(ohlcv_15m or [])
    candles_1h = list(ohlcv_1h or [])
    if len(candles_15m) < 12 or len(candles_1h) < 8:
        return context

    last_close = _safe_close(candles_15m[-1]) or _safe_close(candles_1h[-1])
    if last_close <= 0:
        return context

    atr_15m = _recent_atr(candles_15m, length=14)
    atr_1h = _recent_atr(candles_1h, length=14)
    atr_distance = max(atr_15m, atr_1h, last_close * 0.001)
    sr_band = max(atr_distance * 0.45, last_close * 0.0015)
    swing_tol = max(atr_distance * 0.20, last_close * 0.001)

    highs_15m, lows_15m = _extract_pivots(candles_15m, left=2, right=2)
    highs_1h, lows_1h = _extract_pivots(candles_1h, left=3, right=3)
    highs = (highs_1h[-4:] + highs_15m[-6:])[-8:]
    lows = (lows_1h[-4:] + lows_15m[-6:])[-8:]

    swing_state = _derive_swing_state(highs, lows, swing_tol)
    if swing_state == SWING_MIXED:
        swing_state = _derive_swing_state_fallback(candles_15m, swing_tol)
    structure_trend = _derive_structure_trend(swing_state, candles_1h, swing_tol)
    support, resistance = _nearest_sr_levels(last_close, highs, lows, sr_band)
    support, resistance = _fallback_sr_levels(candles_15m, support, resistance)
    sr_context = _derive_sr_context(last_close, support, resistance, sr_band)
    compression_state = _derive_compression_state(candles_15m, atr_distance)
    breakout_retest_state = _derive_breakout_retest_state(
        last_close,
        _recent_closes(candles_15m, count=3),
        support,
        resistance,
        sr_band,
        side,
    )

    safe_side = str(side or "").upper()
    continuation_votes = 0
    reclaim_votes = 0
    total_votes = 3

    if safe_side == "LONG":
        if swing_state == SWING_HH_HL or structure_trend == STRUCTURE_TREND_UP:
            continuation_votes += 1
        if breakout_retest_state == RETEST_BULL_HOLD:
            continuation_votes += 1
        if sr_context in (SR_ABOVE_SUPPORT, SR_AT_BREAKOUT_LEVEL):
            continuation_votes += 1

        if structure_trend in (STRUCTURE_TREND_DOWN, STRUCTURE_TREND_MIXED, STRUCTURE_TREND_RANGE) or swing_state in (SWING_MIXED, SWING_FLAT):
            reclaim_votes += 1
        if sr_context in (SR_ABOVE_SUPPORT, SR_INSIDE_RANGE):
            reclaim_votes += 1
        if (
            breakout_retest_state not in (RETEST_FAILED_BREAKOUT, RETEST_FAILED_BREAKDOWN)
            and support > 0
            and last_close >= support
            and last_close <= support + (sr_band * 1.5)
            and sr_context in (SR_ABOVE_SUPPORT, SR_INSIDE_RANGE)
        ):
            reclaim_votes += 1
    elif safe_side == "SHORT":
        if swing_state == SWING_LH_LL or structure_trend == STRUCTURE_TREND_DOWN:
            continuation_votes += 1
        if breakout_retest_state == RETEST_BEAR_HOLD:
            continuation_votes += 1
        if sr_context in (SR_BELOW_RESISTANCE, SR_AT_BREAKOUT_LEVEL):
            continuation_votes += 1

        if structure_trend in (STRUCTURE_TREND_UP, STRUCTURE_TREND_MIXED, STRUCTURE_TREND_RANGE) or swing_state in (SWING_MIXED, SWING_FLAT):
            reclaim_votes += 1
        if sr_context in (SR_BELOW_RESISTANCE, SR_INSIDE_RANGE):
            reclaim_votes += 1
        if (
            breakout_retest_state not in (RETEST_FAILED_BREAKOUT, RETEST_FAILED_BREAKDOWN)
            and resistance > 0
            and last_close <= resistance
            and last_close >= resistance - (sr_band * 1.5)
            and sr_context in (SR_BELOW_RESISTANCE, SR_INSIDE_RANGE)
        ):
            reclaim_votes += 1
    else:
        if swing_state == SWING_HH_HL or structure_trend == STRUCTURE_TREND_UP:
            continuation_votes += 1
        if breakout_retest_state == RETEST_BULL_HOLD:
            continuation_votes += 1
        if sr_context in (SR_ABOVE_SUPPORT, SR_AT_BREAKOUT_LEVEL):
            continuation_votes += 1

        if swing_state in (SWING_MIXED, SWING_FLAT) or structure_trend in (STRUCTURE_TREND_MIXED, STRUCTURE_TREND_RANGE):
            reclaim_votes += 1
        if sr_context in (SR_INSIDE_RANGE, SR_AT_BREAKOUT_LEVEL):
            reclaim_votes += 1
        if breakout_retest_state == RETEST_NONE:
            reclaim_votes += 1

    strongest_vote = max(continuation_votes, reclaim_votes)

    if strongest_vote <= 1:
        pattern_bias = PATTERN_BIAS_NEUTRAL
        pattern_confidence = round((strongest_vote / total_votes) * 0.5, 4)
    elif continuation_votes >= 2 and continuation_votes > reclaim_votes:
        pattern_bias = PATTERN_BIAS_CONTINUATION
        pattern_confidence = min(1.0, continuation_votes / total_votes)
    elif reclaim_votes >= 2 and reclaim_votes >= continuation_votes:
        pattern_bias = PATTERN_BIAS_RECLAIM
        pattern_confidence = min(1.0, reclaim_votes / total_votes)
    else:
        pattern_bias = PATTERN_BIAS_NEUTRAL
        pattern_confidence = round(max(continuation_votes, reclaim_votes) / total_votes * 0.5, 4)

    context.update(
        {
            "structureTrend": structure_trend,
            "swingState": swing_state,
            "compressionState": compression_state,
            "breakoutRetestState": breakout_retest_state,
            "srContext": sr_context,
            "patternBias": pattern_bias,
            "patternConfidence": round(max(0.0, min(1.0, pattern_confidence)), 4),
        }
    )
    return context
