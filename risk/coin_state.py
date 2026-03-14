from __future__ import annotations

from typing import Sequence

from risk.market_structure import (
    RETEST_BEAR_HOLD,
    RETEST_BULL_HOLD,
    RETEST_FAILED_BREAKDOWN,
    RETEST_FAILED_BREAKOUT,
    STRUCTURE_TREND_DOWN,
    STRUCTURE_TREND_RANGE,
    STRUCTURE_TREND_UP,
)


COIN_STATE_SOURCE = "INTERNAL_MTF_CONTEXT"
COIN_STATE_VERSION = "v1"

MICRO_STATE_TREND_UP = "TREND_UP"
MICRO_STATE_TREND_DOWN = "TREND_DOWN"
MICRO_STATE_RANGE = "RANGE"
MICRO_STATE_COMPRESSION = "COMPRESSION"
MICRO_STATE_REVERSAL_ATTEMPT = "REVERSAL_ATTEMPT"

SETUP_STATE_CONTINUATION = "CONTINUATION"
SETUP_STATE_RECLAIM = "RECLAIM"
SETUP_STATE_REVERSAL_RETEST = "REVERSAL_RETEST"
SETUP_STATE_RANGE_FADE = "RANGE_FADE"
SETUP_STATE_NEUTRAL = "NEUTRAL"

BACKDROP_STATE_TREND = "TREND"
BACKDROP_STATE_RANGE = "RANGE"
BACKDROP_STATE_MIXED = "MIXED"
BACKDROP_STATE_TRANSITION = "TRANSITION"

TRANSITION_NONE = "NONE"
TRANSITION_FAILED_BREAKDOWN = "FAILED_BREAKDOWN"
TRANSITION_FAILED_BREAKOUT = "FAILED_BREAKOUT"
TRANSITION_RECLAIMING = "RECLAIMING"
TRANSITION_BREAKOUT_RETEST = "BREAKOUT_RETEST"
TRANSITION_EXHAUSTION = "EXHAUSTION"

EXIT_PROFILE_TREND_EXPANSION = "TREND_EXPANSION"
EXIT_PROFILE_BALANCED = "BALANCED"
EXIT_PROFILE_RANGE_EFFICIENCY = "RANGE_EFFICIENCY"
EXIT_PROFILE_TRANSITION_DEFENSE = "TRANSITION_DEFENSE"


def default_coin_state_context() -> dict:
    return {
        "microState5m": MICRO_STATE_RANGE,
        "setupState15m": SETUP_STATE_NEUTRAL,
        "backdropState1h": BACKDROP_STATE_MIXED,
        "macroState4h": BACKDROP_STATE_MIXED,
        "transitionState": TRANSITION_NONE,
        "stateConfidence": 0.0,
        "stateFreshness": "AVAILABLE",
        "dominantSide": "NEUTRAL",
        "allowedEntryFamilies": [],
        "preferredExitProfile": EXIT_PROFILE_BALANCED,
        "coinStateSource": COIN_STATE_SOURCE,
        "coinStateVersion": COIN_STATE_VERSION,
        "coinStateRouteReason": "",
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


def _window_return(closes: Sequence[float]) -> float:
    sample = [float(close) for close in closes if _to_float(close, 0.0) > 0]
    if len(sample) < 2 or sample[0] <= 0:
        return 0.0
    return (sample[-1] - sample[0]) / sample[0]


def _derive_micro_state(closes_15m: Sequence[float], structure_ctx: dict) -> str:
    if len(closes_15m) < 6:
        return MICRO_STATE_RANGE
    recent = list(closes_15m)[-6:]
    ret = _window_return(recent)
    breakout_state = str(structure_ctx.get("breakoutRetestState", "NONE") or "NONE").upper()
    compression = str(structure_ctx.get("compressionState", "NONE") or "NONE").upper()
    if breakout_state in (RETEST_FAILED_BREAKDOWN, RETEST_FAILED_BREAKOUT):
        return MICRO_STATE_REVERSAL_ATTEMPT
    if compression in ("TIGHT", "MILD"):
        return MICRO_STATE_COMPRESSION
    if ret >= 0.006:
        return MICRO_STATE_TREND_UP
    if ret <= -0.006:
        return MICRO_STATE_TREND_DOWN
    return MICRO_STATE_RANGE


def _derive_backdrop_state(closes_1h: Sequence[float], structure_ctx: dict) -> tuple[str, str]:
    trend = str(structure_ctx.get("structureTrend", "MIXED") or "MIXED").upper()
    breakout_state = str(structure_ctx.get("breakoutRetestState", "NONE") or "NONE").upper()
    long_ret = _window_return(list(closes_1h)[-8:])
    macro_ret = _window_return(list(closes_1h)[-16:]) if len(closes_1h) >= 16 else long_ret

    if breakout_state in (RETEST_FAILED_BREAKDOWN, RETEST_FAILED_BREAKOUT):
        return BACKDROP_STATE_TRANSITION, BACKDROP_STATE_TRANSITION
    if trend in (STRUCTURE_TREND_UP, STRUCTURE_TREND_DOWN) and abs(long_ret) >= 0.01:
        return BACKDROP_STATE_TREND, BACKDROP_STATE_TREND if abs(macro_ret) >= 0.015 else BACKDROP_STATE_MIXED
    if trend == STRUCTURE_TREND_RANGE or abs(long_ret) <= 0.004:
        return BACKDROP_STATE_RANGE, BACKDROP_STATE_RANGE
    return BACKDROP_STATE_MIXED, BACKDROP_STATE_MIXED


def _derive_transition_state(structure_ctx: dict, micro_state: str, closes_15m: Sequence[float]) -> str:
    breakout_state = str(structure_ctx.get("breakoutRetestState", "NONE") or "NONE").upper()
    bias = str(structure_ctx.get("patternBias", "NEUTRAL") or "NEUTRAL").upper()
    compression = str(structure_ctx.get("compressionState", "NONE") or "NONE").upper()
    if breakout_state == RETEST_FAILED_BREAKDOWN:
        return TRANSITION_FAILED_BREAKDOWN
    if breakout_state == RETEST_FAILED_BREAKOUT:
        return TRANSITION_FAILED_BREAKOUT
    if breakout_state in (RETEST_BULL_HOLD, RETEST_BEAR_HOLD):
        return TRANSITION_BREAKOUT_RETEST
    if bias == "RECLAIM":
        return TRANSITION_RECLAIMING
    if compression == "EXPANDING" and len(closes_15m) >= 8:
        recent = list(closes_15m)[-4:]
        prior = list(closes_15m)[-8:-4]
        if abs(_window_return(recent)) < abs(_window_return(prior)) * 0.5:
            return TRANSITION_EXHAUSTION
    if micro_state == MICRO_STATE_REVERSAL_ATTEMPT:
        return TRANSITION_RECLAIMING
    return TRANSITION_NONE


def _derive_setup_state(structure_ctx: dict, micro_state: str, transition_state: str) -> str:
    bias = str(structure_ctx.get("patternBias", "NEUTRAL") or "NEUTRAL").upper()
    confidence = _to_float(structure_ctx.get("patternConfidence", 0.0), 0.0)
    compression = str(structure_ctx.get("compressionState", "NONE") or "NONE").upper()

    if transition_state in (
        TRANSITION_FAILED_BREAKDOWN,
        TRANSITION_FAILED_BREAKOUT,
    ):
        return SETUP_STATE_REVERSAL_RETEST
    if transition_state == TRANSITION_BREAKOUT_RETEST and bias == "CONTINUATION" and confidence >= 0.55:
        return SETUP_STATE_CONTINUATION
    if transition_state == TRANSITION_BREAKOUT_RETEST:
        return SETUP_STATE_REVERSAL_RETEST
    if bias == "CONTINUATION" and confidence >= 0.55:
        return SETUP_STATE_CONTINUATION
    if bias == "RECLAIM" and confidence >= 0.45:
        return SETUP_STATE_RECLAIM
    if compression in ("TIGHT", "MILD") or micro_state == MICRO_STATE_RANGE:
        return SETUP_STATE_RANGE_FADE
    return SETUP_STATE_NEUTRAL


def _derive_dominant_side(structure_ctx: dict, backdrop_state: str, micro_state: str, side: str) -> str:
    trend = str(structure_ctx.get("structureTrend", "MIXED") or "MIXED").upper()
    breakout_state = str(structure_ctx.get("breakoutRetestState", "NONE") or "NONE").upper()
    safe_side = str(side or "").upper().strip()
    if breakout_state == RETEST_FAILED_BREAKDOWN:
        return "LONG"
    if breakout_state == RETEST_FAILED_BREAKOUT:
        return "SHORT"
    if trend == STRUCTURE_TREND_UP or micro_state == MICRO_STATE_TREND_UP:
        return "LONG"
    if trend == STRUCTURE_TREND_DOWN or micro_state == MICRO_STATE_TREND_DOWN:
        return "SHORT"
    if backdrop_state == BACKDROP_STATE_TREND and safe_side in ("LONG", "SHORT"):
        return safe_side
    return "NEUTRAL"


def _derive_allowed_entry_families(setup_state: str, dominant_side: str, transition_state: str) -> list[str]:
    families: list[str] = []
    if setup_state == SETUP_STATE_CONTINUATION and dominant_side in ("LONG", "SHORT"):
        families.extend(["continuation", "reclaim"])
    elif setup_state == SETUP_STATE_RECLAIM:
        families.extend(["reclaim", "continuation"])
    elif setup_state == SETUP_STATE_REVERSAL_RETEST:
        families.extend(["reversal_retest", "reclaim"])
    elif setup_state == SETUP_STATE_RANGE_FADE:
        families.extend(["reclaim", "range_fade"])
    if transition_state in (TRANSITION_FAILED_BREAKDOWN, TRANSITION_FAILED_BREAKOUT):
        if "reversal_retest" not in families:
            families.insert(0, "reversal_retest")
    return families


def _derive_preferred_exit_profile(
    setup_state: str,
    backdrop_state: str,
    macro_state: str,
    transition_state: str,
    confidence: float,
) -> str:
    if transition_state in (
        TRANSITION_FAILED_BREAKDOWN,
        TRANSITION_FAILED_BREAKOUT,
        TRANSITION_RECLAIMING,
        TRANSITION_EXHAUSTION,
    ) and confidence >= 0.4:
        return EXIT_PROFILE_TRANSITION_DEFENSE
    if setup_state == SETUP_STATE_CONTINUATION and backdrop_state == BACKDROP_STATE_TREND and macro_state in (BACKDROP_STATE_TREND, BACKDROP_STATE_MIXED) and confidence >= 0.55:
        return EXIT_PROFILE_TREND_EXPANSION
    if setup_state == SETUP_STATE_RANGE_FADE or backdrop_state == BACKDROP_STATE_RANGE:
        return EXIT_PROFILE_RANGE_EFFICIENCY
    return EXIT_PROFILE_BALANCED


def analyze_coin_state(
    symbol: str,
    ohlcv_15m,
    ohlcv_1h,
    *,
    side: str = "",
    structure_ctx: dict | None = None,
) -> dict:
    result = default_coin_state_context()
    candles_15m = list(ohlcv_15m or [])
    candles_1h = list(ohlcv_1h or [])
    if len(candles_15m) < 8 or len(candles_1h) < 8:
        return result

    safe_structure = structure_ctx if isinstance(structure_ctx, dict) else {}
    closes_15m = [_safe_close(candle) for candle in candles_15m if _safe_close(candle) > 0]
    closes_1h = [_safe_close(candle) for candle in candles_1h if _safe_close(candle) > 0]
    if len(closes_15m) < 8 or len(closes_1h) < 8:
        return result

    micro_state = _derive_micro_state(closes_15m, safe_structure)
    backdrop_state, macro_state = _derive_backdrop_state(closes_1h, safe_structure)
    transition_state = _derive_transition_state(safe_structure, micro_state, closes_15m)
    setup_state = _derive_setup_state(safe_structure, micro_state, transition_state)

    votes = 0.0
    if setup_state != SETUP_STATE_NEUTRAL:
        votes += 1.0
    if backdrop_state != BACKDROP_STATE_MIXED:
        votes += 1.0
    if macro_state != BACKDROP_STATE_MIXED:
        votes += 1.0
    if transition_state != TRANSITION_NONE:
        votes += 1.0
    if micro_state in (MICRO_STATE_TREND_UP, MICRO_STATE_TREND_DOWN, MICRO_STATE_REVERSAL_ATTEMPT):
        votes += 0.5
    confidence = min(1.0, votes / 4.5)

    dominant_side = _derive_dominant_side(safe_structure, backdrop_state, micro_state, side)
    allowed_entry_families = _derive_allowed_entry_families(setup_state, dominant_side, transition_state)
    preferred_exit_profile = _derive_preferred_exit_profile(setup_state, backdrop_state, macro_state, transition_state, confidence)

    result.update(
        {
            "microState5m": micro_state,
            "setupState15m": setup_state,
            "backdropState1h": backdrop_state,
            "macroState4h": macro_state,
            "transitionState": transition_state,
            "stateConfidence": round(confidence, 4),
            "stateFreshness": "READY",
            "dominantSide": dominant_side,
            "allowedEntryFamilies": allowed_entry_families,
            "preferredExitProfile": preferred_exit_profile,
            "coinStateSource": COIN_STATE_SOURCE,
            "coinStateVersion": COIN_STATE_VERSION,
        }
    )
    return result
