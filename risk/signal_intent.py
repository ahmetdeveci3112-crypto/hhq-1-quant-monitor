"""Signal intent resolver for SMART_V3_RUNNER.

Provides a side/archetype proposal layer that sits above the legacy z-score
trigger path. Non-runner modes remain untouched.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from risk.strategy_profile import STRATEGY_MODE_SMART_V3_RUNNER, normalize_strategy_mode

logger = logging.getLogger(__name__)

SIGNAL_INTENT_POLICY_OFF = "off"
SIGNAL_INTENT_POLICY_SHADOW = "shadow"
SIGNAL_INTENT_POLICY_APPLY = "apply"
VALID_SIGNAL_INTENT_POLICIES = (
    SIGNAL_INTENT_POLICY_OFF,
    SIGNAL_INTENT_POLICY_SHADOW,
    SIGNAL_INTENT_POLICY_APPLY,
)
SIGNAL_INTENT_VERSION = "signal_intent_v1"

ENTRY_ARCHETYPE_CONTINUATION = "continuation"
ENTRY_ARCHETYPE_RECLAIM = "reclaim"
ENTRY_ARCHETYPE_EXHAUSTION = "exhaustion"
ENTRY_ARCHETYPE_RECOVERY = "recovery"

RUNNER_CONTEXT_TREND = "trend_aligned"
RUNNER_CONTEXT_COUNTER = "countertrend"
RUNNER_CONTEXT_RECOVERY = "recovery"
RUNNER_CONTEXT_INTRADAY = "intraday_continuation"

UNDERWATER_ADVERSE_STRONG = "ADVERSE_STRONG"
UNDERWATER_RECOVERING = "RECOVERING"
CONTINUATION_CHOP = "CHOP"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_policy_mode(raw: Optional[str]) -> str:
    value = str(raw or SIGNAL_INTENT_POLICY_SHADOW).strip().lower()
    if value in VALID_SIGNAL_INTENT_POLICIES:
        return value
    logger.warning(
        "SIGNAL_INTENT_POLICY_FALLBACK: unknown '%s' -> %s",
        raw,
        SIGNAL_INTENT_POLICY_SHADOW,
    )
    return SIGNAL_INTENT_POLICY_SHADOW


SIGNAL_INTENT_V3_MODE = _normalize_policy_mode(
    os.environ.get("SIGNAL_INTENT_V3_MODE", SIGNAL_INTENT_POLICY_SHADOW)
)


def _breakout_side(breakout: str) -> str:
    value = str(breakout or "").upper()
    if value in ("BREAKOUT_LONG", "BULLISH_BREAKOUT", "LONG_BREAKOUT"):
        return "LONG"
    if value in ("BREAKOUT_SHORT", "BEARISH_BREAKOUT", "SHORT_BREAKOUT"):
        return "SHORT"
    return ""


def _daily_trend_side(daily_trend: str) -> str:
    value = str(daily_trend or "").upper()
    if value in ("BULLISH", "STRONG_BULLISH"):
        return "LONG"
    if value in ("BEARISH", "STRONG_BEARISH"):
        return "SHORT"
    return ""


def _ob_side(ob_trend: float, threshold: float = 2.0) -> str:
    safe_ob = _coerce_float(ob_trend, 0.0)
    limit = abs(_coerce_float(threshold, 2.0))
    if safe_ob >= limit:
        return "LONG"
    if safe_ob <= -limit:
        return "SHORT"
    return ""


def _zscore_side(zscore: float) -> str:
    safe_z = _coerce_float(zscore, 0.0)
    if safe_z < 0:
        return "LONG"
    if safe_z > 0:
        return "SHORT"
    return ""


def _liq_side(liq_state: str) -> str:
    value = str(liq_state or "").upper()
    if value.startswith("SELL_"):
        return "LONG"
    if value.startswith("BUY_"):
        return "SHORT"
    return ""


def _select_side(*candidates: str) -> str:
    for candidate in candidates:
        safe_candidate = str(candidate or "").upper()
        if safe_candidate in ("LONG", "SHORT"):
            return safe_candidate
    return ""


def _regime_bucket(regime: str) -> str:
    value = str(regime or "").upper()
    if value in ("TRENDING", "TRENDING_UP", "TRENDING_DOWN"):
        return "TREND"
    if value in ("RANGING", "QUIET"):
        return "RANGE"
    if value == "VOLATILE":
        return "VOLATILE"
    return "UNKNOWN"


def _candidate(
    *,
    side: str,
    entry_archetype: str,
    direction_owner: str,
    direction_reason: str,
    score: float,
    confidence: float,
    runner_context: str,
) -> Dict[str, Any]:
    return {
        "accepted": True,
        "side": side,
        "entryArchetype": entry_archetype,
        "directionOwner": direction_owner,
        "directionConfidence": round(_clamp(confidence, 0.0, 0.99), 4),
        "directionReason": direction_reason,
        "intentScore": round(_clamp(score, 0.0, 100.0), 4),
        "alternateIntent": None,
        "runnerContextHint": runner_context,
        "signalIntentVersion": SIGNAL_INTENT_VERSION,
    }


def _continuation_candidate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    breakout_side = _breakout_side(data.get("breakout", ""))
    daily_side = _daily_trend_side(data.get("coinDailyTrend", ""))
    ob_side = _ob_side(data.get("obImbalanceTrend", 0.0))
    volume_ratio = _coerce_float(data.get("volumeRatio", 1.0), 1.0)
    adx = _coerce_float(data.get("adx", 0.0), 0.0)
    hurst = _coerce_float(data.get("hurst", 0.5), 0.5)
    is_volume_spike = bool(data.get("isVolumeSpike", False))
    regime_bucket = _regime_bucket(data.get("marketRegime", "RANGING"))
    trend_like = regime_bucket == "TREND" or adx >= 25.0 or hurst >= 0.54
    intraday_like = (
        breakout_side
        and (volume_ratio >= 1.20 or is_volume_spike)
        and ob_side == breakout_side
        and adx >= 24.0
        and hurst >= 0.50
    )
    accepted = False
    side = ""
    if breakout_side and (volume_ratio >= 1.20 or is_volume_spike) and ob_side == breakout_side:
        accepted = True
        side = breakout_side
    elif trend_like and daily_side and ob_side == daily_side and (volume_ratio >= 1.30 or is_volume_spike):
        accepted = True
        side = daily_side
    if not accepted or not side:
        return None
    score = 52.0
    score += min(14.0, max(0.0, volume_ratio - 1.0) * 12.0)
    score += min(10.0, max(0.0, adx - 20.0) * 0.5)
    score += 6.0 if intraday_like else 2.0 if trend_like else 0.0
    confidence = 0.58 + min(0.12, max(0.0, volume_ratio - 1.0) * 0.18)
    confidence += 0.08 if ob_side == side else 0.0
    confidence += 0.08 if daily_side == side else 0.0
    return _candidate(
        side=side,
        entry_archetype=ENTRY_ARCHETYPE_CONTINUATION,
        direction_owner="continuation",
        direction_reason="BREAKOUT_VOLUME_OBI" if breakout_side else "TREND_FLOW_ALIGN",
        score=score,
        confidence=confidence,
        runner_context=RUNNER_CONTEXT_INTRADAY if intraday_like else RUNNER_CONTEXT_TREND,
    )


def _reclaim_candidate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    zscore = _coerce_float(data.get("zscore", 0.0), 0.0)
    side = _zscore_side(zscore)
    fib_active = bool(data.get("fibActive", False))
    has_sr = bool(data.get("srNearestSupport") or data.get("srNearestResistance"))
    regime_bucket = _regime_bucket(data.get("marketRegime", "RANGING"))
    adx = _coerce_float(data.get("adx", 0.0), 0.0)
    hurst = _coerce_float(data.get("hurst", 0.5), 0.5)
    if not side:
        return None
    if abs(zscore) < 1.05 and not fib_active and not has_sr:
        return None
    score = 46.0 + min(18.0, abs(zscore) * 6.0)
    score += 5.0 if fib_active else 0.0
    score += 4.0 if has_sr else 0.0
    if regime_bucket == "RANGE" or (adx <= 23.0 and hurst <= 0.50):
        score += 5.0
    confidence = 0.50 + min(0.16, abs(zscore) * 0.06)
    confidence += 0.05 if fib_active else 0.0
    confidence += 0.04 if has_sr else 0.0
    return _candidate(
        side=side,
        entry_archetype=ENTRY_ARCHETYPE_RECLAIM,
        direction_owner="reclaim",
        direction_reason="STRETCH_FIB_SR",
        score=score,
        confidence=confidence,
        runner_context=RUNNER_CONTEXT_COUNTER,
    )


def _exhaustion_candidate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    liq_score = _coerce_float(data.get("liqEchoScore", 0.0), 0.0)
    liq_state = str(data.get("liqEchoState", "NONE") or "NONE").upper()
    side = _liq_side(liq_state)
    if not side or liq_score < 6.0:
        return None
    zscore = _coerce_float(data.get("zscore", 0.0), 0.0)
    regime_bucket = _regime_bucket(data.get("marketRegime", "RANGING"))
    score = 50.0 + min(16.0, liq_score * 1.4)
    score += min(10.0, abs(zscore) * 4.0)
    score += 4.0 if regime_bucket in ("RANGE", "VOLATILE") else 0.0
    confidence = 0.56 + min(0.12, liq_score / 30.0)
    confidence += min(0.10, abs(zscore) * 0.04)
    return _candidate(
        side=side,
        entry_archetype=ENTRY_ARCHETYPE_EXHAUSTION,
        direction_owner="exhaustion",
        direction_reason="LIQ_ECHO_REVERSAL",
        score=score,
        confidence=confidence,
        runner_context=RUNNER_CONTEXT_COUNTER,
    )


def _recovery_candidate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    underwater = str(data.get("underwaterTapeState", "NEUTRAL") or "NEUTRAL").upper()
    continuation_flow = str(data.get("continuationFlowState", "NEUTRAL") or "NEUTRAL").upper()
    recovery_armed = bool(data.get("recoveryArmed", False))
    if underwater not in (UNDERWATER_ADVERSE_STRONG, UNDERWATER_RECOVERING) and continuation_flow != CONTINUATION_CHOP and not recovery_armed:
        return None
    side = _select_side(
        _breakout_side(data.get("breakout", "")),
        _ob_side(data.get("obImbalanceTrend", 0.0)),
        _daily_trend_side(data.get("coinDailyTrend", "")),
        _zscore_side(data.get("zscore", 0.0)),
    )
    if not side:
        return None
    score = 58.0
    score += 8.0 if underwater == UNDERWATER_ADVERSE_STRONG else 4.0 if underwater == UNDERWATER_RECOVERING else 0.0
    score += 6.0 if continuation_flow == CONTINUATION_CHOP else 0.0
    score += 4.0 if recovery_armed else 0.0
    confidence = 0.62
    confidence += 0.08 if underwater in (UNDERWATER_ADVERSE_STRONG, UNDERWATER_RECOVERING) else 0.0
    confidence += 0.06 if continuation_flow == CONTINUATION_CHOP else 0.0
    return _candidate(
        side=side,
        entry_archetype=ENTRY_ARCHETYPE_RECOVERY,
        direction_owner="recovery",
        direction_reason="UNDERWATER_RECOVERY" if underwater != "NEUTRAL" else "FAILED_CONTINUATION_RECOVERY",
        score=score,
        confidence=confidence,
        runner_context=RUNNER_CONTEXT_RECOVERY,
    )


def _candidate_priority(candidate: Dict[str, Any], data: Dict[str, Any]) -> float:
    archetype = str(candidate.get("entryArchetype", "") or "").lower()
    score = _coerce_float(candidate.get("intentScore", 0.0), 0.0)
    breakout_side = _breakout_side(data.get("breakout", ""))
    regime_bucket = _regime_bucket(data.get("marketRegime", "RANGING"))
    adx = _coerce_float(data.get("adx", 0.0), 0.0)
    hurst = _coerce_float(data.get("hurst", 0.5), 0.5)
    volume_ratio = _coerce_float(data.get("volumeRatio", 1.0), 1.0)
    is_volume_spike = bool(data.get("isVolumeSpike", False))
    candidate_side = str(candidate.get("side", "") or "").upper()
    daily_side = _daily_trend_side(data.get("coinDailyTrend", ""))
    ob_side = _ob_side(data.get("obImbalanceTrend", 0.0))
    intraday_like = (
        breakout_side == candidate_side
        and (volume_ratio >= 1.20 or is_volume_spike)
        and ob_side == candidate_side
        and adx >= 24.0
        and hurst >= 0.50
    )
    trend_like = regime_bucket == "TREND" or adx >= 25.0 or hurst >= 0.54
    range_like = regime_bucket in ("RANGE", "VOLATILE") or (adx <= 23.0 and hurst <= 0.50)
    priority = score
    if archetype == ENTRY_ARCHETYPE_RECOVERY:
        priority += 500.0
    elif archetype == ENTRY_ARCHETYPE_CONTINUATION:
        priority += 400.0 if intraday_like else 340.0 if trend_like else 220.0
    elif archetype == ENTRY_ARCHETYPE_RECLAIM:
        priority += 360.0 if range_like else 240.0
    elif archetype == ENTRY_ARCHETYPE_EXHAUSTION:
        priority += 350.0 if range_like else 230.0
    if candidate_side and candidate_side == _zscore_side(data.get("zscore", 0.0)):
        priority += 2.0
    if candidate_side and candidate_side == daily_side:
        priority += 1.0
    return priority


def _disabled_response(policy_mode: str, reason: str) -> Dict[str, Any]:
    return {
        "accepted": False,
        "side": "",
        "entryArchetype": "",
        "directionOwner": "",
        "directionConfidence": 0.0,
        "directionReason": reason,
        "intentScore": 0.0,
        "alternateIntent": None,
        "runnerContextHint": "",
        "signalIntentVersion": SIGNAL_INTENT_VERSION,
        "policyMode": policy_mode,
    }


def resolve_signal_intent(
    inputs: Dict[str, Any],
    *,
    mode: str,
    policy_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve the best V3 signal intent for a compact signal payload."""
    safe_mode = normalize_strategy_mode(mode)
    safe_policy_mode = _normalize_policy_mode(policy_mode or SIGNAL_INTENT_V3_MODE)
    if safe_mode != STRATEGY_MODE_SMART_V3_RUNNER:
        return _disabled_response(safe_policy_mode, "MODE_NOT_RUNNER")
    if safe_policy_mode == SIGNAL_INTENT_POLICY_OFF:
        return _disabled_response(safe_policy_mode, "SIGNAL_INTENT_DISABLED")

    data = dict(inputs or {})
    data.setdefault("marketRegime", "RANGING")
    data.setdefault("coinDailyTrend", "NEUTRAL")

    candidates = []
    for builder in (
        _recovery_candidate,
        _continuation_candidate,
        _reclaim_candidate,
        _exhaustion_candidate,
    ):
        candidate = builder(data)
        if candidate and candidate.get("accepted", False):
            candidates.append(candidate)

    if not candidates:
        return _disabled_response(safe_policy_mode, "NO_INTENT")

    ranked = sorted(
        candidates,
        key=lambda candidate: _candidate_priority(candidate, data),
        reverse=True,
    )
    selected = dict(ranked[0])
    if len(ranked) > 1:
        selected["alternateIntent"] = {
            "side": ranked[1].get("side", ""),
            "entryArchetype": ranked[1].get("entryArchetype", ""),
            "directionOwner": ranked[1].get("directionOwner", ""),
            "directionConfidence": ranked[1].get("directionConfidence", 0.0),
            "directionReason": ranked[1].get("directionReason", ""),
            "intentScore": ranked[1].get("intentScore", 0.0),
            "runnerContextHint": ranked[1].get("runnerContextHint", ""),
        }
    selected["policyMode"] = safe_policy_mode
    return selected
