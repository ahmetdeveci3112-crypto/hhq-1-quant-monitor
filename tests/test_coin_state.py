import main

from risk.coin_state import (
    BACKDROP_STATE_RANGE,
    BACKDROP_STATE_TREND,
    EXIT_PROFILE_RANGE_EFFICIENCY,
    EXIT_PROFILE_TRANSITION_DEFENSE,
    EXIT_PROFILE_TREND_EXPANSION,
    SETUP_STATE_CONTINUATION,
    SETUP_STATE_RANGE_FADE,
    SETUP_STATE_REVERSAL_RETEST,
    TRANSITION_BREAKOUT_RETEST,
    TRANSITION_FAILED_BREAKDOWN,
    analyze_coin_state,
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


def test_analyze_coin_state_prefers_trend_expansion_for_continuation_trend():
    ohlcv_15m = _candles_from_closes([100, 101, 100.8, 102.2, 101.7, 103.1, 102.6, 104.2, 103.8, 105.4, 105.0, 106.5])
    ohlcv_1h = _candles_from_closes([99, 101, 100.5, 102.8, 102.2, 104.5, 104.0, 106.9, 106.3, 108.8], step_ms=3600000, wick=1.5)

    state = analyze_coin_state(
        "TESTUSDT",
        ohlcv_15m,
        ohlcv_1h,
        side="LONG",
        structure_ctx={
            "structureTrend": "UP",
            "patternBias": "CONTINUATION",
            "patternConfidence": 0.78,
            "breakoutRetestState": "BULL_RETEST_HOLD",
            "compressionState": "EXPANDING",
        },
    )

    assert state["setupState15m"] == SETUP_STATE_CONTINUATION
    assert state["backdropState1h"] == BACKDROP_STATE_TREND
    assert state["preferredExitProfile"] == EXIT_PROFILE_TREND_EXPANSION
    assert "continuation" in state["allowedEntryFamilies"]


def test_analyze_coin_state_prefers_transition_defense_on_failed_breakdown():
    ohlcv_15m = _candles_from_closes([102, 101.4, 101.2, 100.9, 100.7, 100.4, 100.1, 99.8, 100.0, 100.5, 100.9, 101.2])
    ohlcv_1h = _candles_from_closes([104, 103.2, 102.5, 101.8, 101.0, 100.2, 99.8, 100.6, 101.4], step_ms=3600000, wick=1.2)

    state = analyze_coin_state(
        "TESTUSDT",
        ohlcv_15m,
        ohlcv_1h,
        side="LONG",
        structure_ctx={
            "structureTrend": "DOWN",
            "patternBias": "RECLAIM",
            "patternConfidence": 0.62,
            "breakoutRetestState": "FAILED_BREAKDOWN",
            "compressionState": "NONE",
        },
    )

    assert state["setupState15m"] == SETUP_STATE_REVERSAL_RETEST
    assert state["transitionState"] == TRANSITION_FAILED_BREAKDOWN
    assert state["preferredExitProfile"] == EXIT_PROFILE_TRANSITION_DEFENSE
    assert "reversal_retest" in state["allowedEntryFamilies"]


def test_analyze_coin_state_prefers_range_efficiency_on_flat_range():
    ohlcv_15m = _candles_from_closes([100.0, 100.15, 100.05, 100.18, 100.02, 100.16, 100.08, 100.12, 100.04, 100.14, 100.09, 100.11], wick=0.05)
    ohlcv_1h = _candles_from_closes([100.0, 100.1, 100.0, 100.08, 100.03, 100.09, 100.04, 100.07, 100.02], step_ms=3600000, wick=0.08)

    state = analyze_coin_state(
        "TESTUSDT",
        ohlcv_15m,
        ohlcv_1h,
        side="SHORT",
        structure_ctx={
            "structureTrend": "RANGE",
            "patternBias": "NEUTRAL",
            "patternConfidence": 0.21,
            "breakoutRetestState": "NONE",
            "compressionState": "MILD",
        },
    )

    assert state["setupState15m"] == SETUP_STATE_RANGE_FADE
    assert state["backdropState1h"] == BACKDROP_STATE_RANGE
    assert state["preferredExitProfile"] == EXIT_PROFILE_RANGE_EFFICIENCY
    assert "range_fade" in state["allowedEntryFamilies"]


def test_analyze_coin_state_marks_breakout_retest_transition():
    ohlcv_15m = _candles_from_closes([100, 100.8, 100.5, 101.4, 101.1, 102.1, 101.8, 102.9, 102.6, 103.4, 103.2, 103.8])
    ohlcv_1h = _candles_from_closes([99.5, 100.4, 101.0, 101.8, 102.5, 103.1, 103.7, 104.4], step_ms=3600000, wick=1.0)

    state = analyze_coin_state(
        "TESTUSDT",
        ohlcv_15m,
        ohlcv_1h,
        side="LONG",
        structure_ctx={
            "structureTrend": "UP",
            "patternBias": "CONTINUATION",
            "patternConfidence": 0.7,
            "breakoutRetestState": "BULL_RETEST_HOLD",
            "compressionState": "NONE",
        },
    )

    assert state["transitionState"] == TRANSITION_BREAKOUT_RETEST
    assert state["dominantSide"] == "LONG"


def test_classify_signal_entry_archetype_returns_reversal_retest_for_coin_state_setup():
    archetype = main.classify_signal_entry_archetype(
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "setupState15m": "REVERSAL_RETEST",
            "transitionState": "FAILED_BREAKDOWN",
            "dominantSide": "LONG",
            "stateConfidence": 0.71,
            "allowedEntryFamilies": ["reversal_retest", "reclaim"],
        },
        default_mode=main.STRATEGY_MODE_SMART_V3_RUNNER,
    )

    assert archetype == main.ENTRY_ARCHETYPE_REVERSAL_RETEST


def test_reconcile_entry_archetype_with_coin_state_routes_continuation_to_reversal_retest():
    archetype, reason = main.reconcile_entry_archetype_with_coin_state(
        main.ENTRY_ARCHETYPE_CONTINUATION,
        "LONG",
        {
            "setupState15m": "REVERSAL_RETEST",
            "transitionState": "FAILED_BREAKDOWN",
            "dominantSide": "LONG",
            "stateConfidence": 0.74,
            "allowedEntryFamilies": ["reversal_retest", "reclaim"],
        },
    )

    assert archetype == main.ENTRY_ARCHETYPE_REVERSAL_RETEST
    assert reason == "COIN_STATE_ROUTE_REVERSAL"


def test_reconcile_entry_archetype_with_coin_state_blocks_continuation_against_dominant_side():
    archetype, reason = main.reconcile_entry_archetype_with_coin_state(
        main.ENTRY_ARCHETYPE_CONTINUATION,
        "SHORT",
        {
            "setupState15m": "CONTINUATION",
            "transitionState": "NONE",
            "dominantSide": "LONG",
            "stateConfidence": 0.82,
            "allowedEntryFamilies": ["continuation", "reclaim"],
        },
    )

    assert archetype == ""
    assert reason == "COIN_STATE_CONTINUATION_BLOCKED"
