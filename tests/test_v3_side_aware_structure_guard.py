import time

import main


def _candles_from_levels(levels):
    candles = []
    for idx, (high, low, close) in enumerate(levels):
        candles.append([idx * 300000, close, high, low, close, 1000.0])
    return candles


def test_side_aware_barrier_blocks_continuation_short_into_local_support():
    recent_candles = _candles_from_levels([
        (100.60, 100.20, 100.30),
        (100.40, 99.92, 100.05),
        (100.55, 100.10, 100.20),
        (100.30, 99.88, 100.00),
        (100.45, 100.08, 100.12),
        (100.20, 99.90, 100.02),
    ])

    result = main.analyze_side_aware_entry_barrier(
        symbol="ACTUSDT",
        side="SHORT",
        current_price=100.0,
        atr=0.4,
        entry_archetype=main.ENTRY_ARCHETYPE_CONTINUATION,
        runner_context_resolved=main.V3_RUNNER_CONTEXT_TREND,
        recent_candles=recent_candles,
        structure_ctx={},
        sr_levels_4h={},
    )

    assert result["barrierVerdict"] == "BLOCK"
    assert result["barrierState"] == "SHORT_INTO_SUPPORT"
    assert result["adverseDistancePct"] > 0


def test_side_aware_barrier_blocks_continuation_long_into_local_resistance():
    recent_candles = _candles_from_levels([
        (100.05, 99.60, 99.90),
        (100.12, 99.72, 99.95),
        (100.18, 99.74, 100.00),
        (100.14, 99.78, 99.98),
        (100.16, 99.82, 100.01),
        (100.10, 99.80, 99.99),
    ])

    result = main.analyze_side_aware_entry_barrier(
        symbol="TESTUSDT",
        side="LONG",
        current_price=100.0,
        atr=0.4,
        entry_archetype=main.ENTRY_ARCHETYPE_CONTINUATION,
        runner_context_resolved=main.V3_RUNNER_CONTEXT_TREND,
        recent_candles=recent_candles,
        structure_ctx={},
        sr_levels_4h={},
    )

    assert result["barrierVerdict"] == "BLOCK"
    assert result["barrierState"] == "LONG_INTO_RESISTANCE"


def test_side_aware_barrier_keeps_reclaim_soft_near_support():
    recent_candles = _candles_from_levels([
        (100.60, 100.20, 100.30),
        (100.40, 99.92, 100.05),
        (100.55, 100.10, 100.20),
        (100.30, 99.88, 100.00),
        (100.45, 100.08, 100.12),
        (100.20, 99.90, 100.02),
    ])

    result = main.analyze_side_aware_entry_barrier(
        symbol="TESTUSDT",
        side="SHORT",
        current_price=100.0,
        atr=0.4,
        entry_archetype=main.ENTRY_ARCHETYPE_RECLAIM,
        runner_context_resolved=main.V3_RUNNER_CONTEXT_COUNTER,
        recent_candles=recent_candles,
        structure_ctx={},
        sr_levels_4h={},
    )

    assert result["barrierVerdict"] == "CAUTION"
    assert result["barrierState"] == "SHORT_INTO_SUPPORT"


def test_side_aware_barrier_marks_structure_blind_continuation():
    result = main.analyze_side_aware_entry_barrier(
        symbol="TESTUSDT",
        side="SHORT",
        current_price=100.0,
        atr=0.5,
        entry_archetype=main.ENTRY_ARCHETYPE_CONTINUATION,
        runner_context_resolved=main.V3_RUNNER_CONTEXT_TREND,
        recent_candles=[],
        structure_ctx={},
        sr_levels_4h={},
    )

    assert result["barrierVerdict"] == "CAUTION"
    assert result["barrierState"] == "STRUCTURE_BLIND"
    assert result["barrierReason"] == "STRUCTURE_BLIND"


def test_revalidate_pending_entry_waits_when_fresh_short_runs_into_support():
    now_ms = int(time.time() * 1000)
    order = {
        "symbol": "ACTUSDT",
        "side": "SHORT",
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "spreadPct": 0.05,
        "signalScore": 86,
        "minConfidenceScoreSnapshot": 74,
        "createdAt": now_ms - (2 * 60 * 1000),
        "entryPrice": 100.0,
        "atr": 0.4,
        "pullbackPct": 0.35,
        "signal_snapshot": {},
    }
    opportunity = {
        "symbol": "ACTUSDT",
        "signalAction": "SHORT",
        "spreadPct": 0.05,
        "volumeRatio": 1.3,
        "obImbalanceTrend": -3.5,
        "price": 100.0,
        "recent_candles": _candles_from_levels([
            (100.60, 100.20, 100.30),
            (100.40, 99.92, 100.05),
            (100.55, 100.10, 100.20),
            (100.30, 99.88, 100.00),
            (100.45, 100.08, 100.12),
            (100.20, 99.90, 100.02),
        ]),
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result["decision"] == "WARN_WAIT"
    assert any(reason.startswith("BARRIER_WAIT") for reason in result["reasons"])


def test_revalidate_pending_entry_drops_stale_short_into_support():
    now_ms = int(time.time() * 1000)
    order = {
        "symbol": "ACTUSDT",
        "side": "SHORT",
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "spreadPct": 0.05,
        "signalScore": 86,
        "minConfidenceScoreSnapshot": 74,
        "createdAt": now_ms - (12 * 60 * 1000),
        "entryPrice": 100.0,
        "atr": 0.4,
        "pullbackPct": 0.35,
        "signal_snapshot": {},
    }
    opportunity = {
        "symbol": "ACTUSDT",
        "signalAction": "SHORT",
        "spreadPct": 0.05,
        "volumeRatio": 1.3,
        "obImbalanceTrend": -3.5,
        "price": 100.0,
        "recent_candles": _candles_from_levels([
            (100.60, 100.20, 100.30),
            (100.40, 99.92, 100.05),
            (100.55, 100.10, 100.20),
            (100.30, 99.88, 100.00),
            (100.45, 100.08, 100.12),
            (100.20, 99.90, 100.02),
        ]),
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result["decision"] == "FAIL_DROP"
    assert result["hard_reject"] is True
    assert any(reason.startswith("BARRIER_REJECT") for reason in result["reasons"])


def test_countertrend_barrier_guard_hard_blocks_reclaim_long_signal():
    result = main.evaluate_countertrend_barrier_guard(
        main.ENTRY_ARCHETYPE_RECLAIM,
        {
            "barrierVerdict": "BLOCK",
            "barrierState": "LONG_INTO_RESISTANCE",
            "adverseDistancePct": 0.18,
            "barrierReason": "LOCAL_RESISTANCE_NEARBY",
        },
        hard_block_enabled=True,
    )

    assert result["countertrendFallbackProtected"] is True
    assert result["marketFallbackDisallowed"] is True
    assert result["marketFallbackBlockReason"] == "COUNTERTREND_BARRIER_BLOCK"
    assert result["rejectSignal"] is True
    assert result["decisionCode"] == "ENTRY__COUNTERTREND_BARRIER_BLOCK"


def test_countertrend_barrier_guard_hard_blocks_reversal_retest_long_signal():
    result = main.evaluate_countertrend_barrier_guard(
        main.ENTRY_ARCHETYPE_REVERSAL_RETEST,
        {
            "barrierVerdict": "BLOCK",
            "barrierState": "LONG_INTO_RESISTANCE",
            "adverseDistancePct": 0.22,
            "barrierReason": "FAILED_BREAKDOWN_INTO_RESISTANCE",
        },
        hard_block_enabled=True,
    )

    assert result["countertrendFallbackProtected"] is True
    assert result["marketFallbackDisallowed"] is True
    assert result["marketFallbackBlockReason"] == "COUNTERTREND_BARRIER_BLOCK"
    assert result["rejectSignal"] is True
    assert result["decisionCode"] == "ENTRY__COUNTERTREND_BARRIER_BLOCK"


def test_countertrend_barrier_guard_locks_market_fallback_on_caution():
    result = main.evaluate_countertrend_barrier_guard(
        main.ENTRY_ARCHETYPE_RECLAIM,
        {
            "barrierVerdict": "CAUTION",
            "barrierState": "LONG_NEAR_RESISTANCE",
            "adverseDistancePct": 0.41,
            "barrierReason": "LOCAL_RESISTANCE_NEARBY",
        },
        hard_block_enabled=False,
    )

    assert result["countertrendFallbackProtected"] is True
    assert result["marketFallbackDisallowed"] is True
    assert result["marketFallbackBlockReason"] == "COUNTERTREND_BARRIER_CAUTION"
    assert result["rejectSignal"] is False
