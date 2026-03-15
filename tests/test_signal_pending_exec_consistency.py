import asyncio
import json
import time

import pytest

import main


def _candles_from_levels(levels):
    candles = []
    for idx, (high, low, close) in enumerate(levels):
        candles.append([idx * 900000, close, high, low, close, 1000.0])
    return candles


def _build_post_exit_signal():
    return {
        "signalId": "SIG_REENTRY_NEW",
        "symbol": "AAVEUSDT",
        "action": "LONG",
        "confidenceScore": 88.0,
        "_rawConfidenceScore": 91.0,
        "entryPrice": 99.775,
        "pullbackPct": 0.225,
        "spreadPct": 0.05,
        "volumeRatio": 1.25,
        "atr": 0.30,
        "leverage": 8,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "decisionContext": {"entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION, "directionOwner": "continuation"},
        "expectancy": {"rankingScore": 118.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "pendingPatienceBias": 1.0,
        "structureTrend": "UP",
        "swingState": "HH_HL",
        "compressionState": "TIGHT",
        "breakoutRetestState": "BULL_RETEST_HOLD",
        "srContext": "ABOVE_SUPPORT",
        "patternBias": "CONTINUATION",
        "patternConfidence": 0.78,
        "microState5m": "TREND_UP",
        "setupState15m": "CONTINUATION",
        "backdropState1h": "TREND",
        "macroState4h": "TREND",
        "transitionState": "BREAKOUT_RETEST",
        "stateConfidence": 0.74,
        "stateFreshness": "AVAILABLE",
        "dominantSide": "LONG",
        "allowedEntryFamilies": ["continuation", "reclaim"],
        "preferredExitProfile": main.V3_EXIT_PROFILE_TREND_EXPANSION,
        "coinStateSource": "INTERNAL_MTF_CONTEXT",
        "coinStateVersion": "v1",
        "coinStateRouteReason": "COIN_STATE_ROUTE_CONTINUATION",
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "adverseDistancePct": 0.65,
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
        "isPostExitReentry": True,
        "postExitReentryEntryMode": "SHALLOW_PULLBACK",
        "postExitReentryPullbackPctOriginal": 0.98,
        "postExitReentryPullbackPctApplied": 0.225,
        "postExitReentryConfirmDelaySec": 30,
        "postExitReentryExpiresSec": 240,
        "postExitReentryExecutionPriority": "WATCHER",
        "postExitReentrySizeCapMult": 0.35,
        "postExitReentryStopMult": 0.80,
    }


def _configure_trader_for_open(monkeypatch, trader):
    monkeypatch.setattr(trader, "enabled", True)
    monkeypatch.setattr(trader, "symbol", "AAVEUSDT")
    monkeypatch.setattr(trader, "positions", [])
    monkeypatch.setattr(trader, "balance", 100.0)
    monkeypatch.setattr(trader, "max_positions", 10)
    monkeypatch.setattr(trader, "allow_hedging", True)
    monkeypatch.setattr(trader, "is_coin_blacklisted", lambda symbol: False)
    monkeypatch.setattr(trader, "get_microstructure_cooldown", lambda symbol: None)
    monkeypatch.setattr(trader, "add_log", lambda message: None)
    monkeypatch.setattr(main.live_binance_trader, "enabled", True)
    monkeypatch.setattr(main, "portfolio_risk_service", None)
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)

    async def _noop_save_entry_forecast_event(**kwargs):
        return None

    monkeypatch.setattr(main.sqlite_manager, "save_entry_forecast_event", _noop_save_entry_forecast_event)


class _DummyRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def _patch_pending_event_sinks(monkeypatch, trader, events, finalized=None, logs=None):
    monkeypatch.setattr(main, "record_signal_event_memory", lambda payload: events.append(payload))
    monkeypatch.setattr(main, "queue_decision_snapshot_from_event_payload", lambda *args, **kwargs: None)

    async def _noop_update_signal_decision(*args, **kwargs):
        return None

    async def _noop_save_signal_event(*args, **kwargs):
        return None

    monkeypatch.setattr(main.sqlite_manager, "update_signal_decision", _noop_update_signal_decision)
    monkeypatch.setattr(main.sqlite_manager, "save_signal_event", _noop_save_signal_event)
    monkeypatch.setattr(
        trader,
        "_finalize_forecast_event",
        lambda *args, **kwargs: finalized.append((args, kwargs)) if finalized is not None else None,
    )
    monkeypatch.setattr(
        trader,
        "add_log",
        lambda message: logs.append(message) if logs is not None else None,
    )


def test_open_position_keeps_opposite_pending_when_replacement_signal_rejected(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    existing_pending = {
        "id": "PO_OLD",
        "signalId": "SIG_OLD",
        "symbol": "AAVEUSDT",
        "side": "SHORT",
        "signalScore": 77.0,
        "sizeUsd": 10.0,
        "leverage": 8,
    }
    monkeypatch.setattr(trader, "pending_orders", [existing_pending])
    monkeypatch.setattr(trader, "is_coin_blacklisted", lambda symbol: True)

    result = asyncio.run(
        trader.open_position("LONG", 100.0, 0.30, _build_post_exit_signal(), symbol="AAVEUSDT")
    )

    assert result is None
    assert trader.pending_orders == [existing_pending]


def test_open_position_supersedes_opposite_pending_only_after_success(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    monkeypatch.setattr(trader, "max_positions", 1)
    existing_pending = {
        "id": "PO_OLD",
        "signalId": "SIG_OLD",
        "symbol": "AAVEUSDT",
        "side": "SHORT",
        "signalScore": 77.0,
        "sizeUsd": 10.0,
        "leverage": 8,
    }
    monkeypatch.setattr(trader, "pending_orders", [existing_pending])
    recycled = []
    finalized = []
    monkeypatch.setattr(
        trader,
        "record_pending_recycle_event",
        lambda order, code, detail="", trace=None: recycled.append((order["id"], code, detail, trace)),
    )
    monkeypatch.setattr(
        trader,
        "_finalize_forecast_event",
        lambda order_id, status, outcome_label, reason, ts_ms, fill_price=None, force_market=0: finalized.append(
            (order_id, status, reason)
        ),
    )

    result = asyncio.run(
        trader.open_position("LONG", 100.0, 0.30, _build_post_exit_signal(), symbol="AAVEUSDT")
    )

    assert result is not None
    assert result["resultKind"] == main.PENDING_OPEN_RESULT_CREATED_PENDING
    assert len(trader.pending_orders) == 1
    assert trader.pending_orders[0]["side"] == "LONG"
    assert trader.pending_orders[0]["id"] != "PO_OLD"
    assert recycled and recycled[0][1] == "PENDING__SUPERSEDED"
    assert finalized and finalized[0] == ("PO_OLD", "CANCELLED", "superseded")


def test_open_position_carries_coin_state_context_into_pending(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    monkeypatch.setattr(trader, "pending_orders", [])

    result = asyncio.run(
        trader.open_position("LONG", 100.0, 0.30, _build_post_exit_signal(), symbol="AAVEUSDT")
    )

    assert result is not None
    assert result["microState5m"] == "TREND_UP"
    assert result["setupState15m"] == "CONTINUATION"
    assert result["backdropState1h"] == "TREND"
    assert result["macroState4h"] == "TREND"
    assert result["transitionState"] == "BREAKOUT_RETEST"
    assert result["dominantSide"] == "LONG"
    assert result["allowedEntryFamilies"] == ["continuation", "reclaim"]
    assert result["preferredExitProfile"] == main.V3_EXIT_PROFILE_TREND_EXPANSION
    assert result["coinStateRouteReason"] == "COIN_STATE_ROUTE_CONTINUATION"


def test_reinforce_pending_refreshes_context_and_preserves_shallow_reentry(monkeypatch):
    trader = main.global_paper_trader
    now_ms = int(time.time() * 1000)
    pending = {
        "id": "PO_REENTRY",
        "signalId": "SIG_ORIGIN",
        "symbol": "AAVEUSDT",
        "side": "LONG",
        "signalScore": 86.0,
        "signalScoreRaw": 88.0,
        "entryPrice": 99.775,
        "signalPrice": 100.0,
        "pullbackPct": 0.225,
        "pullbackLocked": 0.00225,
        "createdAt": now_ms - 30000,
        "confirmAfter": now_ms + 30000,
        "expiresAt": now_ms + 240000,
        "atr": 0.30,
        "spreadPct": 0.05,
        "volumeRatio": 1.10,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "decisionContext": {"directionOwner": "continuation"},
        "expectancy": {"rankingScore": 110.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_NEUTRAL,
        "pendingPatienceBias": 1.0,
        "patternBias": "NEUTRAL",
        "patternConfidence": 0.25,
        "barrierState": "CLEAR",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "OLD",
        "isPostExitReentry": True,
        "postExitReentryEntryMode": "SHALLOW_PULLBACK",
        "postExitReentryPullbackPctOriginal": 0.98,
        "postExitReentryPullbackPctApplied": 0.225,
        "postExitReentryConfirmDelaySec": 30,
        "postExitReentryExpiresSec": 240,
        "postExitReentryExecutionPriority": "WATCHER",
        "postExitReentrySizeCapMult": 0.35,
        "postExitReentryStopMult": 0.80,
        "signal_snapshot": {
            "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
            "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
            "barrierReason": "OLD",
        },
    }
    signal = {
        "signalId": "SIG_NEW",
        "confidenceScore": 91.0,
        "_rawConfidenceScore": 93.0,
        "entryPrice": 98.0,
        "pullbackPct": 0.98,
        "spreadPct": 0.06,
        "volumeRatio": 1.35,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "decisionContext": {"directionOwner": "continuation", "note": "fresh"},
        "expectancy": {"rankingScore": 125.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "pendingPatienceBias": 0.9,
        "structureTrend": "UP",
        "swingState": "HH_HL",
        "compressionState": "TIGHT",
        "breakoutRetestState": "BULL_RETEST_HOLD",
        "srContext": "ABOVE_SUPPORT",
        "patternBias": "CONTINUATION",
        "patternConfidence": 0.82,
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "adverseDistancePct": 0.72,
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
    }

    reinforced = trader._reinforce_pending_order(pending, "LONG", 100.0, 0.30, signal, "AAVEUSDT")

    assert reinforced["decisionContext"]["note"] == "fresh"
    assert reinforced["expectancyBand"] == main.DECISION_EXPECTANCY_BAND_GOOD
    assert reinforced["patternBias"] == "CONTINUATION"
    assert reinforced["barrierState"] == "LONG_ABOVE_SUPPORT"
    assert reinforced["signal_snapshot"]["patternBias"] == "CONTINUATION"
    assert reinforced["signal_snapshot"]["barrierReason"] == "SUPPORTIVE_BARRIER_CONTEXT"
    assert reinforced["signal_snapshot"]["isPostExitReentry"] is True
    assert reinforced["entryPrice"] == pytest.approx(99.775)
    assert reinforced["pullbackPct"] == pytest.approx(0.225)
    assert reinforced["postExitReentryEntryMode"] == "SHALLOW_PULLBACK"


def test_reinforce_pending_tracks_latest_suggested_entry_and_drift_for_countertrend_reentry(monkeypatch):
    trader = main.global_paper_trader
    now_ms = int(time.time() * 1000)
    pending = {
        "id": "PO_CT_REENTRY",
        "signalId": "SIG_CT_ORIGIN",
        "symbol": "1000LUNCUSDT",
        "side": "LONG",
        "signalScore": 84.0,
        "signalScoreRaw": 86.0,
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.0,
        "pendingEntryDriftPct": 0.0,
        "signalPrice": 100.4,
        "pullbackPct": 0.25,
        "pullbackLocked": 0.0025,
        "createdAt": now_ms - 30000,
        "confirmAfter": now_ms + 30000,
        "expiresAt": now_ms + 180000,
        "atr": 0.30,
        "currentAtrPct": 0.30,
        "spreadPct": 0.05,
        "volumeRatio": 1.05,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "decisionContext": {"directionOwner": "reclaim"},
        "expectancy": {"rankingScore": 108.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_NEUTRAL,
        "pendingPatienceBias": 1.0,
        "barrierState": "LONG_NEAR_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "RECLAIM_HOLD",
        "marketFallbackDisallowed": False,
        "marketFallbackBlockReason": "",
        "countertrendFallbackProtected": True,
        "isPostExitReentry": True,
        "postExitReentryEntryMode": "SHALLOW_PULLBACK",
        "postExitReentryPullbackPctOriginal": 0.90,
        "postExitReentryPullbackPctApplied": 0.25,
        "postExitReentryConfirmDelaySec": 30,
        "postExitReentryExpiresSec": 240,
        "postExitReentryExecutionPriority": "WATCHER",
        "postExitReentrySizeCapMult": 0.40,
        "postExitReentryStopMult": 0.80,
        "signal_snapshot": {
            "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
            "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
            "isPostExitReentry": True,
        },
    }
    signal = {
        "signalId": "SIG_CT_REFRESH",
        "confidenceScore": 90.0,
        "_rawConfidenceScore": 92.0,
        "entryPrice": 101.0,
        "latestSuggestedEntryPrice": 101.0,
        "pullbackPct": 0.80,
        "spreadPct": 0.05,
        "volumeRatio": 1.20,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "decisionContext": {"directionOwner": "reclaim", "note": "fresh"},
        "expectancy": {"rankingScore": 118.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "pendingPatienceBias": 0.95,
        "barrierState": "LONG_NEAR_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "RECLAIM_HOLD",
        "countertrendFallbackProtected": True,
        "marketFallbackDisallowed": False,
    }

    reinforced = trader._reinforce_pending_order(pending, "LONG", 100.6, 0.30, signal, "1000LUNCUSDT")

    assert reinforced["entryPrice"] == pytest.approx(100.0)
    assert reinforced["latestSuggestedEntryPrice"] == pytest.approx(101.0)
    assert reinforced["pendingEntryDriftPct"] == pytest.approx(1.0)
    assert reinforced["countertrendFallbackProtected"] is True
    assert reinforced["decisionContext"]["note"] == "fresh"


def test_reinforce_pending_tracks_latest_suggested_entry_and_drift_for_continuation(monkeypatch):
    trader = main.global_paper_trader
    now_ms = int(time.time() * 1000)
    pending = {
        "id": "PO_CONT_STALE",
        "signalId": "SIG_CONT_ORIGIN",
        "symbol": "ACTUSDT",
        "side": "LONG",
        "signalScore": 86.0,
        "signalScoreRaw": 88.0,
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.0,
        "pendingEntryDriftPct": 0.0,
        "createdAt": now_ms - 20000,
        "confirmAfter": now_ms + 10000,
        "expiresAt": now_ms + 120000,
        "atr": 0.20,
        "currentAtrPct": 0.20,
        "spreadPct": 0.05,
        "volumeRatio": 1.10,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
        "signal_snapshot": {
            "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
            "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        },
    }
    signal = {
        "signalId": "SIG_CONT_REFRESH",
        "confidenceScore": 91.0,
        "_rawConfidenceScore": 93.0,
        "entryPrice": 100.35,
        "latestSuggestedEntryPrice": 100.35,
        "spreadPct": 0.05,
        "volumeRatio": 1.24,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
    }

    reinforced = trader._reinforce_pending_order(pending, "LONG", 100.10, 0.20, signal, "ACTUSDT")

    assert reinforced["latestSuggestedEntryPrice"] == pytest.approx(100.35)
    assert reinforced["pendingEntryDriftPct"] == pytest.approx(
        round(
            main.compute_pending_entry_drift_pct(
                "LONG",
                reinforced["entryPrice"],
                reinforced["latestSuggestedEntryPrice"],
            ),
            4,
        )
    )


def test_reinforce_pending_stops_extending_expiry_when_countertrend_entry_turns_stale(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "COUNTERTREND_MARKET_FALLBACK_GUARD_ENABLED", True)
    now_ms = int(time.time() * 1000)
    original_expiry = now_ms + 60000
    pending = {
        "id": "PO_CT_STALE",
        "signalId": "SIG_CT_STALE",
        "symbol": "1000LUNCUSDT",
        "side": "LONG",
        "signalScore": 83.0,
        "signalScoreRaw": 85.0,
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.0,
        "pendingEntryDriftPct": 0.0,
        "createdAt": now_ms - 20000,
        "confirmAfter": now_ms + 10000,
        "expiresAt": original_expiry,
        "atr": 0.20,
        "currentAtrPct": 0.10,
        "spreadPct": 0.05,
        "volumeRatio": 1.00,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "barrierState": "LONG_NEAR_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "RECLAIM_HOLD",
        "marketFallbackDisallowed": False,
        "marketFallbackBlockReason": "",
        "countertrendFallbackProtected": True,
        "signal_snapshot": {
            "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
            "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        },
    }
    signal = {
        "signalId": "SIG_CT_STALE_REFRESH",
        "confidenceScore": 89.0,
        "_rawConfidenceScore": 91.0,
        "entryPrice": 100.40,
        "latestSuggestedEntryPrice": 100.40,
        "pullbackPct": 0.60,
        "spreadPct": 0.05,
        "volumeRatio": 1.10,
        "memoryExtensionSec": 600,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "barrierState": "LONG_NEAR_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "RECLAIM_HOLD",
        "countertrendFallbackProtected": True,
        "marketFallbackDisallowed": False,
    }
    stale_before = trader.pipeline_metrics.get("countertrend_pending_stale", 0)

    reinforced = trader._reinforce_pending_order(pending, "LONG", 100.2, 0.20, signal, "1000LUNCUSDT")

    assert reinforced["expiresAt"] == original_expiry
    assert reinforced["pendingEntryDriftPct"] > main.resolve_countertrend_pending_stale_drift_threshold_pct(reinforced)
    assert trader.pipeline_metrics["countertrend_pending_stale"] == stale_before + 1


def test_revalidate_pending_entry_uses_cached_15m_structure_for_barrier_parity(monkeypatch):
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
        "recent_candles": [],
    }
    cached_15m = _candles_from_levels([
        (100.60, 100.20, 100.30),
        (100.40, 99.92, 100.05),
        (100.55, 100.10, 100.20),
        (100.30, 99.88, 100.00),
        (100.45, 100.08, 100.12),
        (100.20, 99.90, 100.02),
    ])
    monkeypatch.setattr(main, "_get_cached_ohlcv_for_structure", lambda symbol: (cached_15m, []))

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result["decision"] == "FAIL_DROP"
    assert any(reason.startswith("BARRIER_REJECT") for reason in result["reasons"])


def test_open_position_uses_reversal_retest_pending_profile(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    monkeypatch.setattr(trader, "pending_orders", [])
    signal = {
        "signalId": "SIG_REVERSAL",
        "symbol": "AEVOUSDT",
        "action": "LONG",
        "confidenceScore": 84.0,
        "_rawConfidenceScore": 87.0,
        "entryPrice": 99.0,
        "pullbackPct": 0.95,
        "spreadPct": 0.05,
        "volumeRatio": 1.18,
        "atr": 0.30,
        "leverage": 8,
        "entryArchetype": main.ENTRY_ARCHETYPE_REVERSAL_RETEST,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "decisionContext": {"entryArchetype": main.ENTRY_ARCHETYPE_REVERSAL_RETEST, "directionOwner": "reversal_retest"},
        "expectancy": {"rankingScore": 108.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "pendingPatienceBias": 0.9,
        "breakoutRetestState": "FAILED_BREAKDOWN",
        "transitionState": "FAILED_BREAKDOWN",
        "supportiveLevelPrice": 99.82,
        "supportiveDistancePct": 0.18,
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
    }

    result = asyncio.run(
        trader.open_position("LONG", 100.0, 0.30, signal, symbol="AEVOUSDT")
    )

    assert result is not None
    assert result["resultKind"] == main.PENDING_OPEN_RESULT_CREATED_PENDING
    assert result["entryArchetype"] == main.ENTRY_ARCHETYPE_REVERSAL_RETEST
    assert result["reversalRetestEntryMode"] == "RETEST_POCKET"
    assert result["reversalRetestPullbackPctApplied"] < 0.30
    assert result["pullbackPct"] == pytest.approx(result["reversalRetestPullbackPctApplied"], rel=1e-6)
    assert result["reversalRetestConfirmDelaySec"] == main.V3_REVERSAL_RETEST_CONFIRM_DELAY_SEC
    assert result["reversalRetestExpiresSec"] == main.V3_REVERSAL_RETEST_EXPIRES_SEC
    assert result["reversalRetestZoneState"] == "READY"
    assert result["reversalRetestPocketPrice"] > 0
    assert result["reversalRetestZoneConfidence"] > 0.5


def test_open_position_uses_post_exit_exposure_reserve_on_max_exposure(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    monkeypatch.setattr(trader, "positions", [{"symbol": "SOLUSDT", "side": "LONG", "sizeUsd": 15.0, "leverage": 5}])
    monkeypatch.setattr(trader, "pending_orders", [])
    monkeypatch.setattr(trader, "max_positions", 1)

    signal = _build_post_exit_signal()
    signal.update({
        "symbol": "SAHARAUSDT",
        "action": "SHORT",
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_CONTINUATION_RESCUE,
        "postExitFollowthroughActive": True,
        "postExitFollowthroughMode": main.POST_EXIT_RESOLUTION_MODE_SAME_SIDE,
        "postExitPreferredSide": "SHORT",
        "postExitPreferredEntryFamilies": [main.ENTRY_ARCHETYPE_CONTINUATION, main.ENTRY_ARCHETYPE_RECLAIM],
        "postExitFollowthroughConfidence": 0.78,
        "postExitWatchRegisterResult": "REGISTERED",
        "postExitWatchRegisterReason": "REGISTER_OK",
        "entryExecScore": 74.0,
        "sizeMultiplier": 1.0,
    })

    result = asyncio.run(
        trader.open_position("SHORT", 0.0245, 0.0003, signal, symbol="SAHARAUSDT")
    )

    assert result is not None
    assert result["resultKind"] == main.PENDING_OPEN_RESULT_CREATED_PENDING
    assert result["postExitExposureReserveUsed"] is True
    assert result["postExitExposureReserveReason"] == "POST_EXIT_EXPOSURE_RESERVE"
    assert result["sizeMultiplier"] <= main.V3_POST_EXIT_EXPOSURE_RESERVE_SIZE_MULT


def test_gate_and_execute_bootstraps_position_runtime_state_from_pending(monkeypatch):
    trader = main.global_paper_trader
    _configure_trader_for_open(monkeypatch, trader)
    monkeypatch.setattr(trader, "pending_orders", [])
    monkeypatch.setattr(trader, "positions", [])
    signal = {
        "signalId": "SIG_REV_BOOT",
        "symbol": "AEVOUSDT",
        "action": "LONG",
        "confidenceScore": 86.0,
        "_rawConfidenceScore": 89.0,
        "entryPrice": 99.0,
        "pullbackPct": 0.85,
        "spreadPct": 0.05,
        "volumeRatio": 1.22,
        "atr": 0.30,
        "leverage": 8,
        "entryArchetype": main.ENTRY_ARCHETYPE_REVERSAL_RETEST,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "decisionContext": {"entryArchetype": main.ENTRY_ARCHETYPE_REVERSAL_RETEST, "directionOwner": "reversal_retest"},
        "expectancy": {"rankingScore": 112.0},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "pendingPatienceBias": 0.9,
        "structureTrend": "UP",
        "swingState": "HH_HL",
        "compressionState": "MILD",
        "breakoutRetestState": "FAILED_BREAKDOWN",
        "srContext": "ABOVE_SUPPORT",
        "patternBias": "RECLAIM",
        "patternConfidence": 0.72,
        "microState5m": "REVERSAL_ATTEMPT",
        "setupState15m": "REVERSAL_RETEST",
        "backdropState1h": "TREND",
        "macroState4h": "TRANSITION",
        "transitionState": "FAILED_BREAKDOWN",
        "stateConfidence": 0.72,
        "stateFreshness": "AVAILABLE",
        "dominantSide": "LONG",
        "allowedEntryFamilies": ["reversal_retest", "reclaim"],
        "preferredExitProfile": main.V3_EXIT_PROFILE_TRANSITION_DEFENSE,
        "coinStateSource": "INTERNAL_MTF_CONTEXT",
        "coinStateVersion": "v1",
        "coinStateRouteReason": "COIN_STATE_ROUTE_REVERSAL",
        "supportiveLevelPrice": 99.82,
        "supportiveDistancePct": 0.18,
        "barrierState": "LONG_ABOVE_SUPPORT",
        "barrierVerdict": "SUPPORTIVE",
        "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
    }
    pending = asyncio.run(trader.open_position("LONG", 100.0, 0.30, signal, symbol="AEVOUSDT"))
    assert pending is not None
    order = trader.pending_orders[0]
    order["confirmed"] = True
    order["confirmAfter"] = int(time.time() * 1000) - 1000
    monkeypatch.setattr(main.live_binance_trader, "enabled", False)
    monkeypatch.setattr(
        main,
        "revalidate_pending_entry",
        lambda order, opportunity, now_ms, force_market=False: {
            "decision": "PASS",
            "allow_market": False,
            "hard_reject": False,
            "recheck_score": float(order.get("signalScore", 0.0)),
            "reasons": ["PASS"],
            "reason_summary": "PASS",
        },
    )
    monkeypatch.setattr(main, "queue_decision_snapshot_from_event_payload", lambda *args, **kwargs: None)

    async def _noop_update_signal_decision(*args, **kwargs):
        return None

    async def _noop_save_signal_event(*args, **kwargs):
        return None

    monkeypatch.setattr(main.sqlite_manager, "update_signal_decision", _noop_update_signal_decision)
    monkeypatch.setattr(main.sqlite_manager, "save_signal_event", _noop_save_signal_event)

    now_ms = int(time.time() * 1000)
    result = asyncio.run(trader._gate_and_execute(order, order["entryPrice"], [{"symbol": "AEVOUSDT"}], now_ms))

    assert result is True
    assert trader.positions
    pos = trader.positions[0]
    assert pos["positionThesisState"] == main.POSITION_THESIS_ENTRY
    assert pos["microState5m"] == "REVERSAL_ATTEMPT"
    assert pos["setupState15m"] == "REVERSAL_RETEST"
    assert pos["transitionState"] == "FAILED_BREAKDOWN"
    assert pos["preferredExitProfile"] == main.V3_EXIT_PROFILE_TRANSITION_DEFENSE
    assert pos["coinStateRouteReason"] == "COIN_STATE_ROUTE_REVERSAL"
    assert pos["reversalRetestZoneState"] == "READY"
    assert pos["reversalRetestPocketPrice"] > 0
    assert pos["runtimeExitProfile"] == main.V3_EXIT_PROFILE_TRANSITION_DEFENSE
    assert pos["runtimeExitOwner"] == main.V3_EXIT_OWNER_TRANSITION_PROTECT


def test_manual_market_order_bootstraps_coin_state_and_exit_owner(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(trader, "positions", [])
    monkeypatch.setattr(trader, "balance", 100.0)
    monkeypatch.setattr(trader, "max_positions", 10)
    monkeypatch.setattr(trader, "risk_per_trade", 0.02)
    monkeypatch.setattr(trader, "leverage", 8)
    monkeypatch.setattr(trader, "strategy_mode", main.STRATEGY_MODE_SMART_V3_RUNNER)
    monkeypatch.setattr(trader, "exit_tightness", 1.0)
    monkeypatch.setattr(trader, "sl_atr", 2.0)
    monkeypatch.setattr(trader, "tp_atr", 3.0)
    monkeypatch.setattr(trader, "trail_activation_atr", 1.0)
    monkeypatch.setattr(trader, "trail_distance_atr", 1.0)
    monkeypatch.setattr(trader, "is_coin_blacklisted", lambda symbol: False)
    monkeypatch.setattr(trader, "add_log", lambda message: None)
    monkeypatch.setattr(trader, "save_state", lambda: None)
    monkeypatch.setattr(main.multi_coin_scanner, "analyzers", {})
    monkeypatch.setattr(main.live_binance_trader, "enabled", True)

    async def _set_leverage(symbol, leverage):
        return True

    async def _place_market_order(symbol, side, size_usd, leverage):
        return {"average": 100.0, "filled": 1.0}

    protective_stop = {}

    async def _set_stop_loss(symbol, side, amount, stop_price):
        protective_stop.update({
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "stop_price": stop_price,
        })
        return {"id": "sl_manual_1", "stopPrice": stop_price}

    async def _noop_save_open_position(position):
        return None

    monkeypatch.setattr(main.live_binance_trader, "set_leverage", _set_leverage)
    monkeypatch.setattr(main.live_binance_trader, "place_market_order", _place_market_order)
    monkeypatch.setattr(main.live_binance_trader, "set_stop_loss", _set_stop_loss)
    monkeypatch.setattr(main.sqlite_manager, "save_open_position", _noop_save_open_position)
    monkeypatch.setattr(
        main,
        "apply_market_structure_context",
        lambda payload, **kwargs: {
            **payload,
            "microState5m": "TREND_UP",
            "setupState15m": "CONTINUATION",
            "backdropState1h": "TREND",
            "macroState4h": "TREND",
            "transitionState": "BREAKOUT_RETEST",
            "stateConfidence": 0.74,
            "stateFreshness": "AVAILABLE",
            "dominantSide": "LONG",
            "allowedEntryFamilies": ["continuation", "reclaim"],
            "preferredExitProfile": main.V3_EXIT_PROFILE_TREND_EXPANSION,
            "coinStateSource": "INTERNAL_MTF_CONTEXT",
            "coinStateVersion": "v1",
            "structureTrend": "UP",
            "swingState": "HH_HL",
            "compressionState": "TIGHT",
            "breakoutRetestState": "BULL_RETEST_HOLD",
            "srContext": "ABOVE_SUPPORT",
            "patternBias": "CONTINUATION",
            "patternConfidence": 0.82,
        },
    )
    monkeypatch.setattr(main, "classify_v3_runner_context", lambda payload, default_mode=None: main.V3_RUNNER_CONTEXT_TREND)
    monkeypatch.setattr(
        main,
        "analyze_side_aware_entry_barrier",
        lambda **kwargs: {
            "barrierState": "LONG_ABOVE_SUPPORT",
            "barrierVerdict": "SUPPORTIVE",
            "adverseLevelType": "LOCAL_RESISTANCE",
            "adverseLevelPrice": 101.2,
            "adverseDistancePct": 1.2,
            "supportiveLevelType": "LOCAL_SUPPORT",
            "supportiveLevelPrice": 99.7,
            "supportiveDistancePct": 0.3,
            "barrierReason": "SUPPORTIVE_BARRIER_CONTEXT",
            "barrierConfidence": 0.75,
            "levelsDetected": True,
        },
    )
    monkeypatch.setattr(
        main,
        "resolve_v3_exit_behavior_profile",
        lambda pos, **kwargs: {
            "profile": main.V3_EXIT_PROFILE_TREND_EXPANSION,
            "reason": "COIN_STATE_TREND_EXPANSION",
            "exit_owner": main.V3_EXIT_OWNER_TREND_CONTINUATION,
            "exit_owner_reason": "TREND_CONTINUATION_CONTROL",
            "exit_owner_tighten_bias": -0.12,
            "exit_owner_allow_hold": True,
        },
    )

    response = asyncio.run(
        main.paper_trading_market_order(
            _DummyRequest({"symbol": "AAVEUSDT", "side": "LONG", "price": 100.0, "signalLeverage": 8})
        )
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    pos = payload["position"]
    assert pos["entryArchetype"] == main.ENTRY_ARCHETYPE_CONTINUATION
    assert pos["positionThesisState"] == main.POSITION_THESIS_ENTRY
    assert pos["coinStateRouteReason"] == "COIN_STATE_ROUTE_CONTINUATION"
    assert pos["runtimeExitProfile"] == main.V3_EXIT_PROFILE_TREND_EXPANSION
    assert pos["runtimeExitOwner"] == main.V3_EXIT_OWNER_TREND_CONTINUATION
    assert pos["barrierState"] == "LONG_ABOVE_SUPPORT"
    assert pos["runtimeExchangeProtectiveMode"] == main.V3_EXCHANGE_PROTECTIVE_MODE_EMERGENCY
    assert pos["runtimeExchangeProtectionAuthority"] == "EMERGENCY_FLOOR"
    assert pos["runtimeExchangeProtectionRole"] == "LOSS_FAILSAFE"
    assert protective_stop["stop_price"] == pos["runtimeExchangeProtectiveStopPrice"]


def test_gate_and_execute_emits_recheck_wait_event(monkeypatch):
    trader = main.global_paper_trader
    now_ms = int(time.time() * 1000)
    order = {
        "id": "PO_WAIT",
        "signalId": "SIG_WAIT",
        "symbol": "ACTUSDT",
        "side": "SHORT",
        "entryPrice": 100.0,
        "signalScore": 86.0,
        "signalScoreRaw": 88.0,
        "sizeMultiplier": 1.0,
        "leverage": 8,
    }
    monkeypatch.setattr(trader, "pending_orders", [order])
    monkeypatch.setattr(main.live_binance_trader, "enabled", False)
    monkeypatch.setattr(
        main,
        "revalidate_pending_entry",
        lambda order, opportunity, now_ms, force_market=False: {
            "decision": "WARN_WAIT",
            "allow_market": False,
            "hard_reject": False,
            "recheck_score": 51.0,
            "reasons": ["BARRIER_WAIT"],
            "reason_summary": "BARRIER_WAIT",
        },
    )
    events = []
    monkeypatch.setattr(main, "record_signal_event_memory", lambda payload: events.append(payload))
    monkeypatch.setattr(main, "queue_decision_snapshot_from_event_payload", lambda *args, **kwargs: None)

    async def _noop_update_signal_decision(*args, **kwargs):
        return None

    async def _noop_save_signal_event(*args, **kwargs):
        return None

    monkeypatch.setattr(main.sqlite_manager, "update_signal_decision", _noop_update_signal_decision)
    monkeypatch.setattr(main.sqlite_manager, "save_signal_event", _noop_save_signal_event)

    result = asyncio.run(trader._gate_and_execute(order, 100.0, [{"symbol": "ACTUSDT"}], now_ms))

    assert result is False
    assert any(event.get("decision_code") == "PENDING__RECHECK_WAIT" for event in events)


def test_gate_and_execute_emits_recheck_fail_event(monkeypatch):
    trader = main.global_paper_trader
    now_ms = int(time.time() * 1000)
    order = {
        "id": "PO_FAIL",
        "signalId": "SIG_FAIL",
        "symbol": "ACTUSDT",
        "side": "SHORT",
        "entryPrice": 100.0,
        "signalScore": 86.0,
        "signalScoreRaw": 88.0,
        "sizeMultiplier": 1.0,
        "leverage": 8,
    }
    monkeypatch.setattr(trader, "pending_orders", [order])
    monkeypatch.setattr(main.live_binance_trader, "enabled", False)
    monkeypatch.setattr(
        main,
        "revalidate_pending_entry",
        lambda order, opportunity, now_ms, force_market=False: {
            "decision": "FAIL_DROP",
            "allow_market": False,
            "hard_reject": True,
            "recheck_score": 22.0,
            "reasons": ["BARRIER_REJECT"],
            "reason_summary": "BARRIER_REJECT",
        },
    )
    events = []
    monkeypatch.setattr(main, "record_signal_event_memory", lambda payload: events.append(payload))
    monkeypatch.setattr(main, "queue_decision_snapshot_from_event_payload", lambda *args, **kwargs: None)

    async def _noop_update_signal_decision(*args, **kwargs):
        return None

    async def _noop_save_signal_event(*args, **kwargs):
        return None

    monkeypatch.setattr(main.sqlite_manager, "update_signal_decision", _noop_update_signal_decision)
    monkeypatch.setattr(main.sqlite_manager, "save_signal_event", _noop_save_signal_event)
    monkeypatch.setattr(trader, "_finalize_forecast_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(trader, "add_log", lambda message: None)

    result = asyncio.run(trader._gate_and_execute(order, 100.0, [{"symbol": "ACTUSDT"}], now_ms))

    assert result is True
    assert any(event.get("decision_code") == "PENDING__RECHECK_FAIL" for event in events)


def test_check_pending_orders_blocks_countertrend_market_fallback_chase_on_expire(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "COUNTERTREND_MARKET_FALLBACK_GUARD_ENABLED", True)
    order = {
        "id": "PO_LUNC_BLOCK",
        "signalId": "SIG_LUNC_BLOCK",
        "symbol": "1000LUNCUSDT",
        "side": "LONG",
        "entryPrice": 0.041048,
        "latestSuggestedEntryPrice": 0.0413,
        "pendingEntryDriftPct": 0.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "countertrendFallbackProtected": True,
        "marketFallbackDisallowed": False,
        "marketFallbackBlockReason": "",
        "barrierVerdict": "SUPPORTIVE",
        "signalScore": 92.0,
        "signalScoreRaw": 94.0,
        "minConfidenceScoreSnapshot": 74.0,
        "sizeMultiplier": 1.0,
        "leverage": 6,
        "confirmed": True,
        "expiresAt": 0,
        "atr": 0.00045,
        "currentAtrPct": 1.0,
    }
    events = []
    finalized = []
    logs = []
    _patch_pending_event_sinks(monkeypatch, trader, events, finalized=finalized, logs=logs)
    monkeypatch.setattr(main, "apply_live_flow_context_to_pending_order", lambda order, opp: None)
    gate_calls = []

    async def _unexpected_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(trader, "_gate_and_execute", _unexpected_gate)
    monkeypatch.setattr(trader, "pending_orders", [order])
    blocked_before = trader.pipeline_metrics.get("countertrend_fallback_blocked", 0)
    expired_before = trader.pipeline_metrics.get("pending_expired", 0)

    asyncio.run(trader.check_pending_orders([{"symbol": "1000LUNCUSDT", "price": 0.04161}]))

    assert gate_calls == []
    assert trader.pending_orders == []
    assert trader.pipeline_metrics["countertrend_fallback_blocked"] == blocked_before + 1
    assert trader.pipeline_metrics["pending_expired"] == expired_before + 1
    assert order["marketFallbackBlockReason"] == "COUNTERTREND_FALLBACK_CHASE_BLOCK"
    assert any(event.get("decision_code") == "PENDING__COUNTERTREND_FALLBACK_BLOCK" for event in events)
    assert any(event.get("decision_detail") == "COUNTERTREND_FALLBACK_CHASE_BLOCK" for event in events)
    assert finalized


def test_check_pending_orders_allows_countertrend_market_fallback_when_price_is_still_close(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "COUNTERTREND_MARKET_FALLBACK_GUARD_ENABLED", True)
    order = {
        "id": "PO_LUNC_ALLOW",
        "signalId": "SIG_LUNC_ALLOW",
        "symbol": "1000LUNCUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.05,
        "pendingEntryDriftPct": 0.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "countertrendFallbackProtected": True,
        "marketFallbackDisallowed": False,
        "marketFallbackBlockReason": "",
        "barrierVerdict": "SUPPORTIVE",
        "signalScore": 91.0,
        "signalScoreRaw": 93.0,
        "minConfidenceScoreSnapshot": 74.0,
        "sizeMultiplier": 1.0,
        "leverage": 6,
        "confirmed": True,
        "expiresAt": 0,
        "atr": 0.6,
        "currentAtrPct": 0.6,
    }
    events = []
    _patch_pending_event_sinks(monkeypatch, trader, events)
    monkeypatch.setattr(main, "apply_live_flow_context_to_pending_order", lambda order, opp: None)
    gate_calls = []

    async def _gate_stub(order_arg, current_price, opportunities, current_time, force_market=False):
        gate_calls.append(
            {
                "order_id": order_arg["id"],
                "current_price": current_price,
                "force_market": force_market,
            }
        )
        if order_arg in trader.pending_orders:
            trader.pending_orders.remove(order_arg)
        return True

    monkeypatch.setattr(trader, "_gate_and_execute", _gate_stub)
    monkeypatch.setattr(trader, "pending_orders", [order])

    asyncio.run(trader.check_pending_orders([{"symbol": "1000LUNCUSDT", "price": 100.10}]))

    assert gate_calls and gate_calls[0]["force_market"] is True
    assert gate_calls[0]["order_id"] == "PO_LUNC_ALLOW"
    assert trader.pending_orders == []


def test_check_pending_orders_blocks_continuation_market_fallback_when_entry_is_stale(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "CONTINUATION_MARKET_FALLBACK_GUARD_ENABLED", True)
    order = {
        "id": "PO_CONT_BLOCK",
        "signalId": "SIG_CONT_BLOCK",
        "symbol": "ACTUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.40,
        "pendingEntryDriftPct": 0.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "barrierVerdict": "SUPPORTIVE",
        "signalScore": 90.0,
        "signalScoreRaw": 91.0,
        "minConfidenceScoreSnapshot": 74.0,
        "sizeMultiplier": 1.0,
        "leverage": 8,
        "confirmed": True,
        "expiresAt": 0,
        "atr": 0.4,
        "currentAtrPct": 0.4,
    }
    events = []
    _patch_pending_event_sinks(monkeypatch, trader, events)
    monkeypatch.setattr(main, "apply_live_flow_context_to_pending_order", lambda order, opp: None)
    gate_calls = []

    async def _gate_stub(order_arg, current_price, opportunities, current_time, force_market=False):
        gate_calls.append((order_arg["id"], current_price, force_market))
        if order_arg in trader.pending_orders:
            trader.pending_orders.remove(order_arg)
        return True

    monkeypatch.setattr(trader, "_gate_and_execute", _gate_stub)
    monkeypatch.setattr(trader, "pending_orders", [order])

    asyncio.run(trader.check_pending_orders([{"symbol": "ACTUSDT", "price": 100.45}]))

    assert gate_calls == []
    assert trader.pending_orders == []
    assert any(event.get("decision_code") == "PENDING__CONTINUATION_FALLBACK_BLOCK" for event in events)


def test_check_pending_orders_allows_supportive_continuation_market_fallback(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "CONTINUATION_MARKET_FALLBACK_GUARD_ENABLED", True)
    order = {
        "id": "PO_CONT_ALLOW",
        "signalId": "SIG_CONT_ALLOW",
        "symbol": "ACTUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.05,
        "pendingEntryDriftPct": 0.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "barrierVerdict": "SUPPORTIVE",
        "signalScore": 90.0,
        "signalScoreRaw": 91.0,
        "minConfidenceScoreSnapshot": 74.0,
        "sizeMultiplier": 1.0,
        "leverage": 8,
        "confirmed": True,
        "expiresAt": 0,
        "atr": 0.4,
        "currentAtrPct": 0.4,
    }
    events = []
    _patch_pending_event_sinks(monkeypatch, trader, events)
    monkeypatch.setattr(main, "apply_live_flow_context_to_pending_order", lambda order, opp: None)
    gate_calls = []

    async def _gate_stub(order_arg, current_price, opportunities, current_time, force_market=False):
        gate_calls.append((order_arg["id"], current_price, force_market))
        if order_arg in trader.pending_orders:
            trader.pending_orders.remove(order_arg)
        return True

    monkeypatch.setattr(trader, "_gate_and_execute", _gate_stub)
    monkeypatch.setattr(trader, "pending_orders", [order])

    asyncio.run(trader.check_pending_orders([{"symbol": "ACTUSDT", "price": 100.08}]))

    assert gate_calls == [("PO_CONT_ALLOW", 100.08, True)]
    assert trader.pending_orders == []
    assert not any(event.get("decision_code") == "PENDING__CONTINUATION_FALLBACK_BLOCK" for event in events)


def test_check_pending_orders_blocks_continuation_fallback_on_market_relation_deterioration(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(main, "MARKET_RELATION_PENDING_GUARD_ENABLED", True)
    order = {
        "id": "PO_CONT_REL_BLOCK",
        "signalId": "SIG_CONT_REL_BLOCK",
        "symbol": "ACTUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "latestSuggestedEntryPrice": 100.02,
        "pendingEntryDriftPct": 0.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_TREND,
        "barrierVerdict": "SUPPORTIVE",
        "marketRelationState": "SUPPORTIVE",
        "altBtcState": "STRONG",
        "triangleState": "CLEAR",
        "signalScore": 90.0,
        "signalScoreRaw": 91.0,
        "minConfidenceScoreSnapshot": 74.0,
        "sizeMultiplier": 1.0,
        "leverage": 8,
        "confirmed": True,
        "expiresAt": 0,
        "atr": 0.4,
        "currentAtrPct": 0.4,
    }
    events = []
    _patch_pending_event_sinks(monkeypatch, trader, events)
    gate_calls = []

    async def _gate_stub(order_arg, current_price, opportunities, current_time, force_market=False):
        gate_calls.append((order_arg["id"], current_price, force_market))
        if order_arg in trader.pending_orders:
            trader.pending_orders.remove(order_arg)
        return True

    monkeypatch.setattr(trader, "_gate_and_execute", _gate_stub)
    monkeypatch.setattr(trader, "pending_orders", [order])

    asyncio.run(
        trader.check_pending_orders(
            [
                {
                    "symbol": "ACTUSDT",
                    "price": 100.06,
                    "marketRelationState": "CAUTION",
                    "altBtcState": "WEAK",
                    "triangleState": "CLEAR",
                }
            ]
        )
    )

    assert gate_calls == []
    assert trader.pending_orders == []
    assert any(event.get("decision_code") == "PENDING__MARKET_RELATION_BLOCK" for event in events)
