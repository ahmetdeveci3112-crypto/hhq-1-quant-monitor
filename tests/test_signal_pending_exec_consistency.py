import asyncio
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
