import asyncio
import time

import main


def _build_reclaim_pos(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "POS_RECLAIM_AAVE",
        "signalId": "SIG_RECLAIM_AAVE",
        "symbol": "AAVEUSDT",
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "side": "LONG",
        "entryPrice": 100.0,
        "signalScore": 86.0,
        "atr": 1.0,
        "openTime": now_ms - (18 * 60 * 1000),
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "signal_snapshot": {
            "decisionContext": {"directionOwner": "reclaim"},
            "expectancy": {"expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD, "rankingScore": 118.0},
        },
        "expectancy": {"expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD, "rankingScore": 118.0},
    }
    pos.update(overrides)
    return pos


def _build_closed_trade(**overrides):
    trade = {
        "id": "TRD_RECLAIM_AAVE",
        "tradeId": "TRD_RECLAIM_AAVE",
        "signalId": "SIG_RECLAIM_AAVE",
        "positionId": "POS_RECLAIM_AAVE",
        "symbol": "AAVEUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "exitPrice": 100.25,
        "reason": "SMART_V3_RUNNER__SIDEWAYS_RECLAIM_CLOSE",
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "signalSnapshot": {
            "decisionContext": {"directionOwner": "reclaim"},
            "expectancy": {"expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD, "rankingScore": 118.0},
        },
    }
    trade.update(overrides)
    return trade


def _build_watch(**overrides):
    base_ts = 1000.0
    watch = {
        "watcherId": "PEW_AAVE",
        "symbol": "AAVEUSDT",
        "signalId": "SIG_RECLAIM_AAVE",
        "positionId": "POS_RECLAIM_AAVE",
        "originalTradeId": "TRD_RECLAIM_AAVE",
        "side": "LONG",
        "exitTs": base_ts,
        "exitPrice": 100.0,
        "entryPrice": 99.2,
        "atr": 1.0,
        "exitReason": "SIDEWAYS_RECLAIM_CLOSE",
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "peakRoiPct": 1.4,
        "watchState": main.POST_EXIT_REENTRY_STATE_ARMED,
        "cooldownUntilTs": base_ts + main.V3_POST_EXIT_REENTRY_COOLDOWN_SEC,
        "watchExpiresTs": base_ts + main.V3_POST_EXIT_REENTRY_WATCH_WINDOW_SEC,
        "candidateSinceTs": 0.0,
        "confirmCount": 0,
        "reentryUsed": False,
        "reentryTriggered": False,
        "reentryTriggerReason": "",
        "reentryPendingEntryId": "",
        "reentryPositionId": "",
        "cancelReason": "",
        "decisionContext": {"directionOwner": "reclaim"},
        "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        "expectancyRankingScore": 118.0,
        "originalSignalScore": 86.0,
        "continuationFlowState": main.V3_CONTINUATION_NEUTRAL,
        "currentVolumeRatio": 0.0,
        "currentIsVolumeSpike": False,
        "currentObImbalanceTrend": 0.0,
        "currentImbalance": 0.0,
        "currentExecScore": 0.0,
        "lastPrice": 100.0,
        "breakoutDistancePrice": 0.0,
    }
    watch.update(overrides)
    return watch


def _build_candidate_opp(**overrides):
    opp = {
        "symbol": "AAVEUSDT",
        "price": 100.6,
        "signalAction": "LONG",
        "signalScore": 88.0,
        "continuationFlowState": main.V3_CONTINUATION_SUPPORTING,
        "volumeRatio": 1.25,
        "isVolumeSpike": False,
        "obImbalanceTrend": 2.5,
        "imbalance": 2.2,
        "entryExecScore": 72.0,
        "spreadPct": 0.05,
        "atr": 1.0,
    }
    opp.update(overrides)
    return opp


def _build_candidate_signal(**overrides):
    signal = {
        "symbol": "AAVEUSDT",
        "action": "LONG",
        "confidenceScore": 88.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "decisionContext": {"entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION},
        "currentExecScore": 72.0,
    }
    signal.update(overrides)
    return signal


def test_register_post_exit_reentry_watch_arms_for_eligible_reclaim_close(monkeypatch):
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {})
    monkeypatch.setattr(main.global_paper_trader, "positions", [])
    monkeypatch.setattr(main.global_paper_trader, "pending_orders", [])

    watch = main.register_post_exit_reentry_watch(
        _build_reclaim_pos(),
        _build_closed_trade(),
        original_reason="SMART_V3_RUNNER__SIDEWAYS_RECLAIM_CLOSE",
        normalized_reason="SMART_V3_RUNNER__SIDEWAYS_RECLAIM_CLOSE",
        exit_price=100.25,
    )

    assert watch is not None
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_ARMED
    assert main.post_exit_reentry_watchers["AAVEUSDT"]["originalTradeId"] == "TRD_RECLAIM_AAVE"


def test_register_post_exit_reentry_watch_skips_hard_stop_close(monkeypatch):
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {})
    monkeypatch.setattr(main.global_paper_trader, "positions", [])
    monkeypatch.setattr(main.global_paper_trader, "pending_orders", [])

    watch = main.register_post_exit_reentry_watch(
        _build_reclaim_pos(),
        _build_closed_trade(reason="SL_HIT"),
        original_reason="SL_HIT",
        normalized_reason="SL_HIT",
        exit_price=99.0,
    )

    assert watch is None
    assert main.post_exit_reentry_watchers == {}


def test_resolve_reentry_cooldown_gate_allows_post_exit_override():
    now_ts = 1000.0
    gate = main.resolve_reentry_cooldown_gate(
        {"isPostExitReentry": True},
        {"last_close_ts": now_ts - 120.0},
        now_ts,
    )

    assert gate["blocked"] is False
    assert gate["override_applied"] is True
    assert gate["reason"] == "POST_EXIT_REENTRY_OVERRIDE"


def test_evaluate_post_exit_reentry_watchers_confirms_after_persistence(monkeypatch):
    events = []
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol: False)

    async def _fake_dispatch(safe_watch, signal, opportunity):
        safe_watch["watchState"] = main.POST_EXIT_REENTRY_STATE_CONFIRMED
        safe_watch["reentryUsed"] = True
        safe_watch["reentryTriggered"] = True
        safe_watch["reentryTriggerReason"] = "CONTINUATION_REENTRY_DISPATCHED"
        events.append((safe_watch["watcherId"], signal["symbol"], opportunity["symbol"]))
        return True

    monkeypatch.setattr(main, "dispatch_post_exit_reentry_watch", _fake_dispatch)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([_build_candidate_opp()], [_build_candidate_signal()]))
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_CANDIDATE
    assert watch["confirmCount"] == 1

    monkeypatch.setattr(main.time, "time", lambda: 1061.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([_build_candidate_opp()], [_build_candidate_signal()]))

    assert events == [("PEW_AAVE", "AAVEUSDT", "AAVEUSDT")]
    assert watch["reentryTriggered"] is True


def test_evaluate_post_exit_reentry_watchers_cancels_on_hostile_imbalance(monkeypatch):
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol: False)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    asyncio.run(
        main.evaluate_post_exit_reentry_watchers(
            [_build_candidate_opp(imbalance=-9.5)],
            [_build_candidate_signal()],
        )
    )

    assert "AAVEUSDT" not in main.post_exit_reentry_watchers


def test_build_post_exit_reentry_watchers_summary_reports_counts(monkeypatch):
    monkeypatch.setattr(
        main,
        "post_exit_reentry_watchers",
        {
            "AAVEUSDT": _build_watch(),
            "SOLUSDT": _build_watch(
                watcherId="PEW_SOL",
                symbol="SOLUSDT",
                watchState=main.POST_EXIT_REENTRY_STATE_CANDIDATE,
                confirmCount=1,
            ),
            "LINKUSDT": _build_watch(
                watcherId="PEW_LINK",
                symbol="LINKUSDT",
                watchState=main.POST_EXIT_REENTRY_STATE_CONFIRMED,
                reentryTriggered=True,
            ),
        },
    )

    summary = main.build_post_exit_reentry_watchers_summary(now_ts=1005.0)

    assert summary["active"] == 2
    assert summary["candidate"] == 1
    assert summary["confirmed"] == 1
    assert summary["triggered"] == 1


def test_extract_post_exit_reentry_watch_summary_reads_trigger_and_reason():
    snapshots = [
        {
            "stage": "post_exit_reentry_watch",
            "createdTs": 1000,
            "decision": {"watchState": main.POST_EXIT_REENTRY_STATE_CANDIDATE, "reason": "candidate"},
            "outcome": {"watchState": main.POST_EXIT_REENTRY_STATE_CANDIDATE, "confirmCount": 1, "reentryTriggered": False},
        },
        {
            "stage": "post_exit_reentry_watch",
            "createdTs": 1100,
            "decision": {"watchState": main.POST_EXIT_REENTRY_STATE_CONFIRMED, "reentryTriggerReason": "CONTINUATION_REENTRY_DISPATCHED"},
            "outcome": {"watchState": main.POST_EXIT_REENTRY_STATE_CONFIRMED, "reentryTriggered": True, "reentryTriggerReason": "CONTINUATION_REENTRY_DISPATCHED"},
        },
    ]

    summary = main._extract_post_exit_reentry_watch_summary(snapshots)

    assert summary["postExitWatchState"] == main.POST_EXIT_REENTRY_STATE_CONFIRMED
    assert summary["postExitReentryTriggered"] is True
    assert summary["postExitReentryReason"] == "CONTINUATION_REENTRY_DISPATCHED"
    assert summary["postExitReentryOutcome"] == "TRIGGERED"
