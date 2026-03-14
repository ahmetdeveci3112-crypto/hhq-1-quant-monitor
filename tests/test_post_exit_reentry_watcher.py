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
        "resolutionMode": main.POST_EXIT_RESOLUTION_MODE_SAME_SIDE,
        "resolutionTargetSide": "LONG",
        "resolutionReason": "WATCH_ARMED",
        "resolutionConfidence": 0.0,
        "resolutionEntryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
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


def test_resolve_post_exit_reentry_entry_profile_clamps_deep_pullback_long():
    profile = main.resolve_post_exit_reentry_entry_profile(
        {
            "action": "LONG",
            "entryPrice": 99.02,
            "pullbackPct": 0.98,
            "currentAtrPct": 0.30,
        },
        {"price": 100.0},
        {"side": "LONG", "exitPrice": 100.0},
    )

    assert profile["postExitReentryEntryMode"] == "SHALLOW_PULLBACK"
    assert round(profile["postExitReentryPullbackPctApplied"], 4) == 0.225
    assert round(profile["entryPrice"], 4) == 99.775


def test_resolve_post_exit_reentry_entry_profile_clamps_deep_pullback_short():
    profile = main.resolve_post_exit_reentry_entry_profile(
        {
            "action": "SHORT",
            "entryPrice": 100.98,
            "pullbackPct": 0.98,
            "currentAtrPct": 0.30,
        },
        {"price": 100.0},
        {"side": "SHORT", "exitPrice": 100.0},
    )

    assert profile["postExitReentryEntryMode"] == "SHALLOW_PULLBACK"
    assert round(profile["postExitReentryPullbackPctApplied"], 4) == 0.225
    assert round(profile["entryPrice"], 4) == 100.225


def test_build_post_exit_reentry_signal_reanchors_shallow_profile():
    signal = _build_candidate_signal(price=100.0, entryPrice=99.02, pullbackPct=0.98, currentAtrPct=0.30)
    built = main._build_post_exit_reentry_signal(
        _build_watch(exitPrice=100.0, lastPrice=100.0),
        signal,
        _build_candidate_opp(price=100.0, atr=0.30),
    )

    assert built["isPostExitReentry"] is True
    assert built["entryArchetype"] == main.ENTRY_ARCHETYPE_CONTINUATION
    assert built["postExitReentryEntryMode"] == "SHALLOW_PULLBACK"
    assert round(built["postExitReentryPullbackPctApplied"], 4) == 0.225
    assert round(built["signalPrice"], 4) == 100.0
    assert round(built["entryPrice"], 4) == 99.775
    assert built["postExitReentryConfirmDelaySec"] == 30
    assert built["postExitReentryExpiresSec"] == 240


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


def test_register_post_exit_reentry_watch_arms_for_profit_giveback_exit(monkeypatch):
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {})
    monkeypatch.setattr(main.global_paper_trader, "positions", [])
    monkeypatch.setattr(main.global_paper_trader, "pending_orders", [])

    watch = main.register_post_exit_reentry_watch(
        _build_reclaim_pos(),
        _build_closed_trade(reason="SMART_V3_RUNNER__PROFIT_GIVEBACK_EXIT"),
        original_reason="SMART_V3_RUNNER__PROFIT_GIVEBACK_EXIT",
        normalized_reason="SMART_V3_RUNNER__PROFIT_GIVEBACK_EXIT",
        exit_price=100.55,
    )

    assert watch is not None
    assert watch["exitReason"] == "PROFIT_GIVEBACK_EXIT"
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_ARMED


def test_register_post_exit_reentry_watch_ignores_current_closing_position_exposure(monkeypatch):
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {})
    monkeypatch.setattr(main.global_paper_trader, "positions", [_build_reclaim_pos(_closing=True)])
    monkeypatch.setattr(main.global_paper_trader, "pending_orders", [])

    watch = main.register_post_exit_reentry_watch(
        _build_reclaim_pos(_closing=True),
        _build_closed_trade(),
        original_reason="SMART_V3_RUNNER__SIDEWAYS_RECLAIM_CLOSE",
        normalized_reason="SMART_V3_RUNNER__SIDEWAYS_RECLAIM_CLOSE",
        exit_price=100.25,
    )

    assert watch is not None
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_ARMED


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


def test_should_suppress_generic_signal_for_post_exit_watch_same_side(monkeypatch):
    monkeypatch.setattr(
        main,
        "post_exit_reentry_watchers",
        {"AAVEUSDT": _build_watch(symbol="AAVEUSDT", watchState=main.POST_EXIT_REENTRY_STATE_ARMED)},
    )

    assert main._should_suppress_generic_signal_for_post_exit_watch(
        {"symbol": "AAVEUSDT", "action": "LONG"}
    ) is True
    assert main._should_suppress_generic_signal_for_post_exit_watch(
        {"symbol": "AAVEUSDT", "action": "SHORT"}
    ) is False


def test_open_position_applies_shallow_reentry_pending_profile(monkeypatch):
    trader = main.global_paper_trader
    monkeypatch.setattr(trader, "enabled", True)
    monkeypatch.setattr(trader, "symbol", "AAVEUSDT")
    monkeypatch.setattr(trader, "positions", [])
    monkeypatch.setattr(trader, "pending_orders", [])
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

    signal = main._build_post_exit_reentry_signal(
        _build_watch(exitPrice=100.0, lastPrice=100.0),
        _build_candidate_signal(
            price=100.0,
            entryPrice=99.02,
            pullbackPct=0.98,
            currentAtrPct=0.30,
            leverage=8,
            confidenceScore=88.0,
            spreadPct=0.05,
        ),
        _build_candidate_opp(price=100.0, atr=0.30),
    )

    start_ms = int(time.time() * 1000)
    pending = asyncio.run(trader.open_position("LONG", 100.0, 0.30, signal, symbol="AAVEUSDT"))

    assert pending is not None
    assert pending["isPostExitReentry"] is True
    assert pending["postExitReentryEntryMode"] == "SHALLOW_PULLBACK"
    assert round(float(pending["pullbackPct"]), 4) == 0.225
    assert round(float(pending["entryPrice"]), 4) == 99.775
    assert pending["postExitReentryConfirmDelaySec"] == 30
    assert pending["postExitReentryExpiresSec"] == 240
    assert 29000 <= pending["confirmAfter"] - start_ms <= 31000
    assert 239000 <= pending["expiresAt"] - start_ms <= 241000


def test_evaluate_post_exit_reentry_watchers_confirms_after_persistence(monkeypatch):
    events = []
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)

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


def test_evaluate_post_exit_reentry_watchers_requires_supportive_structure(monkeypatch):
    events = []
    watch = _build_watch(
        cooldownUntilTs=1000.0,
        patternBias="CONTINUATION",
        patternConfidence=0.72,
        breakoutRetestState="BULL_RETEST_HOLD",
    )
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)

    async def _fake_dispatch(safe_watch, signal, opportunity):
        safe_watch["watchState"] = main.POST_EXIT_REENTRY_STATE_CONFIRMED
        safe_watch["reentryUsed"] = True
        safe_watch["reentryTriggered"] = True
        events.append((safe_watch["watcherId"], signal["symbol"], opportunity["symbol"]))
        return True

    monkeypatch.setattr(main, "dispatch_post_exit_reentry_watch", _fake_dispatch)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    asyncio.run(
        main.evaluate_post_exit_reentry_watchers(
            [_build_candidate_opp(patternBias="CONTINUATION", patternConfidence=0.78, breakoutRetestState="BULL_RETEST_HOLD")],
            [_build_candidate_signal(patternBias="CONTINUATION", patternConfidence=0.78, breakoutRetestState="BULL_RETEST_HOLD")],
        )
    )
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_CANDIDATE

    monkeypatch.setattr(main.time, "time", lambda: 1061.0)
    asyncio.run(
        main.evaluate_post_exit_reentry_watchers(
            [_build_candidate_opp(patternBias="CONTINUATION", patternConfidence=0.78, breakoutRetestState="BULL_RETEST_HOLD")],
            [_build_candidate_signal(patternBias="CONTINUATION", patternConfidence=0.78, breakoutRetestState="BULL_RETEST_HOLD")],
        )
    )

    assert events == [("PEW_AAVE", "AAVEUSDT", "AAVEUSDT")]
    assert watch["reentryTriggered"] is True


def test_evaluate_post_exit_reentry_watchers_blocks_on_structure_failure(monkeypatch):
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    asyncio.run(
        main.evaluate_post_exit_reentry_watchers(
            [_build_candidate_opp(patternBias="NEUTRAL", patternConfidence=0.25, breakoutRetestState="FAILED_BREAKOUT")],
            [_build_candidate_signal(patternBias="NEUTRAL", patternConfidence=0.25, breakoutRetestState="FAILED_BREAKOUT")],
        )
    )

    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_ARMED
    assert watch["confirmCount"] == 0
    assert watch["candidateSinceTs"] == 0.0


def test_evaluate_post_exit_reentry_watchers_uses_signal_fallback_without_opportunity(monkeypatch):
    events = []
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)

    async def _fake_dispatch(safe_watch, signal, opportunity):
        safe_watch["watchState"] = main.POST_EXIT_REENTRY_STATE_CONFIRMED
        safe_watch["reentryUsed"] = True
        safe_watch["reentryTriggered"] = True
        safe_watch["reentryTriggerReason"] = "CONTINUATION_REENTRY_DISPATCHED"
        events.append((safe_watch["watcherId"], signal["symbol"], opportunity["symbol"], round(opportunity["price"], 4)))
        return True

    signal = _build_candidate_signal(
        price=100.65,
        currentVolumeRatio=1.28,
        currentObImbalanceTrend=2.7,
        currentImbalance=2.5,
        currentExecScore=74.0,
        currentSpreadPct=0.05,
        currentAtrPct=0.9,
        continuationFlowState=main.V3_CONTINUATION_SUPPORTING,
        runnerContextResolved=main.V3_RUNNER_CONTEXT_INTRADAY,
        entryArchetype=main.ENTRY_ARCHETYPE_RECLAIM,
        directionOwner="continuation",
    )

    monkeypatch.setattr(main, "dispatch_post_exit_reentry_watch", _fake_dispatch)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([], [signal]))
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_CANDIDATE
    assert watch["confirmCount"] == 1

    monkeypatch.setattr(main.time, "time", lambda: 1061.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([], [signal]))

    assert events == [("PEW_AAVE", "AAVEUSDT", "AAVEUSDT", 100.65)]
    assert watch["reentryTriggered"] is True


def test_evaluate_post_exit_reentry_watchers_cancels_on_hostile_imbalance(monkeypatch):
    watch = _build_watch(cooldownUntilTs=1000.0)
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AAVEUSDT": watch})
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    asyncio.run(
        main.evaluate_post_exit_reentry_watchers(
            [_build_candidate_opp(imbalance=-9.5)],
            [],
        )
    )

    assert "AAVEUSDT" in main.post_exit_reentry_watchers
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_ARMED


def test_evaluate_post_exit_reentry_watchers_can_flip_to_opposite_reversal(monkeypatch):
    events = []
    watch = _build_watch(
        side="SHORT",
        symbol="AEVOUSDT",
        watcherId="PEW_AEVO",
        signalId="SIG_RECLAIM_AEVO",
        positionId="POS_RECLAIM_AEVO",
        originalTradeId="TRD_RECLAIM_AEVO",
        resolutionTargetSide="SHORT",
        cooldownUntilTs=1000.0,
    )
    monkeypatch.setattr(main, "post_exit_reentry_watchers", {"AEVOUSDT": watch})
    monkeypatch.setattr(main, "post_exit_reentry_watch_history", [])
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "_has_symbol_exposure", lambda symbol, exclude_position_id='': False)

    async def _fake_dispatch(safe_watch, signal, opportunity):
        safe_watch["watchState"] = main.POST_EXIT_REENTRY_STATE_CONFIRMED
        safe_watch["reentryUsed"] = True
        safe_watch["reentryTriggered"] = True
        safe_watch["reentryTriggerReason"] = "OPPOSITE_REVERSAL"
        events.append(
            (
                safe_watch["watcherId"],
                safe_watch["resolutionMode"],
                safe_watch["resolutionTargetSide"],
                signal["action"],
                opportunity["symbol"],
            )
        )
        return True

    opposite_signal = _build_candidate_signal(
        symbol="AEVOUSDT",
        action="LONG",
        confidenceScore=78.0,
        currentExecScore=68.0,
        currentVolumeRatio=1.18,
        currentObImbalanceTrend=2.8,
        currentImbalance=9.4,
        patternBias="CONTINUATION",
        patternConfidence=0.74,
        breakoutRetestState="BULL_RETEST_HOLD",
        entryArchetype=main.ENTRY_ARCHETYPE_RECLAIM,
        directionOwner="reclaim",
        runnerContextResolved=main.V3_RUNNER_CONTEXT_COUNTER,
    )
    opposite_opp = _build_candidate_opp(
        symbol="AEVOUSDT",
        signalAction="LONG",
        signalScore=78.0,
        price=100.85,
        currentExecScore=68.0,
        entryExecScore=68.0,
        volumeRatio=1.18,
        obImbalanceTrend=2.8,
        imbalance=9.4,
        patternBias="CONTINUATION",
        patternConfidence=0.74,
        breakoutRetestState="BULL_RETEST_HOLD",
    )

    monkeypatch.setattr(main, "dispatch_post_exit_reentry_watch", _fake_dispatch)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([opposite_opp], [opposite_signal]))
    assert watch["watchState"] == main.POST_EXIT_REENTRY_STATE_CANDIDATE
    assert watch["resolutionMode"] == main.POST_EXIT_RESOLUTION_MODE_OPPOSITE
    assert watch["resolutionTargetSide"] == "LONG"

    monkeypatch.setattr(main.time, "time", lambda: 1061.0)
    asyncio.run(main.evaluate_post_exit_reentry_watchers([opposite_opp], [opposite_signal]))

    assert events == [("PEW_AEVO", main.POST_EXIT_RESOLUTION_MODE_OPPOSITE, "LONG", "LONG", "AEVOUSDT")]
    assert watch["reentryTriggered"] is True


def test_coin_state_resolution_helper_blocks_same_side_when_opposite_side_is_dominant():
    ok, reason, boost = main._coin_state_supports_resolution(
        main.POST_EXIT_RESOLUTION_MODE_SAME_SIDE,
        {
            "setupState15m": "REVERSAL_RETEST",
            "transitionState": "FAILED_BREAKDOWN",
            "dominantSide": "LONG",
            "stateConfidence": 0.74,
            "allowedEntryFamilies": ["reversal_retest", "reclaim"],
            "preferredExitProfile": main.V3_EXIT_PROFILE_TRANSITION_DEFENSE,
        },
        "SHORT",
    )

    assert ok is False
    assert reason == "COIN_STATE_OPPOSITE_DOMINANT"
    assert boost == 0.0


def test_coin_state_resolution_helper_supports_opposite_reversal_when_transition_is_active():
    ok, reason, boost = main._coin_state_supports_resolution(
        main.POST_EXIT_RESOLUTION_MODE_OPPOSITE,
        {
            "setupState15m": "REVERSAL_RETEST",
            "transitionState": "FAILED_BREAKDOWN",
            "dominantSide": "LONG",
            "stateConfidence": 0.78,
            "allowedEntryFamilies": ["reversal_retest", "reclaim"],
            "preferredExitProfile": main.V3_EXIT_PROFILE_TRANSITION_DEFENSE,
        },
        "LONG",
    )

    assert ok is True
    assert reason == "COIN_STATE_OPPOSITE_OK"
    assert boost > 0.0


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
    monkeypatch.setattr(
        main,
        "post_exit_reentry_watch_history",
        [
            {"ts": 1000.0, "reason": "armed", "resolutionMode": main.POST_EXIT_RESOLUTION_MODE_SAME_SIDE, "reentryTriggered": False},
            {"ts": 1010.0, "reason": "cancelled", "resolutionMode": main.POST_EXIT_RESOLUTION_MODE_SAME_SIDE, "reentryTriggered": False},
            {"ts": 1020.0, "reason": "confirmed", "resolutionMode": main.POST_EXIT_RESOLUTION_MODE_OPPOSITE, "reentryTriggered": True},
        ],
    )

    summary = main.build_post_exit_reentry_watchers_summary(now_ts=1005.0)

    assert summary["active"] == 2
    assert summary["candidate"] == 1
    assert summary["confirmed"] == 1
    assert summary["triggered"] == 1
    assert summary["recentArmed"] == 1
    assert summary["recentCancelled"] == 1
    assert summary["recentTriggered"] == 1
    assert summary["recentOppositeTriggered"] == 1


def test_build_post_exit_reentry_signal_defaults_opposite_mode_to_reversal_retest():
    watch = _build_watch(
        symbol="AEVOUSDT",
        side="SHORT",
        resolutionMode=main.POST_EXIT_RESOLUTION_MODE_OPPOSITE,
        resolutionTargetSide="LONG",
        resolutionEntryArchetype="",
    )
    signal = _build_candidate_signal(
        symbol="AEVOUSDT",
        action="LONG",
        entryArchetype="",
        decisionContext={},
        directionOwner="",
    )
    opp = _build_candidate_opp(
        symbol="AEVOUSDT",
        signalAction="LONG",
        price=100.8,
        breakoutRetestState="BULL_RETEST_HOLD",
        patternBias="CONTINUATION",
        patternConfidence=0.72,
    )

    built = main._build_post_exit_reentry_signal(watch, signal, opp)

    assert built["entryArchetype"] == main.ENTRY_ARCHETYPE_REVERSAL_RETEST


def test_build_post_exit_reentry_signal_exposes_coin_state_route_reason():
    watch = {
        "symbol": "AEVOUSDT",
        "side": "SHORT",
        "resolutionMode": main.POST_EXIT_RESOLUTION_MODE_OPPOSITE,
        "resolutionTargetSide": "LONG",
        "watcherId": "PEW_TEST",
        "originalTradeId": "TRD_TEST",
        "exitReason": "SIDEWAYS_RECLAIM_CLOSE",
    }
    signal = {
        "symbol": "AEVOUSDT",
        "action": "LONG",
        "price": 100.0,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "setupState15m": "REVERSAL_RETEST",
        "transitionState": "FAILED_BREAKDOWN",
        "dominantSide": "LONG",
        "stateConfidence": 0.71,
        "allowedEntryFamilies": ["reversal_retest", "reclaim"],
    }
    opp = {"symbol": "AEVOUSDT", "price": 100.0}

    built = main._build_post_exit_reentry_signal(watch, signal, opp)

    assert built["entryArchetype"] == main.ENTRY_ARCHETYPE_REVERSAL_RETEST
    assert built["coinStateRouteReason"] == "COIN_STATE_ROUTE_REVERSAL"
    assert built["directionOwner"] == "reversal_retest"


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
