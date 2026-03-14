import asyncio
import time

import main


def _build_v3_position(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "POS_TEST",
        "symbol": "TESTUSDT",
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "side": "LONG",
        "entryPrice": 100.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "marginUsd": 10.0,
        "initialMargin": 10.0,
        "leverage": 10,
        "openTime": now_ms - (20 * 60 * 1000),
        "stopLoss": 99.2,
        "takeProfit": 103.0,
        "trailActivation": 100.8,
        "trailDistance": 0.3,
        "runtimeTrailDistance": 0.3,
        "runtimeTrailActivationRoiPct": 6.0,
        "runtimeTrailDistanceRoiPct": 3.0,
        "spreadPct": 0.05,
        "volumeRatio": 1.0,
        "currentVolumeRatio": 1.0,
        "currentIsVolumeSpike": False,
        "currentImbalance": 0.0,
        "currentObImbalanceTrend": 0.0,
        "currentExecScore": 72.0,
        "entryImbalance": 0.0,
        "obImbalanceTrend": 0.0,
        "atr": 0.3,
        "runnerContext": main.V3_RUNNER_CONTEXT_TREND,
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.4, "roi_pct": 4.0, "close_pct": 0.1},
            {"key": "tp2", "price_pct": 0.8, "roi_pct": 8.0, "close_pct": 0.15},
            {"key": "tp3", "price_pct": 1.2, "roi_pct": 12.0, "close_pct": 0.25},
            {"key": "tp_final", "price_pct": 1.8, "roi_pct": 18.0, "close_pct": 0.5},
        ],
        "partial_tp_state": {},
        "profitPeakRoiPct": 0.0,
        "runtimeProfitPeakRoiPct": 0.0,
        "runtimeBreakevenFloorRoiPct": 0.0,
        "isLive": False,
    }
    pos.update(overrides)
    return pos


def test_classify_v3_runner_context_intraday_continuation_on_neutral_daily_trend():
    ctx = main.classify_v3_runner_context(
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "coinDailyTrend": "NEUTRAL",
            "adx": 29.0,
            "hurst": 0.57,
            "volumeRatio": 1.35,
            "isVolumeSpike": False,
            "obImbalanceTrend": 2.6,
        },
        default_mode=main.STRATEGY_MODE_SMART_V3_RUNNER,
    )

    assert ctx == main.V3_RUNNER_CONTEXT_INTRADAY


def test_classify_v3_runner_context_keeps_countertrend_precedence():
    ctx = main.classify_v3_runner_context(
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "coinDailyTrend": "BEARISH",
            "adx": 33.0,
            "hurst": 0.60,
            "volumeRatio": 1.8,
            "isVolumeSpike": True,
            "obImbalanceTrend": 3.1,
        },
        default_mode=main.STRATEGY_MODE_SMART_V3_RUNNER,
    )

    assert ctx == main.V3_RUNNER_CONTEXT_COUNTER


def test_apply_live_flow_context_to_position_updates_current_runtime_fields():
    pos = _build_v3_position(currentVolumeRatio=1.0, currentObImbalanceTrend=0.0, currentImbalance=0.0)
    opp = {
        "volumeRatio": 1.42,
        "isVolumeSpike": True,
        "imbalance": 4.5,
        "obImbalanceTrend": 2.8,
        "spreadPct": 0.07,
        "atrPct": 1.9,
        "microstructureScore": 73.0,
        "flowToxicityScore": 36.0,
        "entryExecScore": 81.0,
    }

    main.apply_live_flow_context_to_position(pos, opp)

    assert pos["currentVolumeRatio"] == 1.42
    assert pos["currentIsVolumeSpike"] is True
    assert pos["currentImbalance"] == 4.5
    assert pos["currentObImbalanceTrend"] == 2.8
    assert pos["currentExecScore"] == 81.0


def test_apply_live_flow_context_to_pending_order_updates_current_runtime_fields():
    order = {
        "id": "PEND_TEST",
        "symbol": "TESTUSDT",
        "side": "LONG",
        "volumeRatio": 1.0,
        "currentVolumeRatio": 1.0,
        "currentIsVolumeSpike": False,
        "currentImbalance": 0.0,
        "currentObImbalanceTrend": 0.0,
        "currentSpreadPct": 0.05,
        "currentAtrPct": 0.9,
        "currentExecScore": 70.0,
        "entryExecScore": 70.0,
    }
    opp = {
        "volumeRatio": 1.36,
        "isVolumeSpike": True,
        "imbalance": 6.0,
        "obImbalanceTrend": 2.4,
        "spreadPct": 0.08,
        "volatilityPct": 1.7,
        "microstructureScore": 76.0,
        "flowToxicityScore": 34.0,
    }

    main.apply_live_flow_context_to_pending_order(order, opp)

    assert order["currentVolumeRatio"] == 1.36
    assert order["currentIsVolumeSpike"] is True
    assert order["currentImbalance"] == 6.0
    assert order["currentObImbalanceTrend"] == 2.4
    assert order["currentAtrPct"] == 1.7
    assert order["currentExecScore"] >= main.EXEC_QUALITY_MIN_SCORE


def test_coin_opportunity_reset_scan_signal_state_clears_stale_signal_fields():
    opp = main.CoinOpportunity("TESTUSDT")
    opp.signal_score = 88
    opp.signal_action = "LONG"
    opp.pullback_pct = 1.3
    opp.entry_quality_pass = True
    opp.entry_quality_reasons = ["A:Vol"]
    opp.fib_active = True
    opp.entry_price = 101.2
    opp.strategy_mode = main.STRATEGY_MODE_SMART_V3_RUNNER
    opp.execution_reject_reason = "OLD"
    opp.recheck_score = 77.0
    opp.recheck_reasons = ["OLD"]

    opp.reset_scan_signal_state()
    opp.update_live_flow_snapshot(
        volume_ratio=0.91,
        is_volume_spike=False,
        ob_imbalance_trend=-0.4,
        live_exec_score=71.0,
    )

    assert opp.signal_score == 0
    assert opp.signal_action == "NONE"
    assert opp.pullback_pct == 0.0
    assert opp.entry_quality_pass is False
    assert opp.entry_quality_reasons == []
    assert opp.fib_active is False
    assert opp.entry_price == 0.0
    assert opp.execution_reject_reason is None
    assert opp.recheck_score == 0.0
    assert opp.volume_ratio == 0.91
    assert opp.entry_exec_score == 71.0


def test_event_alpha_trigger_uses_live_exec_fallback_without_signal_snapshot():
    opp = {
        "symbol": "TESTUSDT",
        "signalAction": "NONE",
        "signalScore": 0.0,
        "zscore": -1.35,
        "spreadPct": 0.05,
        "volumeRatio": 1.32,
        "isVolumeSpike": False,
        "imbalance": 9.0,
        "obImbalanceTrend": 3.4,
        "adx": 28.0,
        "hurst": 0.58,
        "microstructureScore": 74.0,
        "flowToxicityScore": 32.0,
        "liqEchoScore": 7.0,
        "liqEchoState": "SELL_SWEEP",
    }

    result = main.evaluate_event_alpha_trigger(opp)

    assert result["triggered"] is True
    assert result["side"] == "LONG"
    assert result["reason"].startswith("LIQ_ECHO")


def test_evaluate_v3_continuation_flow_state_detects_supporting_long_flow():
    pos = _build_v3_position(
        currentVolumeRatio=1.28,
        currentIsVolumeSpike=False,
        currentImbalance=2.0,
        currentObImbalanceTrend=2.6,
        currentExecScore=79.0,
    )

    result = main.evaluate_v3_continuation_flow_state(
        pos,
        current_price=100.08,
        current_roi_pct=0.8,
        profit_peak_roi_pct=1.4,
        breakeven_floor_roi_pct=0.5,
    )

    assert result["state"] == main.V3_CONTINUATION_SUPPORTING
    assert result["supportive"] is True
    assert result["fading"] is False


def test_evaluate_v3_continuation_flow_state_detects_chop_after_giveback():
    pos = _build_v3_position(
        currentVolumeRatio=0.82,
        currentIsVolumeSpike=False,
        currentImbalance=0.0,
        currentObImbalanceTrend=0.3,
        currentExecScore=71.0,
        runtimeProfitPeakRoiPct=9.5,
        profitPeakRoiPct=9.5,
    )

    result = main.evaluate_v3_continuation_flow_state(
        pos,
        current_price=100.10,
        current_roi_pct=1.0,
        profit_peak_roi_pct=9.5,
        breakeven_floor_roi_pct=0.4,
    )

    assert result["state"] == main.V3_CONTINUATION_CHOP
    assert result["chop"] is True
    assert result["giveback_roi_pct"] >= 8.0


def test_apply_v3_continuation_trail_overlay_widens_supporting_and_tightens_chop():
    supporting_pos = _build_v3_position(
        currentVolumeRatio=1.30,
        currentObImbalanceTrend=2.7,
        currentImbalance=1.0,
    )
    supporting = main.apply_v3_continuation_trail_overlay(
        supporting_pos,
        current_price=100.10,
        dynamic_trail_distance=1.0,
        min_price_move_pct=0.8,
        min_roi_pct=8.0,
        threshold_mult=1.0,
        current_roi_pct=1.0,
    )

    chop_pos = _build_v3_position(
        currentVolumeRatio=0.84,
        currentObImbalanceTrend=0.1,
        currentImbalance=0.0,
        partial_tp_state={"tp1": True},
        runtimeTp1RoiPct=4.0,
        runtimeProfitPeakRoiPct=9.0,
        profitPeakRoiPct=9.0,
    )
    chop = main.apply_v3_continuation_trail_overlay(
        chop_pos,
        current_price=100.08,
        dynamic_trail_distance=1.0,
        min_price_move_pct=0.8,
        min_roi_pct=8.0,
        threshold_mult=1.0,
        current_roi_pct=0.8,
    )

    assert round(supporting["trail_distance"], 4) == 1.2
    assert round(supporting["min_price_move_pct"], 4) == 0.864
    assert chop["state"] == main.V3_CONTINUATION_CHOP
    assert round(chop["trail_distance"], 4) == 0.75
    assert chop["chop_tighten_active"] is True


def test_compute_limited_market_override_plan_requires_guarded_continuation():
    eligible = main.compute_limited_market_override_plan(
        {
            "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
            "volumeRatio": 2.4,
            "isVolumeSpike": True,
            "spreadPct": 0.08,
            "entryExecStrictPassed": True,
            "microstructureScore": 74.0,
            "forecastBand": "MEDIUM",
            "pendingPassRequired": True,
        },
        pending_passed=True,
    )
    ineligible = main.compute_limited_market_override_plan(
        {
            "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
            "volumeRatio": 2.8,
            "isVolumeSpike": True,
            "spreadPct": 0.06,
            "entryExecStrictPassed": True,
            "microstructureScore": 88.0,
            "forecastBand": "STRONG",
        },
        pending_passed=True,
    )

    assert eligible["eligible"] is True
    assert eligible["reason"] == "LIMITED_MARKET_CONTINUATION"
    assert ineligible["eligible"] is False


def test_kill_switch_arms_one_shot_pre_stop_hold_for_supportive_v3(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []
            self.pipeline_metrics = {}

        def add_log(self, _msg):
            return None

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

        def _normalize_close_reason(self, reason):
            return reason

    async def _noop_save_trade(_trade):
        return None

    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: coro.close() if hasattr(coro, "close") else None)
    monkeypatch.setattr(main.sqlite_manager, "save_trade", _noop_save_trade)

    pos = _build_v3_position(
        symbol="HOLDUSDT",
        currentPrice=99.97,
        unrealizedPnl=-0.03,
        unrealizedPnlPercent=-0.3,
        stopLoss=99.95,
        takeProfit=101.0,
        trailActivation=100.4,
        trailDistance=0.2,
        runtimeTrailDistance=0.2,
        runtimeTrailActivationRoiPct=2.0,
        atr=0.2,
        currentVolumeRatio=1.45,
        currentIsVolumeSpike=False,
        currentImbalance=1.5,
        currentObImbalanceTrend=2.9,
        runnerContext=main.V3_RUNNER_CONTEXT_INTRADAY,
    )
    trader = DummyTrader(pos)
    kill_switch = main.PositionBasedKillSwitch()

    asyncio.run(kill_switch.check_positions(trader))

    assert pos["preStopHoldArmed"] is True
    assert pos["preStopHoldUsed"] is True
    assert pos.get("preStopReduced", False) is False
    assert trader.trades == []
    assert trader.pipeline_metrics["v3_pre_stop_hold_count"] == 1


def test_v3_runner_reclaim_close_exits_chop_after_tp1(monkeypatch):
    class DummyTrader:
        def __init__(self):
            self.closed = []
            self.pipeline_metrics = {}

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

    time_manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(time_manager, "_get_v3_opposite_pressure", lambda pos: (False, 0.0))
    monkeypatch.setattr(time_manager, "_select_triggered_tp_slice_trail", lambda pos, price: None)
    monkeypatch.setattr(main, "apply_market_structure_context", lambda payload, **kwargs: payload)
    trader = DummyTrader()

    pos = _build_v3_position(
        symbol="CHOPUSDT",
        currentVolumeRatio=0.85,
        currentObImbalanceTrend=0.0,
        currentImbalance=0.0,
        partial_tp_state={"tp1": True},
        runtimeTp1RoiPct=4.0,
        runtimeProfitPeakRoiPct=10.0,
        profitPeakRoiPct=10.0,
        chopSinceTs=time.time() - 700.0,
        runnerContext=main.V3_RUNNER_CONTEXT_INTRADAY,
    )
    actions = {"time_closed": [], "partial_tp": []}

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 100.15, actions))

    assert handled is True
    assert actions["time_closed"] == ["CHOPUSDT reclaim_be"]


def test_resolve_v3_exit_behavior_profile_prefers_coin_state_profile_when_confident():
    pos = _build_v3_position(
        side="SHORT",
        preferredExitProfile=main.V3_EXIT_PROFILE_TRANSITION_DEFENSE,
        stateConfidence=0.72,
        transitionState="FAILED_BREAKOUT",
        setupState15m="REVERSAL_RETEST",
        structureTrend="UP",
        patternBias="RECLAIM",
        patternConfidence=0.66,
        continuationFlowState=main.V3_CONTINUATION_FADING,
    )

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=99.8,
        profit_ladder={"continuation_flow_state": main.V3_CONTINUATION_FADING, "current_profit_roi_pct": -1.0},
        underwater_ctx={"state": main.V3_UNDERWATER_ADVERSE_WEAK, "price_loss_pct": 0.8, "small_loss_band_pct": 1.2},
        thesis_ctx={"thesis_state": main.POSITION_THESIS_REVALIDATING},
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_TRANSITION_DEFENSE
    assert profile["reason"] == "COIN_STATE_TRANSITION_DEFENSE"
    assert profile["exit_owner"] == main.V3_EXIT_OWNER_TRANSITION_PROTECT


def test_v3_runner_reclaim_close_holds_small_loss_when_trend_expansion_profile_is_active(monkeypatch):
    pos = _build_v3_position(
        symbol="TRENDHOLDUSDT",
        trend_mode=True,
        structureTrend="UP",
        breakoutRetestState="BULL_RETEST_HOLD",
        srContext="ABOVE_SUPPORT",
        patternBias="CONTINUATION",
        patternConfidence=0.84,
        currentVolumeRatio=0.86,
        currentIsVolumeSpike=False,
        currentObImbalanceTrend=0.0,
        currentImbalance=0.0,
        currentAtrPct=0.8,
        currentPrice=99.96,
        openTime=int((time.time() - 30 * 60) * 1000),
        sidewaysSinceTs=time.time() - 800.0,
        worstUnderwaterPrice=99.4,
        worstUnderwaterTs=time.time() - 900.0,
        runtimeBreakevenFloorRoiPct=0.5,
        runnerContext=main.V3_RUNNER_CONTEXT_TREND,
    )
    profit_ladder = {
        "current_profit_roi_pct": -0.4,
        "continuation_flow_state": main.V3_CONTINUATION_SUPPORTING,
    }
    underwater_ctx = {
        "state": main.V3_UNDERWATER_SIDEWAYS,
        "price_loss_pct": 0.04,
        "small_loss_band_pct": 1.0,
        "adverse_flow": False,
        "hostile_imbalance": False,
    }
    thesis_ctx = {"thesis_state": main.POSITION_THESIS_ENTRY}

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=99.96,
        profit_ladder=profit_ladder,
        underwater_ctx=underwater_ctx,
        thesis_ctx=thesis_ctx,
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_TREND_EXPANSION
    assert profile["allow_sideways_reclaim_hold"] is True


def test_v3_runner_reclaim_close_uses_faster_range_efficiency_chop_threshold(monkeypatch):
    class DummyTrader:
        def __init__(self):
            self.closed = []
            self.pipeline_metrics = {}

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

    time_manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(time_manager, "_get_v3_opposite_pressure", lambda pos: (False, 0.0))
    monkeypatch.setattr(time_manager, "_select_triggered_tp_slice_trail", lambda pos, price: None)
    trader = DummyTrader()

    pos = _build_v3_position(
        symbol="RANGEFASTUSDT",
        structureTrend="RANGE",
        patternBias="NEUTRAL",
        patternConfidence=0.28,
        currentVolumeRatio=0.84,
        currentObImbalanceTrend=0.0,
        currentImbalance=0.0,
        partial_tp_state={"tp1": True},
        runtimeTp1RoiPct=4.0,
        runtimeProfitPeakRoiPct=10.0,
        profitPeakRoiPct=10.0,
        chopSinceTs=time.time() - 350.0,
        runnerContext=main.V3_RUNNER_CONTEXT_INTRADAY,
    )
    actions = {"time_closed": [], "partial_tp": []}

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 100.15, actions))

    assert handled is True
    assert actions["time_closed"] == ["RANGEFASTUSDT reclaim_be"]
    assert trader.closed[0][2] == "RECLAIM_BE_CLOSE"
    assert pos["runtimeExitProfile"] == main.V3_EXIT_PROFILE_RANGE_EFFICIENCY


def test_resolve_v3_exit_behavior_profile_detects_aged_range_stagnation():
    pos = _build_v3_position(
        openTime=int((time.time() - (5 * 60 * 60)) * 1000),
        setupState15m="RECLAIM",
        backdropState1h="RANGE",
        macroState4h="RANGE",
        stateConfidence=0.38,
        preferredExitProfile=main.V3_EXIT_PROFILE_BALANCED,
        structureTrend="MIXED",
        patternBias="NEUTRAL",
        patternConfidence=0.24,
        continuationFlowState=main.V3_CONTINUATION_CHOP,
    )

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=99.9,
        profit_ladder={"continuation_flow_state": main.V3_CONTINUATION_CHOP, "current_profit_roi_pct": -1.4},
        underwater_ctx={"state": main.V3_UNDERWATER_SIDEWAYS, "price_loss_pct": 0.4, "small_loss_band_pct": 1.0},
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_RANGE_EFFICIENCY
    assert profile["reason"] == "AGED_RANGE_STAGNATION"
    assert profile["chop_reclaim_min_age_sec"] == 150.0
    assert profile["exit_owner"] == main.V3_EXIT_OWNER_FAILED_THESIS


def test_resolve_v3_exit_behavior_profile_promotes_strong_mtf_trend_continuation():
    pos = _build_v3_position(
        trend_mode=False,
        openTime=int((time.time() - (90 * 60)) * 1000),
        setupState15m="CONTINUATION",
        backdropState1h="TREND",
        macroState4h="TREND",
        stateConfidence=0.74,
        preferredExitProfile=main.V3_EXIT_PROFILE_BALANCED,
        structureTrend="UP",
        srContext="ABOVE_SUPPORT",
        breakoutRetestState="BULL_RETEST_HOLD",
        patternBias="CONTINUATION",
        patternConfidence=0.82,
        continuationFlowState=main.V3_CONTINUATION_SUPPORTING,
    )

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=100.6,
        profit_ladder={"continuation_flow_state": main.V3_CONTINUATION_SUPPORTING, "current_profit_roi_pct": 6.0},
        underwater_ctx={"state": main.V3_UNDERWATER_NEUTRAL, "price_loss_pct": 0.0, "small_loss_band_pct": 1.0},
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_TREND_EXPANSION
    assert profile["reason"] == "MTF_TREND_CONTINUATION"
    assert profile["exit_owner"] == main.V3_EXIT_OWNER_TREND_CONTINUATION


def test_resolve_v3_exit_behavior_profile_keeps_trend_owner_on_supportive_breakout_retest():
    pos = _build_v3_position(
        side="LONG",
        trend_mode=False,
        openTime=int((time.time() - (75 * 60)) * 1000),
        setupState15m="CONTINUATION",
        backdropState1h="TREND",
        macroState4h="TRANSITION",
        transitionState="BREAKOUT_RETEST",
        stateConfidence=0.71,
        preferredExitProfile=main.V3_EXIT_PROFILE_BALANCED,
        structureTrend="UP",
        srContext="ABOVE_SUPPORT",
        breakoutRetestState="BULL_RETEST_HOLD",
        patternBias="CONTINUATION",
        patternConfidence=0.84,
        continuationFlowState=main.V3_CONTINUATION_SUPPORTING,
    )

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=100.5,
        profit_ladder={"continuation_flow_state": main.V3_CONTINUATION_SUPPORTING, "current_profit_roi_pct": 5.0},
        underwater_ctx={"state": main.V3_UNDERWATER_NEUTRAL, "price_loss_pct": 0.0, "small_loss_band_pct": 1.0},
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_TREND_EXPANSION
    assert profile["exit_owner"] == main.V3_EXIT_OWNER_TREND_CONTINUATION
    assert profile["exit_owner_allow_hold"] is True


def test_resolve_v3_exit_owner_uses_transition_protect_for_adverse_failed_breakout():
    owner = main.resolve_v3_exit_owner(
        profile=main.V3_EXIT_PROFILE_TREND_EXPANSION,
        drift_state="ALIGNED",
        drift_severity=0.0,
        continuation_state=main.V3_CONTINUATION_SUPPORTING,
        thesis_state=main.POSITION_THESIS_ENTRY,
        current_roi_pct=4.0,
        age_sec=30 * 60,
        setup_state="CONTINUATION",
        transition_state="FAILED_BREAKOUT",
        dominant_side="LONG",
        side="LONG",
    )

    assert owner["owner"] == main.V3_EXIT_OWNER_TRANSITION_PROTECT
    assert owner["allow_hold"] is False


def test_resolve_v3_exit_behavior_profile_detects_opposite_dominant_state_drift():
    pos = _build_v3_position(
        openTime=int((time.time() - (80 * 60)) * 1000),
        setupState15m="CONTINUATION",
        backdropState1h="MIXED",
        macroState4h="TRANSITION",
        transitionState="FAILED_BREAKOUT",
        dominantSide="SHORT",
        stateConfidence=0.78,
        preferredExitProfile=main.V3_EXIT_PROFILE_BALANCED,
        structureTrend="MIXED",
        patternBias="NEUTRAL",
        patternConfidence=0.32,
        continuationFlowState=main.V3_CONTINUATION_NEUTRAL,
    )

    profile = main.resolve_v3_exit_behavior_profile(
        pos,
        current_price=99.6,
        profit_ladder={"continuation_flow_state": main.V3_CONTINUATION_NEUTRAL, "current_profit_roi_pct": -0.8},
        underwater_ctx={"state": main.V3_UNDERWATER_ADVERSE_WEAK, "price_loss_pct": 0.7, "small_loss_band_pct": 1.0},
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert profile["profile"] == main.V3_EXIT_PROFILE_TRANSITION_DEFENSE
    assert profile["reason"] == "COIN_STATE_OPPOSITE_DOMINANT"
    assert profile["drift_state"] == "OPPOSITE_DOMINANT"
    assert profile["intent_decay_pct"] > 0.5
    assert profile["exit_owner"] == main.V3_EXIT_OWNER_REVERSAL_ESCAPE


def test_evaluate_v3_underwater_tape_state_detects_sideways_small_loss():
    pos = _build_v3_position(
        currentVolumeRatio=0.84,
        currentIsVolumeSpike=False,
        currentImbalance=0.0,
        currentObImbalanceTrend=0.2,
        currentAtrPct=0.6,
    )

    result = main.evaluate_v3_underwater_tape_state(
        pos,
        current_price=99.7,
        current_roi_pct=-3.0,
    )

    assert result["state"] == main.V3_UNDERWATER_SIDEWAYS
    assert result["small_loss_band_pct"] == 1.0


def test_evaluate_v3_underwater_tape_state_detects_recovering_after_meaningful_underwater():
    pos = _build_v3_position(
        currentVolumeRatio=0.90,
        currentIsVolumeSpike=False,
        currentImbalance=0.0,
        currentObImbalanceTrend=0.1,
        atr=0.5,
        worstUnderwaterPrice=98.8,
        worstUnderwaterTs=time.time() - 900.0,
    )

    result = main.evaluate_v3_underwater_tape_state(
        pos,
        current_price=99.2,
        current_roi_pct=-8.0,
    )

    assert result["state"] == main.V3_UNDERWATER_RECOVERING
    assert result["recovered_from_worst"] is True


def test_evaluate_v3_underwater_tape_state_detects_adverse_strong():
    pos = _build_v3_position(
        currentVolumeRatio=1.42,
        currentIsVolumeSpike=False,
        currentImbalance=-11.0,
        currentObImbalanceTrend=-3.1,
        currentAtrPct=1.0,
    )

    result = main.evaluate_v3_underwater_tape_state(
        pos,
        current_price=97.8,
        current_roi_pct=-22.0,
        signal_invalidation_state={"mode": "opposite_signal", "triggerReady": True, "oppositeCount": 2},
    )

    assert result["state"] == main.V3_UNDERWATER_ADVERSE_STRONG
    assert result["opposite_signal_persistent"] is True


def test_evaluate_v3_underwater_tape_state_detects_adverse_weak():
    pos = _build_v3_position(
        currentVolumeRatio=1.08,
        currentIsVolumeSpike=False,
        currentImbalance=-3.5,
        currentObImbalanceTrend=-2.3,
        currentAtrPct=1.2,
    )

    result = main.evaluate_v3_underwater_tape_state(
        pos,
        current_price=99.1,
        current_roi_pct=-9.0,
    )

    assert result["state"] == main.V3_UNDERWATER_ADVERSE_WEAK


def test_evaluate_v3_underwater_tape_state_clamps_small_loss_band():
    low_atr_pos = _build_v3_position(currentAtrPct=0.2)
    high_atr_pos = _build_v3_position(currentAtrPct=4.0)

    low = main.evaluate_v3_underwater_tape_state(low_atr_pos, current_price=99.7, current_roi_pct=-3.0)
    high = main.evaluate_v3_underwater_tape_state(high_atr_pos, current_price=98.2, current_roi_pct=-18.0)

    assert low["small_loss_band_pct"] == 1.0
    assert high["small_loss_band_pct"] == 2.0


def test_kill_switch_suppresses_pre_stop_reduce_for_small_loss_sideways_v3(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []
            self.pipeline_metrics = {}

        def add_log(self, _msg):
            return None

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

        def _normalize_close_reason(self, reason):
            return reason

    async def _noop_save_trade(_trade):
        return None

    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: coro.close() if hasattr(coro, "close") else None)
    monkeypatch.setattr(main.sqlite_manager, "save_trade", _noop_save_trade)

    pos = _build_v3_position(
        symbol="SIDEUSDT",
        currentPrice=99.97,
        unrealizedPnl=-0.03,
        unrealizedPnlPercent=-0.3,
        stopLoss=99.95,
        takeProfit=101.0,
        trailActivation=100.4,
        trailDistance=0.2,
        runtimeTrailDistance=0.2,
        runtimeTrailActivationRoiPct=2.0,
        atr=0.2,
        currentAtrPct=0.5,
        currentVolumeRatio=0.82,
        currentIsVolumeSpike=False,
        currentImbalance=0.0,
        currentObImbalanceTrend=0.0,
        runnerContext=main.V3_RUNNER_CONTEXT_TREND,
    )
    trader = DummyTrader(pos)
    kill_switch = main.PositionBasedKillSwitch()

    asyncio.run(kill_switch.check_positions(trader))

    assert pos.get("preStopReduced", False) is False
    assert pos.get("lossGateSuppressedReason") == "PRE_STOP_REDUCE:SIDEWAYS"
    assert trader.trades == []
    assert trader.pipeline_metrics["v3_loss_gate_suppressed_count"] >= 1


def test_kill_switch_keeps_pre_stop_reduce_for_adverse_strong_v3(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []
            self.pipeline_metrics = {}

        def add_log(self, _msg):
            return None

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

        def _normalize_close_reason(self, reason):
            return reason

    async def _noop_save_trade(_trade):
        return None

    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: coro.close() if hasattr(coro, "close") else None)
    monkeypatch.setattr(main.sqlite_manager, "save_trade", _noop_save_trade)

    pos = _build_v3_position(
        symbol="ADVERSEUSDT",
        currentPrice=98.6,
        unrealizedPnl=-0.14,
        unrealizedPnlPercent=-14.0,
        stopLoss=99.3,
        takeProfit=101.0,
        trailActivation=100.4,
        trailDistance=0.2,
        runtimeTrailDistance=0.2,
        runtimeTrailActivationRoiPct=2.0,
        atr=0.25,
        currentAtrPct=1.0,
        currentVolumeRatio=1.45,
        currentIsVolumeSpike=False,
        currentImbalance=-11.0,
        currentObImbalanceTrend=-3.0,
        runnerContext=main.V3_RUNNER_CONTEXT_TREND,
    )
    pos["runtimeSignalInvalidationState"] = {"mode": "opposite_signal", "triggerReady": True, "oppositeCount": 2}
    trader = DummyTrader(pos)
    kill_switch = main.PositionBasedKillSwitch()

    asyncio.run(kill_switch.check_positions(trader))

    assert pos.get("preStopReduced", False) is True
    assert trader.trades[0]["reason"] == "PRE_STOP_REDUCE"
    assert trader.pipeline_metrics["v3_adverse_strong_reduce_count"] >= 1


def test_v3_runner_sideways_reclaim_close_exits_flat_underwater_chop(monkeypatch):
    class DummyTrader:
        def __init__(self):
            self.closed = []
            self.pipeline_metrics = {}

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

    time_manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(time_manager, "_get_v3_opposite_pressure", lambda pos: (False, 0.0))
    monkeypatch.setattr(time_manager, "_select_triggered_tp_slice_trail", lambda pos, price: None)
    trader = DummyTrader()

    pos = _build_v3_position(
        symbol="FLATUSDT",
        currentVolumeRatio=0.78,
        currentIsVolumeSpike=False,
        currentObImbalanceTrend=0.0,
        currentImbalance=0.0,
        currentAtrPct=0.8,
        currentPrice=99.96,
        openTime=int((time.time() - 30 * 60) * 1000),
        sidewaysSinceTs=time.time() - 800.0,
        worstUnderwaterPrice=99.4,
        worstUnderwaterTs=time.time() - 900.0,
        lastSupportiveFlowTs=0,
        runtimeBreakevenFloorRoiPct=0.5,
        runnerContext=main.V3_RUNNER_CONTEXT_TREND,
    )
    actions = {"time_closed": [], "partial_tp": []}

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 99.96, actions))

    assert handled is True
    assert actions["time_closed"] == ["FLATUSDT sideways_reclaim"]
    assert trader.closed[0][2] == "SIDEWAYS_RECLAIM_CLOSE"
    assert trader.pipeline_metrics["v3_sideways_reclaim_count"] == 1
