import asyncio
import time

import main


def _build_v3_position(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "POS_THESIS",
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
        "runtimeBreakevenFloorRoiPct": 0.4,
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
        "runnerContext": main.V3_RUNNER_CONTEXT_COUNTER,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "baseRunnerContext": main.V3_RUNNER_CONTEXT_COUNTER,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.4, "roi_pct": 4.0, "close_pct": 0.1},
            {"key": "tp2", "price_pct": 0.8, "roi_pct": 8.0, "close_pct": 0.15},
            {"key": "tp3", "price_pct": 1.2, "roi_pct": 12.0, "close_pct": 0.25},
            {"key": "tp_final", "price_pct": 1.8, "roi_pct": 18.0, "close_pct": 0.5},
        ],
        "partial_tp_state": {},
        "profitPeakRoiPct": 0.0,
        "runtimeProfitPeakRoiPct": 0.0,
        "isLive": False,
    }
    pos.update(overrides)
    return pos


def _supporting_profit_ladder(**overrides):
    ladder = {
        "breakeven_floor_roi_pct": 0.4,
        "profit_giveback_roi_pct": 0.0,
        "profit_giveback_arm_roi_pct": 1.4,
        "profit_peak_roi_pct": 2.5,
        "giveback_trail_ready": False,
        "continuation_flow_state": main.V3_CONTINUATION_SUPPORTING,
        "continuation_flow": {
            "state": main.V3_CONTINUATION_SUPPORTING,
            "current_volume_ratio": 1.22,
            "current_ob_imbalance_trend": 2.6,
            "current_exec_score": 78.0,
        },
    }
    ladder.update(overrides)
    return ladder


def _sideways_underwater_ctx(**overrides):
    ctx = {
        "state": main.V3_UNDERWATER_SIDEWAYS,
        "hostile_imbalance": False,
        "opposite_signal_persistent": False,
        "adverse_flow": False,
        "dynamic_be_price": 99.96,
        "dynamic_be_buffer_ratio": 0.0004,
        "worst_underwater_price": 99.4,
    }
    ctx.update(overrides)
    return ctx


def test_evaluate_v3_position_thesis_revalidation_fires_rescue_for_supporting_reclaim_recovery():
    now_ts = time.time()
    pos = _build_v3_position(
        currentVolumeRatio=1.24,
        currentObImbalanceTrend=2.7,
        currentExecScore=81.0,
        sidewaysSinceTs=now_ts - 500.0,
        worstUnderwaterTs=now_ts - 700.0,
        worstUnderwaterPrice=99.3,
        _reclaimRescueCandidateSinceTs=now_ts - 61.0,
    )

    result = main.evaluate_v3_position_thesis_revalidation(
        pos,
        current_price=99.96,
        current_roi_pct=-0.4,
        profit_ladder=_supporting_profit_ladder(),
        underwater_ctx=_sideways_underwater_ctx(),
    )

    assert result["fire_rescue"] is True
    assert result["thesis_state"] == main.POSITION_THESIS_CONTINUATION_RESCUE
    assert pos["positionThesisState"] == main.POSITION_THESIS_CONTINUATION_RESCUE
    assert pos["runnerContextResolved"] == main.V3_RUNNER_CONTEXT_CONTINUATION_RESCUE


def test_evaluate_v3_position_thesis_revalidation_rejects_rescue_when_flow_not_supporting():
    now_ts = time.time()
    pos = _build_v3_position(
        currentVolumeRatio=0.92,
        currentObImbalanceTrend=0.3,
        currentExecScore=72.0,
        sidewaysSinceTs=now_ts - 500.0,
        worstUnderwaterTs=now_ts - 700.0,
        worstUnderwaterPrice=99.3,
        _reclaimRescueCandidateSinceTs=now_ts - 61.0,
    )

    result = main.evaluate_v3_position_thesis_revalidation(
        pos,
        current_price=99.96,
        current_roi_pct=-0.4,
        profit_ladder=_supporting_profit_ladder(
            continuation_flow_state=main.V3_CONTINUATION_CHOP,
            continuation_flow={
                "state": main.V3_CONTINUATION_CHOP,
                "current_volume_ratio": 0.92,
                "current_ob_imbalance_trend": 0.3,
                "current_exec_score": 72.0,
            },
        ),
        underwater_ctx=_sideways_underwater_ctx(),
    )

    assert result["fire_rescue"] is False
    assert pos["positionThesisState"] == main.POSITION_THESIS_ENTRY


def test_evaluate_v3_position_thesis_revalidation_fires_profit_hold_on_supporting_giveback():
    now_ts = time.time()
    pos = _build_v3_position(
        currentVolumeRatio=1.18,
        currentObImbalanceTrend=2.4,
        currentExecScore=80.0,
        _profitContinuationHoldCandidateSinceTs=now_ts - 46.0,
    )

    result = main.evaluate_v3_position_thesis_revalidation(
        pos,
        current_price=100.35,
        current_roi_pct=3.5,
        profit_ladder=_supporting_profit_ladder(
            profit_giveback_roi_pct=1.8,
            profit_giveback_arm_roi_pct=1.2,
            giveback_trail_ready=True,
        ),
        underwater_ctx=_sideways_underwater_ctx(state=main.V3_UNDERWATER_NEUTRAL),
    )

    assert result["fire_profit_hold"] is True
    assert result["thesis_state"] == main.POSITION_THESIS_PROFIT_CONTINUATION_HOLD
    assert pos["positionThesisState"] == main.POSITION_THESIS_PROFIT_CONTINUATION_HOLD


def test_evaluate_v3_position_thesis_revalidation_rejects_profit_hold_when_flow_fading():
    now_ts = time.time()
    pos = _build_v3_position(
        currentVolumeRatio=0.94,
        currentObImbalanceTrend=0.4,
        currentExecScore=74.0,
        _profitContinuationHoldCandidateSinceTs=now_ts - 46.0,
    )

    result = main.evaluate_v3_position_thesis_revalidation(
        pos,
        current_price=100.35,
        current_roi_pct=3.5,
        profit_ladder=_supporting_profit_ladder(
            profit_giveback_roi_pct=1.8,
            profit_giveback_arm_roi_pct=1.2,
            giveback_trail_ready=True,
            continuation_flow_state=main.V3_CONTINUATION_FADING,
            continuation_flow={
                "state": main.V3_CONTINUATION_FADING,
                "current_volume_ratio": 0.94,
                "current_ob_imbalance_trend": 0.4,
                "current_exec_score": 74.0,
            },
        ),
        underwater_ctx=_sideways_underwater_ctx(state=main.V3_UNDERWATER_NEUTRAL),
    )

    assert result["fire_profit_hold"] is False
    assert pos["positionThesisState"] == main.POSITION_THESIS_ENTRY


def test_v3_runner_sideways_reclaim_close_defers_when_rescue_fires(monkeypatch):
    class DummyTrader:
        def __init__(self):
            self.closed = []
            self.pipeline_metrics = {}

        def close_via_engine(self, pos, exit_price, reason, source):
            self.closed.append((pos["symbol"], exit_price, reason, source))
            return {"reason": reason}

    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    time_manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(time_manager, "_get_v3_opposite_pressure", lambda pos: (False, 0.0))
    monkeypatch.setattr(time_manager, "_select_triggered_tp_slice_trail", lambda pos, price: None)
    trader = DummyTrader()
    now_ts = time.time()

    pos = _build_v3_position(
        symbol="RESCUEUSDT",
        currentPrice=99.96,
        currentVolumeRatio=1.26,
        currentIsVolumeSpike=False,
        currentImbalance=1.0,
        currentObImbalanceTrend=2.8,
        currentExecScore=82.0,
        currentAtrPct=0.8,
        openTime=int((now_ts - 30 * 60) * 1000),
        sidewaysSinceTs=now_ts - 800.0,
        worstUnderwaterPrice=99.4,
        worstUnderwaterTs=now_ts - 900.0,
        lastSupportiveFlowTs=0,
        _reclaimRescueCandidateSinceTs=now_ts - 61.0,
    )
    actions = {"time_closed": [], "partial_tp": []}

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 99.96, actions))

    assert handled is True
    assert trader.closed == []
    assert actions["time_closed"] == []
    assert pos["lastExitDecision"] == "CONTINUATION_RESCUE_HOLD"
    assert pos["positionThesisState"] == main.POSITION_THESIS_CONTINUATION_RESCUE
    assert pos["runnerContextResolved"] == main.V3_RUNNER_CONTEXT_CONTINUATION_RESCUE


def test_maybe_defer_v3_profit_exit_arms_hold_and_resets_breach_counters(monkeypatch):
    monkeypatch.setattr(main, "queue_decision_snapshot", lambda **kwargs: None)
    now_ts = time.time()
    pos = _build_v3_position(
        symbol="HOLDUSDT",
        currentPrice=100.35,
        currentVolumeRatio=1.18,
        currentIsVolumeSpike=False,
        currentImbalance=1.0,
        currentObImbalanceTrend=2.5,
        currentExecScore=81.0,
        runtimeProfitGivebackRoiPct=1.8,
        runtimeProfitGivebackArmRoiPct=1.2,
        trailBreachCount=4,
        trailBreachStartTime=now_ts - 20.0,
        slConfirmCount=3,
        slBreachStartTime=now_ts - 12.0,
        _profitContinuationHoldCandidateSinceTs=now_ts - 46.0,
    )

    deferred = main.maybe_defer_v3_profit_exit(
        pos,
        current_price=100.35,
        profit_ladder=_supporting_profit_ladder(
            profit_giveback_roi_pct=1.8,
            profit_giveback_arm_roi_pct=1.2,
            giveback_trail_ready=True,
        ),
        note="test_profit_hold",
    )

    assert deferred is True
    assert pos["lastExitDecision"] == "PROFIT_CONTINUATION_HOLD"
    assert pos["positionThesisState"] == main.POSITION_THESIS_PROFIT_CONTINUATION_HOLD
    assert pos["trailBreachCount"] == 0
    assert pos["slConfirmCount"] == 0
