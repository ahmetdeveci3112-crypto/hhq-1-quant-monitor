import asyncio
import time

import main


def _build_v3_position(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "POS_AGED_GUARD",
        "symbol": "TESTUSDT",
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "side": "LONG",
        "entryPrice": 100.0,
        "exchangeBreakEvenPrice": 100.02,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "marginUsd": 10.0,
        "initialMargin": 10.0,
        "leverage": 10,
        "openTime": now_ms - (5 * 60 * 60 * 1000),
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
        "atr": 0.1,
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
        "agedProfitPositiveSinceTs": time.time() - (4.5 * 3600),
        "agedProfitNonSupportingSinceTs": time.time() - (20 * 60),
    }
    pos.update(overrides)
    return pos


def _profit_ladder(state=main.V3_CONTINUATION_NEUTRAL, **overrides):
    ladder = {
        "current_roi_pct": 2.2,
        "breakeven_floor_roi_pct": 0.4,
        "profit_giveback_roi_pct": 0.0,
        "profit_giveback_arm_roi_pct": 1.4,
        "profit_peak_roi_pct": 2.5,
        "giveback_trail_ready": False,
        "continuation_flow_state": state,
        "continuation_flow": {
            "state": state,
            "current_volume_ratio": 1.0,
            "current_ob_imbalance_trend": 0.0,
            "current_exec_score": 72.0,
        },
    }
    ladder.update(overrides)
    return ladder


def _underwater_ctx(state=main.V3_UNDERWATER_NEUTRAL, **overrides):
    ctx = {
        "state": state,
        "hostile_imbalance": False,
        "opposite_signal_persistent": False,
        "adverse_flow": False,
        "dynamic_be_price": 100.03,
        "dynamic_be_buffer_ratio": 0.0004,
    }
    ctx.update(overrides)
    return ctx


def test_evaluate_v3_aged_profit_guard_watches_supporting_flow_without_floor():
    pos = _build_v3_position(
        agedProfitPositiveSinceTs=time.time() - (5 * 3600),
        agedProfitNonSupportingSinceTs=time.time() - (20 * 60),
    )

    result = main.evaluate_v3_aged_profit_guard(
        pos,
        current_price=100.25,
        profit_ladder=_profit_ladder(state=main.V3_CONTINUATION_SUPPORTING),
        underwater_ctx=_underwater_ctx(),
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert result["watching"] is True
    assert result["should_arm_be_floor"] is False
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_WATCHING
    assert pos["agedProfitNonSupportingSinceTs"] == 0.0


def test_evaluate_v3_aged_profit_guard_arms_be_floor_after_aged_non_supporting_profit():
    pos = _build_v3_position()

    result = main.evaluate_v3_aged_profit_guard(
        pos,
        current_price=100.25,
        profit_ladder=_profit_ladder(state=main.V3_CONTINUATION_CHOP),
        underwater_ctx=_underwater_ctx(),
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert result["watching"] is True
    assert result["should_arm_be_floor"] is True
    assert result["candidate_floor_price"] > 100.0
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_NON_SUPPORTING


def test_evaluate_v3_aged_profit_guard_rejects_floor_when_gap_is_too_tight():
    pos = _build_v3_position(exchangeBreakEvenPrice=100.22)

    result = main.evaluate_v3_aged_profit_guard(
        pos,
        current_price=100.25,
        profit_ladder=_profit_ladder(state=main.V3_CONTINUATION_NEUTRAL),
        underwater_ctx=_underwater_ctx(),
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert result["should_arm_be_floor"] is False
    assert result["reason"] == "AGED_PROFIT_INSUFFICIENT_GAP"
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_NON_SUPPORTING


def test_evaluate_v3_aged_profit_guard_resets_when_roi_turns_non_positive():
    pos = _build_v3_position(
        agedProfitGuardState=main.AGED_PROFIT_GUARD_STATE_NON_SUPPORTING,
        agedProfitGuardReason="NON_SUPPORTING_FLOW",
    )

    result = main.evaluate_v3_aged_profit_guard(
        pos,
        current_price=99.95,
        profit_ladder=_profit_ladder(state=main.V3_CONTINUATION_NEUTRAL, current_roi_pct=-0.2),
        underwater_ctx=_underwater_ctx(),
        thesis_ctx={"thesis_state": main.POSITION_THESIS_ENTRY},
    )

    assert result["should_arm_be_floor"] is False
    assert pos["agedProfitPositiveSinceTs"] == 0.0
    assert pos["agedProfitNonSupportingSinceTs"] == 0.0
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_IDLE


def test_evaluate_v3_aged_profit_guard_skips_when_profit_hold_active():
    pos = _build_v3_position(positionThesisState=main.POSITION_THESIS_PROFIT_CONTINUATION_HOLD)

    result = main.evaluate_v3_aged_profit_guard(
        pos,
        current_price=100.25,
        profit_ladder=_profit_ladder(state=main.V3_CONTINUATION_CHOP),
        underwater_ctx=_underwater_ctx(),
        thesis_ctx={"thesis_state": main.POSITION_THESIS_PROFIT_CONTINUATION_HOLD},
    )

    assert result["should_arm_be_floor"] is False
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_WATCHING
    assert pos["agedProfitGuardReason"] == "THESIS_SUPPORT"


def test_v3_runner_position_arms_be_floor_for_aged_non_supporting_profit(monkeypatch):
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
    monkeypatch.setattr(main, "build_position_profit_ladder", lambda pos, current_price=None: _profit_ladder(state=main.V3_CONTINUATION_CHOP))
    monkeypatch.setattr(main, "evaluate_v3_underwater_tape_state", lambda *args, **kwargs: _underwater_ctx())

    trader = DummyTrader()
    pos = _build_v3_position(symbol="AGEDUSDT")
    actions = {"time_closed": [], "partial_tp": []}
    expected_floor = main.compute_buffered_breakeven_price(pos, reason="AGED_PROFIT_NO_SUPPORT")["price"]

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 100.25, actions))

    assert handled is True
    assert trader.closed == []
    assert pos["agedProfitGuardState"] == main.AGED_PROFIT_GUARD_STATE_BE_FLOOR_ARMED
    assert pos["agedProfitGuardReason"] == "AGED_PROFIT_NO_SUPPORT"
    assert pos["breakeven_activated"] is True
    assert pos["stopLoss"] >= expected_floor > 0
