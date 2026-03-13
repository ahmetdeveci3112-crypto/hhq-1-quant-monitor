import asyncio
import time

import main


def _build_reclaim_pos(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "POS_FAKEOUT_OG",
        "signalId": "SIG_FAKEOUT_OG",
        "symbol": "OGUSDT",
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
        "openTime": now_ms - (18 * 60 * 1000),
        "atr": 1.0,
        "stopLoss": 98.8,
        "takeProfit": 103.0,
        "trailActivation": 101.5,
        "trailDistance": 0.4,
        "runtimeBreakevenFloorRoiPct": 0.4,
        "entryArchetype": main.ENTRY_ARCHETYPE_RECLAIM,
        "runnerContextResolved": main.V3_RUNNER_CONTEXT_COUNTER,
        "runnerContext": main.V3_RUNNER_CONTEXT_COUNTER,
        "baseRunnerContext": main.V3_RUNNER_CONTEXT_COUNTER,
        "positionThesisState": main.POSITION_THESIS_ENTRY,
        "continuationFlowState": main.V3_CONTINUATION_CHOP,
        "underwaterTapeState": main.V3_UNDERWATER_SIDEWAYS,
        "signal_snapshot": {"decisionContext": {"directionOwner": "reclaim"}},
        "expectancy": {"expectancyBand": main.DECISION_EXPECTANCY_BAND_NEUTRAL, "rankingScore": 101.0},
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.6, "roi_pct": 6.0, "close_pct": 0.15},
            {"key": "tp2", "price_pct": 1.0, "roi_pct": 10.0, "close_pct": 0.2},
            {"key": "tp3", "price_pct": 1.4, "roi_pct": 14.0, "close_pct": 0.25},
            {"key": "tp_final", "price_pct": 1.9, "roi_pct": 19.0, "close_pct": 0.4},
        ],
        "partial_tp_state": {},
        "currentVolumeRatio": 0.9,
        "currentObImbalanceTrend": 1.0,
        "currentImbalance": 0.2,
        "currentExecScore": 69.0,
    }
    pos.update(overrides)
    return pos


def _underwater_ctx(**overrides):
    ctx = {
        "state": main.V3_UNDERWATER_SIDEWAYS,
        "dynamic_be_price": 99.96,
        "dynamic_be_buffer_ratio": 0.0004,
        "hostile_imbalance": False,
        "opposite_signal_persistent": False,
        "adverse_flow": False,
        "recovered_from_worst": True,
    }
    ctx.update(overrides)
    return ctx


def _thesis_ctx(**overrides):
    ctx = {
        "thesis_state": main.POSITION_THESIS_ENTRY,
        "armed": False,
        "fire_rescue": False,
        "fire_profit_hold": False,
    }
    ctx.update(overrides)
    return ctx


def test_fakeout_reclaim_guard_arms_hold_for_reclaim_fakeout(monkeypatch):
    pos = _build_reclaim_pos()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["fire_hold"] is True
    assert result["block_close"] is True
    assert pos["fakeoutReclaimHoldArmed"] is True
    assert pos["fakeoutReclaimHoldUsed"] is True
    assert pos["fakeoutReclaimReason"] == "SIDEWAYS_RECLAIM_CLOSE_FAKEOUT_RECHECK"


def test_fakeout_reclaim_guard_skips_hostile_flow(monkeypatch):
    pos = _build_reclaim_pos()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(hostile_imbalance=True),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["fire_hold"] is False
    assert result["block_close"] is False
    assert result["reason"] == "HOSTILE_FLOW"
    assert pos["fakeoutReclaimHoldArmed"] is False


def test_fakeout_reclaim_guard_skips_on_structure_failure(monkeypatch):
    pos = _build_reclaim_pos(
        structureTrend="DOWN",
        breakoutRetestState="FAILED_BREAKDOWN",
        patternBias="NEUTRAL",
        patternConfidence=0.82,
    )
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["fire_hold"] is False
    assert result["block_close"] is False
    assert result["reason"] == "STRUCTURE_FAILURE"


def test_fakeout_reclaim_guard_keeps_hold_when_structure_supports_continuation(monkeypatch):
    pos = _build_reclaim_pos(
        structureTrend="UP",
        breakoutRetestState="BULL_RETEST_HOLD",
        patternBias="CONTINUATION",
        patternConfidence=0.84,
    )
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["fire_hold"] is True
    assert result["block_close"] is True
    assert pos["fakeoutReclaimHoldArmed"] is True


def test_fakeout_reclaim_guard_recovers_after_persistence(monkeypatch):
    pos = _build_reclaim_pos()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    pos["continuationFlowState"] = main.V3_CONTINUATION_SUPPORTING
    monkeypatch.setattr(main.time, "time", lambda: 1001.0)
    interim = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=100.04,
        current_roi_pct=0.4,
        underwater_ctx=_underwater_ctx(state=main.V3_UNDERWATER_RECOVERING),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )
    assert interim["block_close"] is True

    monkeypatch.setattr(main.time, "time", lambda: 1032.0)
    recovered = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=100.05,
        current_roi_pct=0.5,
        underwater_ctx=_underwater_ctx(state=main.V3_UNDERWATER_RECOVERING),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert recovered["recovered"] is True
    assert recovered["block_close"] is True
    assert pos["fakeoutReclaimHoldArmed"] is False
    assert pos["fakeoutReclaimReleaseReason"] == "RECOVERED"


def test_fakeout_reclaim_guard_times_out_without_recovery(monkeypatch):
    pos = _build_reclaim_pos()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    monkeypatch.setattr(main.time, "time", lambda: 1121.0)
    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.94,
        current_roi_pct=-0.4,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["block_close"] is False
    assert result["release_reason"] == "TIMEOUT"
    assert pos["fakeoutReclaimHoldArmed"] is False
    assert pos["fakeoutReclaimReleaseReason"] == "TIMEOUT"


def test_fakeout_reclaim_guard_invalidates_on_roi_deterioration(monkeypatch):
    pos = _build_reclaim_pos()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    monkeypatch.setattr(main.time, "time", lambda: 1010.0)
    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.70,
        current_roi_pct=-1.6,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["release_reason"] == "INVALIDATED"
    assert pos["fakeoutReclaimHoldArmed"] is False
    assert pos["fakeoutReclaimReleaseReason"] == "INVALIDATED"


def test_fakeout_reclaim_guard_does_not_rearm_after_single_use(monkeypatch):
    pos = _build_reclaim_pos(fakeoutReclaimHoldUsed=True)
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)

    result = main.evaluate_v3_fakeout_reclaim_guard(
        pos,
        current_price=99.95,
        current_roi_pct=-0.5,
        underwater_ctx=_underwater_ctx(),
        thesis_ctx=_thesis_ctx(),
        close_reason="SIDEWAYS_RECLAIM_CLOSE",
    )

    assert result["fire_hold"] is False
    assert result["reason"] == "USED_ALREADY"


def test_v3_runner_position_arms_fakeout_hold_before_sideways_reclaim_close(monkeypatch):
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
    monkeypatch.setattr(main, "build_position_profit_ladder", lambda pos, current_price=None: {
        "current_roi_pct": -0.5,
        "breakeven_floor_roi_pct": 0.4,
        "continuation_flow_state": main.V3_CONTINUATION_CHOP,
    })
    monkeypatch.setattr(main, "evaluate_v3_underwater_tape_state", lambda *args, **kwargs: _underwater_ctx())
    monkeypatch.setattr(main, "evaluate_v3_position_thesis_revalidation", lambda *args, **kwargs: _thesis_ctx())
    monkeypatch.setattr(main, "evaluate_v3_aged_profit_guard", lambda *args, **kwargs: {
        "watching": False,
        "should_arm_be_floor": False,
        "reason": "",
        "candidate_floor_price": 0.0,
    })
    trader = DummyTrader()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    pos = _build_reclaim_pos(symbol="OGUSDT", openTime=int((1000.0 - 18 * 60) * 1000))
    actions = {"time_closed": [], "partial_tp": []}

    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 99.95, actions))

    assert handled is True
    assert trader.closed == []
    assert pos["fakeoutReclaimHoldArmed"] is True
    assert pos["lastExitDecision"] == "FAKEOUT_RECLAIM_HOLD"


def test_v3_runner_position_closes_after_fakeout_hold_timeout(monkeypatch):
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
    monkeypatch.setattr(main, "build_position_profit_ladder", lambda pos, current_price=None: {
        "current_roi_pct": -0.5,
        "breakeven_floor_roi_pct": 0.4,
        "continuation_flow_state": main.V3_CONTINUATION_CHOP,
    })
    monkeypatch.setattr(main, "evaluate_v3_underwater_tape_state", lambda *args, **kwargs: _underwater_ctx())
    monkeypatch.setattr(main, "evaluate_v3_position_thesis_revalidation", lambda *args, **kwargs: _thesis_ctx())
    monkeypatch.setattr(main, "evaluate_v3_aged_profit_guard", lambda *args, **kwargs: {
        "watching": False,
        "should_arm_be_floor": False,
        "reason": "",
        "candidate_floor_price": 0.0,
    })

    trader = DummyTrader()
    pos = _build_reclaim_pos(symbol="OGUSDT", openTime=int((1000.0 - 18 * 60) * 1000))
    actions = {"time_closed": [], "partial_tp": []}

    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 99.95, actions))
    assert pos["fakeoutReclaimHoldArmed"] is True

    monkeypatch.setattr(main.time, "time", lambda: 1121.0)
    handled = asyncio.run(time_manager._handle_v3_runner_position(trader, pos, 99.94, actions))

    assert handled is True
    assert trader.closed
    assert trader.closed[0][2] == "SIDEWAYS_RECLAIM_CLOSE"
    assert pos["fakeoutReclaimReleaseReason"] == "TIMEOUT"
