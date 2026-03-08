import asyncio
import time
from datetime import datetime
from types import SimpleNamespace

import main
from main import (
    PaperTradingEngine,
    PositionBasedKillSwitch,
    TimeBasedPositionManager,
    _build_ui_signal_stats,
    _resolve_live_positions_for_ui,
    build_position_profit_ladder,
    build_position_protection_ladder,
    classify_v3_runner_context,
    compute_adaptive_tp_ladder,
    compute_buffered_breakeven_price,
    compute_target_roi_pct,
    derive_profit_exit_reason,
    price_distance_to_roi_pct,
    price_move_pct_to_roi_pct,
    should_activate_trailing_by_roi,
    update_runtime_protection_telemetry,
    update_runtime_trail_telemetry,
    uses_profit_ladder_authority,
)


def _make_entry_engine(monkeypatch, stop_loss: float):
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine.pending_orders = []
    engine.positions = []
    engine.trades = []
    engine.balance = 1000.0
    engine.exit_tightness = 1.2
    engine.sl_atr = 18
    engine.tp_atr = 20
    engine.trail_activation_atr = 1.5
    engine.trail_distance_atr = 1.5
    engine.min_confidence_score = 74
    engine.strategy_mode = main.STRATEGY_MODE_LEGACY
    engine.current_spread_pct = 0.05
    engine.pipeline_metrics = {
        "open_rejected": 0,
        "open_reject_reasons": {},
        "filled": 0,
        "order_attempted": 0,
        "order_success": 0,
    }
    engine.calculate_dynamic_atr_multiplier = lambda atr, price: 1.0
    engine.add_log = lambda _msg: None
    engine.save_state = lambda: None
    engine._finalize_forecast_event = lambda *args, **kwargs: None
    engine.set_execution_feedback = lambda symbol, reason: setattr(engine, "_last_exec_feedback", (symbol, reason))
    engine.clear_execution_feedback = lambda _symbol: None

    async def _noop_async(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        main,
        "live_binance_trader",
        SimpleNamespace(enabled=False, trading_mode="paper", exec_entry_score_min=0.10, last_order_error=None),
    )
    monkeypatch.setattr(main, "TP_LADDER_ADAPTIVE_ENABLED", False)
    monkeypatch.setattr(main, "RFX_EXIT_SM_ENABLED", False)
    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: coro.close() if hasattr(coro, "close") else None)
    monkeypatch.setattr(main.sqlite_manager, "save_open_position", _noop_async)
    monkeypatch.setattr(main.sqlite_manager, "update_signal_outcome", _noop_async)
    monkeypatch.setattr(
        main,
        "resolve_effective_atr_value",
        lambda *_args, **_kwargs: {"atr": 1.0, "source": "test", "fallback_used": False, "floor_pct": 0.0},
    )
    monkeypatch.setattr(
        main,
        "resolve_runner_exit_controls",
        lambda payload, default_mode: {
            "mode": default_mode,
            "trail_act_mult": 1.0,
            "trail_dist_mult": 1.0,
            "tp_tighten": 1.0,
            "be_buffer_mult": 1.0,
            "enabled": False,
        },
    )
    monkeypatch.setattr(
        main,
        "resolve_effective_trail_atr_bases",
        lambda configured_activation_atr, configured_distance_atr, dynamic_activation, dynamic_distance: {
            "base_activation_atr": configured_activation_atr,
            "base_distance_atr": configured_distance_atr,
            "configured_activation_atr": configured_activation_atr,
            "configured_distance_atr": configured_distance_atr,
            "signal_activation_atr": dynamic_activation or configured_activation_atr,
            "signal_distance_atr": dynamic_distance or configured_distance_atr,
        },
    )
    monkeypatch.setattr(
        main,
        "apply_user_configured_exit_atr_floors",
        lambda **kwargs: {
            "sl_atr": kwargs["adjusted_sl_atr"],
            "tp_atr": kwargs["adjusted_tp_atr"],
            "trail_activation_atr": kwargs["adjusted_trail_activation_atr"],
            "trail_distance_atr": kwargs["adjusted_trail_distance_atr"],
            "raw_sl_atr": kwargs["adjusted_sl_atr"],
            "raw_tp_atr": kwargs["adjusted_tp_atr"],
            "raw_trail_activation_atr": kwargs["adjusted_trail_activation_atr"],
            "raw_trail_distance_atr": kwargs["adjusted_trail_distance_atr"],
            "floor_applied": False,
        },
    )
    monkeypatch.setattr(
        main,
        "compute_sl_tp_levels",
        lambda **kwargs: {
            "sl": stop_loss,
            "tp": 110.0,
            "trail_activation": 103.0,
            "trail_distance": 2.0,
            "distance_truth": {},
            "meta": {"sl_source": "test", "tp_source": "test"},
        },
    )
    return engine


def _make_entry_order(signal_score: float = 90.0, exec_score: float = 0.20) -> dict:
    return {
        "id": "PO_TEST",
        "symbol": "TESTUSDT",
        "side": "LONG",
        "createdAt": int(time.time() * 1000),
        "signalPrice": 100.0,
        "entryPrice": 100.0,
        "atr": 1.0,
        "leverage": 10,
        "size": 1.0,
        "sizeUsd": 100.0,
        "marginUsd": 10.0,
        "spreadPct": 0.05,
        "spreadLevel": "LOW",
        "signalScore": signal_score,
        "signalScoreRaw": signal_score,
        "entrySignalScoreSnapshot": signal_score,
        "minConfidenceScoreSnapshot": 74.0,
        "entryExecScore": exec_score,
        "entryExecScoreMinSnapshot": 0.10,
        "entryStopSoftRoiPct": -200.0,
        "entryStopHardRoiPct": -250.0,
        "signal_snapshot": {},
        "exec_snapshot": {},
        "risk_snapshot": {},
        "truth_snapshot": {},
        "execution_profile_source": "neutral",
    }


def test_roi_helpers_and_runtime_trail_telemetry():
    assert price_move_pct_to_roi_pct(3.0, 10) == 30.0
    assert round(price_distance_to_roi_pct(0.5, 10.0, 8), 2) == 40.0
    assert round(compute_target_roi_pct(100.0, 110.0, "LONG", 5), 2) == 50.0
    assert round(compute_target_roi_pct(100.0, 90.0, "SHORT", 5), 2) == 50.0
    assert should_activate_trailing_by_roi(12.0, 10.0) is True
    assert should_activate_trailing_by_roi(9.9, 10.0) is False

    pos = {"leverage": 12}
    update_runtime_trail_telemetry(
        pos=pos,
        dynamic_trail_distance=0.02,
        effective_exit_tightness=1.4,
        min_price_move_pct=0.8,
        min_roi_pct=9.6,
        threshold_mult=1.2,
        entry_price=0.05,
    )

    assert pos["runtimeTrailDistancePct"] == 40.0
    assert pos["runtimeTrailDistanceRoiPct"] == 480.0
    assert pos["runtimeTrailActivationRoiPct"] == 9.6


def test_kill_switch_thresholds_are_canonical_roi_values():
    ks = PositionBasedKillSwitch()
    ks.first_reduction_pct = -90
    ks.full_close_pct = -140

    assert ks.get_dynamic_thresholds(5) == (-90.0, -140.0)
    assert ks.get_dynamic_thresholds(25) == (-90.0, -140.0)


def test_ui_signal_stats_keep_current_and_persistent_counts_separate():
    opportunities = [
        {"symbol": "AAAUSDT", "signalAction": "LONG"},
        {"symbol": "BBBUSDT", "signalAction": "SHORT"},
        {"symbol": "CCCUSDT", "signalAction": "NONE"},
    ]
    persistent = [
        {"symbol": "AAAUSDT", "signalAction": "LONG"},
        {"symbol": "DDDUSDT", "signalAction": "LONG"},
        {"symbol": "EEEUSDT", "signalAction": "SHORT"},
    ]

    stats = _build_ui_signal_stats(opportunities, persistent)

    assert stats["longSignals"] == 1
    assert stats["shortSignals"] == 1
    assert stats["activeSignals"] == 2
    assert stats["persistentLongSignals"] == 2
    assert stats["persistentShortSignals"] == 1
    assert stats["persistentActiveSignals"] == 3


def test_live_ui_positions_prefer_empty_binance_snapshot_over_engine_fallback(monkeypatch):
    monkeypatch.setattr(
        main,
        "live_binance_trader",
        SimpleNamespace(enabled=True, last_positions=[], last_sync_time=123456),
    )
    monkeypatch.setattr(
        main,
        "global_paper_trader",
        SimpleNamespace(positions=[{"symbol": "STALEUSDT", "isLive": True}]),
    )

    positions, source = _resolve_live_positions_for_ui()

    assert positions == []
    assert source == "binance_sync_empty"


def test_protection_ladder_tight_and_wide_stop_behavior():
    tight_pos = {
        "entryPrice": 100.0,
        "currentPrice": 99.0,
        "side": "LONG",
        "leverage": 10,
        "stopLoss": 98.0,
        "takeProfit": 104.0,
        "trailActivation": 101.0,
        "trailDistance": 1.0,
        "unrealizedPnlPercent": -10.0,
    }
    tight = build_position_protection_ladder(tight_pos, -100.0, -150.0, current_price=99.0)
    assert tight["effective_stop_roi_pct"] == -20.0
    assert tight["pre_stop_reduce_roi_pct"] == -12.0
    assert tight["kill_switch_full_roi_pct"] is None

    wide_pos = {
        "entryPrice": 100.0,
        "currentPrice": 92.0,
        "side": "LONG",
        "leverage": 10,
        "stopLoss": 70.0,
        "takeProfit": 106.0,
        "trailActivation": 101.0,
        "trailDistance": 1.0,
        "unrealizedPnlPercent": -80.0,
    }
    wide = build_position_protection_ladder(wide_pos, -100.0, -150.0, current_price=92.0)
    assert wide["effective_stop_roi_pct"] == -300.0
    assert wide["pre_stop_reduce_roi_pct"] == -100.0
    assert wide["kill_switch_full_roi_pct"] == -150.0


def test_entry_stop_gate_normal_soft_and_hard_paths(monkeypatch):
    engine_normal = _make_entry_engine(monkeypatch, stop_loss=81.0)
    order_normal = _make_entry_order()
    engine_normal.pending_orders = [order_normal]
    asyncio.run(engine_normal.execute_pending_order(order_normal, 100.0))
    assert len(engine_normal.positions) == 1
    assert engine_normal.positions[0]["entryStopGateMode"] == "normal"
    assert engine_normal.positions[0]["sizeUsd"] == 100.0

    engine_soft = _make_entry_engine(monkeypatch, stop_loss=78.0)
    order_soft = _make_entry_order()
    engine_soft.pending_orders = [order_soft]
    asyncio.run(engine_soft.execute_pending_order(order_soft, 100.0))
    assert len(engine_soft.positions) == 1
    soft_pos = engine_soft.positions[0]
    assert soft_pos["entryStopGateMode"] == "wide_stop_soft"
    assert soft_pos["runtimeEntryStopGateMode"] == "wide_stop_soft"
    assert soft_pos["entryStopRoiPct"] == -220.0
    assert round(soft_pos["size"], 4) == 0.75
    assert round(soft_pos["sizeUsd"], 4) == 75.0
    assert round(soft_pos["initialMargin"], 4) == 7.5
    assert soft_pos["minConfidenceScoreSnapshot"] == 80.0
    assert round(soft_pos["entryExecScoreMinSnapshot"], 2) == 0.15

    engine_hard = _make_entry_engine(monkeypatch, stop_loss=74.0)
    order_hard = _make_entry_order()
    engine_hard.pending_orders = [order_hard]
    asyncio.run(engine_hard.execute_pending_order(order_hard, 100.0))
    assert engine_hard.positions == []
    assert engine_hard.pipeline_metrics["open_rejected"] == 1
    assert engine_hard.pipeline_metrics["open_reject_reasons"]["ENTRY_STOP_TOO_WIDE"] == 1
    assert engine_hard._last_exec_feedback == ("TESTUSDT", "ENTRY_STOP_TOO_WIDE")


def test_runtime_protection_telemetry_populates_canonical_fields():
    pos = {
        "entryPrice": 100.0,
        "currentPrice": 95.0,
        "side": "LONG",
        "leverage": 10,
        "stopLoss": 90.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "unrealizedPnlPercent": -50.0,
    }
    ladder = update_runtime_protection_telemetry(pos, 95.0, -100.0, -150.0)
    assert ladder["target_tp_roi_pct"] == 100.0
    assert ladder["effective_stop_roi_pct"] == -100.0
    assert pos["runtimeTpRoiPct"] == 100.0
    assert pos["runtimeStopRoiPct"] == -100.0
    assert pos["runtimeProtectionPhase"] == "SL-PRIMARY"


def test_profit_ladder_uses_exchange_break_even_anchor_and_roi_phases():
    pos = {
        "symbol": "PROFITUSDT",
        "entryPrice": 100.0,
        "currentPrice": 112.0,
        "side": "LONG",
        "leverage": 10,
        "takeProfit": 118.0,
        "trailActivation": 104.0,
        "trailDistance": 1.5,
        "runtimeTrailDistanceRoiPct": 15.0,
        "runtimeTrailActivationRoiPct": 30.0,
        "exchangeBreakEvenPrice": 100.4,
        "spreadPct": 0.05,
        "spreadLevel": "Low",
        "partial_tp_state": {"tp1": True},
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.6, "roi_pct": 6.0, "close_pct": 0.2},
            {"key": "tp2", "price_pct": 1.2, "roi_pct": 12.0, "close_pct": 0.2},
            {"key": "tp3", "price_pct": 1.8, "roi_pct": 18.0, "close_pct": 0.2},
            {"key": "tp_final", "price_pct": 2.6, "roi_pct": 26.0, "close_pct": 0.4},
        ],
    }

    be_ctx = compute_buffered_breakeven_price(pos, buffer_pct=0.002)
    assert be_ctx["anchor_source"] == "exchange"
    assert be_ctx["anchor_price"] == 100.4
    assert be_ctx["price"] > 100.4

    ladder = build_position_profit_ladder(pos, current_price=112.0)
    assert ladder["exchange_break_even_price"] == 100.4
    assert ladder["breakeven_anchor_source"] == "exchange"
    assert ladder["tp1_roi_pct"] == 6.0
    assert ladder["tp2_roi_pct"] == 12.0
    assert ladder["profit_peak_roi_pct"] == 120.0
    assert ladder["profit_phase"] == "RUNNER"
    assert ladder["profit_owner"] in ("RUNNER", "BREAKEVEN")


def test_profit_exit_reason_tracks_owner_and_giveback():
    pos = {
        "entryPrice": 100.0,
        "currentPrice": 108.0,
        "side": "LONG",
        "leverage": 10,
        "takeProfit": 120.0,
        "trailActivation": 104.0,
        "trailDistance": 1.0,
        "runtimeTrailDistanceRoiPct": 10.0,
        "runtimeTrailActivationRoiPct": 20.0,
        "profitPeakRoiPct": 150.0,
        "partial_tp_state": {"tp1": True, "tp2": True},
        "isTrailingActive": True,
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.6, "roi_pct": 6.0, "close_pct": 0.2},
            {"key": "tp2", "price_pct": 1.2, "roi_pct": 12.0, "close_pct": 0.2},
            {"key": "tp3", "price_pct": 1.8, "roi_pct": 18.0, "close_pct": 0.2},
            {"key": "tp_final", "price_pct": 2.6, "roi_pct": 26.0, "close_pct": 0.4},
        ],
    }

    ladder = build_position_profit_ladder(pos, current_price=108.0)
    assert ladder["profit_phase"] == "RUNNER"
    assert derive_profit_exit_reason(pos, current_price=108.0, profit_ladder=ladder) == "PROFIT_GIVEBACK_EXIT"


def test_compute_adaptive_tp_ladder_builds_four_tier_runner_profile():
    ladder = compute_adaptive_tp_ladder(
        side="LONG",
        entry_price=100.0,
        atr=1.2,
        leverage=10,
        spread_pct=0.05,
        volume_ratio=1.4,
        adx=34.0,
        hurst=0.55,
        coin_daily_trend="BULLISH",
        exec_score=78.0,
        spread_level="Normal",
    )

    keys = [level["key"] for level in ladder["levels"]]
    close_pct_sum = sum(float(level["close_pct"]) for level in ladder["levels"])

    assert keys == ["tp1", "tp2", "tp3", "tp_final"]
    assert ladder["version"] == "v3_roi_profit_ladder_4tier"
    assert ladder["levels"][3]["roi_pct"] > ladder["levels"][2]["roi_pct"]
    assert round(close_pct_sum, 2) == 1.0
    assert "tp_final_target_roi" in ladder["telemetry"]


def test_profit_ladder_authority_detects_tp_levels():
    pos = {
        "entryPrice": 100.0,
        "side": "LONG",
        "leverage": 10,
        "tp_ladder_levels": [
            {"key": "tp1", "price_pct": 0.8, "close_pct": 0.2},
            {"key": "tp2", "price_pct": 1.6, "close_pct": 0.2},
            {"key": "tp3", "price_pct": 2.4, "close_pct": 0.2},
            {"key": "tp_final", "price_pct": 3.6, "close_pct": 0.4},
        ],
    }

    assert uses_profit_ladder_authority(pos) is True


def test_kill_switch_executes_pre_stop_reduction_and_recovery_close(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []

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

    ks = PositionBasedKillSwitch()
    pos = {
        "id": "pos-1",
        "symbol": "TESTUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 88.0,
        "size": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -12.0,
        "unrealizedPnlPercent": -120.0,
        "leverage": 10,
        "stopLoss": 80.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": 0,
        "isLive": False,
        "recoveryArmed": True,
        "recoveryWorstRoiPct": -120.0,
        "recoveryPeakRoiPct": -30.0,
        "recoveryTrailActive": True,
        "recoveryStage": 2,
        "preStopReduced": True,
    }
    trader = DummyTrader(pos)

    asyncio.run(ks.check_positions(trader))
    assert trader.closed[0][2] == "RECOVERY_TRAIL_CLOSE"

    pos2 = {
        "id": "pos-2",
        "symbol": "TEST2USDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 98.6,
        "size": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -1.4,
        "unrealizedPnlPercent": -14.0,
        "leverage": 10,
        "stopLoss": 98.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": 0,
        "isLive": False,
    }
    trader2 = DummyTrader(pos2)
    asyncio.run(ks.check_positions(trader2))
    assert pos2["preStopReduced"] is True
    assert trader2.closed == []


def test_time_recovery_reductions_arm_on_underwater_time_and_do_not_touch_normal_recovery(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []
            self.stats = {"totalTrades": 0, "winningTrades": 0}

        def add_log(self, _msg):
            return None

        def save_state(self):
            return None

        def _normalize_close_reason(self, reason):
            return reason

    async def _noop_save_trade(_trade):
        return None

    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: coro.close() if hasattr(coro, "close") else None)
    monkeypatch.setattr(main.sqlite_manager, "save_trade", _noop_save_trade)

    now_ms = 10 * 60 * 60 * 1000
    time_manager = TimeBasedPositionManager()
    pos = {
        "id": "time-pos-1",
        "symbol": "TIMEUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 92.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -8.0,
        "unrealizedPnlPercent": -80.0,
        "openTime": 0,
        "isLive": False,
        "leverage": 10,
        "timeUnderwaterSince": now_ms - int(5 * 60 * 60 * 1000),
        "recoveryArmed": False,
        "recoveryStage": 0,
    }
    trader = DummyTrader(pos)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(now_ms / 1000.0, tz)

    monkeypatch.setattr(main, "datetime", FakeDateTime)

    asyncio.run(time_manager.check_positions(trader))
    assert pos["timeRecovery4HArmed"] is True
    assert pos.get("timeRecovery4HDone", False) is False
    assert pos["recoveryArmed"] is False
    assert pos["recoveryStage"] == 0
    assert trader.trades == []

    pos["currentPrice"] = 95.0
    pos["unrealizedPnl"] = -5.0
    pos["unrealizedPnlPercent"] = -50.0
    asyncio.run(time_manager.check_positions(trader))

    assert pos["timeRecovery4HDone"] is True
    assert pos["timeRecovery4HArmed"] is False
    assert pos["recoveryArmed"] is False
    assert pos["recoveryStage"] == 0
    assert trader.trades[-1]["reason"] == "TIME_RECOVERY_STAGE1"
    assert round(pos["contracts"], 4) == 0.8

    pos["timeUnderwaterSince"] = now_ms - int(9 * 60 * 60 * 1000)
    pos["currentPrice"] = 90.0
    pos["unrealizedPnl"] = -7.2
    pos["unrealizedPnlPercent"] = -100.0
    asyncio.run(time_manager.check_positions(trader))
    assert pos["timeRecovery8HArmed"] is True
    assert pos.get("timeRecovery8HDone", False) is False

    pos["currentPrice"] = 94.0
    pos["unrealizedPnl"] = -4.8
    pos["unrealizedPnlPercent"] = -60.0
    asyncio.run(time_manager.check_positions(trader))

    assert pos["timeRecovery8HDone"] is True
    assert pos["timeRecovery8HArmed"] is False
    assert trader.trades[-1]["reason"] == "TIME_RECOVERY_STAGE2"
    assert round(pos["contracts"], 4) == 0.56


def test_legacy_underwater_exits_do_not_front_run_roi_ladder():
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine.max_position_age_hours = 4
    engine.get_effective_exit_tightness = lambda pos: 1.0
    engine.add_log = lambda msg: None
    engine.close_via_engine = lambda pos, price, reason, source: (_ for _ in ()).throw(AssertionError("legacy close should not fire"))

    underwater_pos = {
        "symbol": "OLDUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 92.0,
        "openTime": 0,
        "unrealizedPnlPercent": -80.0,
        "gradual_exit_mode": True,
    }

    assert engine.check_adverse_position_exit(underwater_pos, 92.0, atr=1.0) is False
    assert engine.check_time_based_exit(underwater_pos, 92.0, atr=1.0) is False
    assert underwater_pos["gradual_exit_mode"] is False


def test_runner_context_ignores_legacy_recovery_mode_without_roi_recovery_state():
    assert classify_v3_runner_context(
        {
            "strategyMode": "SMART_V3_RUNNER",
            "side": "LONG",
            "coinDailyTrend": "BULLISH",
            "recovery_mode": True,
            "recoveryStage": 0,
            "runtimeRecoveryState": {},
        }
    ) != main.V3_RUNNER_CONTEXT_RECOVERY


def test_signal_invalidation_gate_has_priority_and_cooldown(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []

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
    monkeypatch.setattr(
        main,
        "active_signals",
        {
            "SIGUSDT": {
                "side": "SHORT",
                "score": 90.0,
                "last_refresh_ts": time.time(),
            }
        },
    )

    pos = {
        "id": "sig-pos-1",
        "symbol": "SIGUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 95.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -5.0,
        "unrealizedPnlPercent": -50.0,
        "leverage": 10,
        "stopLoss": 70.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": 0,
        "isLive": False,
        "entrySignalScoreSnapshot": 92.0,
        "minConfidenceScoreSnapshot": 74.0,
        "lossGateOppositeCount": 2,
        "lossGateOppositeFirstTs": time.time() - 30.0,
        "entrySpreadPct": 0.05,
        "currentSpreadPct": 0.20,
        "entrySpreadLevel": "LOW",
        "currentSpreadLevel": "VERY HIGH",
        "entryAtrPct": 2.0,
        "currentAtrPct": 4.0,
        "entryImbalance": 3.0,
        "currentImbalance": -15.0,
        "recoveryStage": 0,
        "timeRecovery4HArmed": False,
        "timeRecovery8HArmed": False,
    }
    trader = DummyTrader(pos)
    ks = PositionBasedKillSwitch()

    asyncio.run(ks.check_positions(trader))
    assert len(trader.trades) == 1
    assert trader.trades[0]["reason"] == "SIGNAL_INVALIDATION_REDUCE"
    assert round(pos["contracts"], 4) == 0.85
    assert pos["signalInvalidationReduced"] is True
    assert pos["runtimeProtectionPhase"] == "INVALIDATION"
    assert pos["recoveryStage"] == 0

    asyncio.run(ks.check_positions(trader))
    assert len(trader.trades) == 1
    assert round(pos["contracts"], 4) == 0.85


def test_regime_deterioration_execution_risk_and_funding_decay_gates(monkeypatch):
    class DummyTrader:
        def __init__(self, pos):
            self.balance = 100.0
            self.positions = [pos]
            self.trades = []
            self.closed = []

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
    monkeypatch.setattr(main, "active_signals", {})
    ks = PositionBasedKillSwitch()

    regime_pos = {
        "id": "regime-pos-1",
        "symbol": "REGUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 96.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -4.0,
        "unrealizedPnlPercent": -40.0,
        "leverage": 10,
        "stopLoss": 70.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": 0,
        "isLive": False,
        "entrySpreadPct": 0.05,
        "currentSpreadPct": 0.20,
        "entrySpreadLevel": "LOW",
        "currentSpreadLevel": "VERY HIGH",
        "entryAtrPct": 2.0,
        "currentAtrPct": 4.2,
        "entryDepthUsd": 1000.0,
        "currentDepthUsd": 400.0,
        "entryImbalance": 4.0,
        "currentImbalance": -16.0,
    }
    regime_trader = DummyTrader(regime_pos)
    asyncio.run(ks.check_positions(regime_trader))
    assert regime_trader.trades[-1]["reason"] == "REGIME_DETERIORATION_REDUCE"
    assert round(regime_pos["contracts"], 4) == 0.9

    exec_pos = {
        "id": "exec-pos-1",
        "symbol": "EXECUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 97.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -3.0,
        "unrealizedPnlPercent": -30.0,
        "leverage": 10,
        "stopLoss": 70.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": 0,
        "isLive": False,
        "entrySpreadPct": 0.05,
        "currentSpreadPct": 0.05,
        "entrySpreadLevel": "LOW",
        "currentSpreadLevel": "LOW",
        "entryAtrPct": 2.0,
        "currentAtrPct": 2.0,
        "entryImbalance": 1.0,
        "currentImbalance": 1.0,
        "exec_snapshot": {"rolling_exit_slippage_p90": 2.0},
    }
    exec_trader = DummyTrader(exec_pos)
    asyncio.run(ks.check_positions(exec_trader))
    assert exec_trader.trades[-1]["reason"] == "EXECUTION_RISK_REDUCE"
    assert round(exec_pos["contracts"], 4) == 0.9

    funding_pos = {
        "id": "carry-pos-1",
        "symbol": "CARRYUSDT",
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 97.0,
        "size": 1.0,
        "contracts": 1.0,
        "sizeUsd": 100.0,
        "initialMargin": 10.0,
        "unrealizedPnl": -3.0,
        "unrealizedPnlPercent": -30.0,
        "leverage": 10,
        "stopLoss": 70.0,
        "takeProfit": 110.0,
        "trailActivation": 103.0,
        "trailDistance": 2.0,
        "runtimeTrailDistance": 2.0,
        "runtimeTrailActivationRoiPct": 12.0,
        "openTime": int((time.time() - 9 * 3600) * 1000),
        "isLive": False,
        "entrySpreadPct": 0.05,
        "currentSpreadPct": 0.05,
        "entrySpreadLevel": "LOW",
        "currentSpreadLevel": "LOW",
        "entryAtrPct": 2.0,
        "currentAtrPct": 2.0,
        "entryImbalance": 1.0,
        "currentImbalance": 1.0,
        "estimatedFundingCost": 1.0,
        "underwaterMeaningfulRecoveryTs": time.time() - (3 * 3600),
    }
    funding_trader = DummyTrader(funding_pos)
    asyncio.run(ks.check_positions(funding_trader))
    assert funding_trader.trades[-1]["reason"] == "FUNDING_DECAY_REDUCE"
    assert round(funding_pos["contracts"], 4) == 0.9
