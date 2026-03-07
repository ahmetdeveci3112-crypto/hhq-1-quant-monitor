import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class DummyPaperTrader:
    def __init__(self):
        self.balance = 1000.0
        self.trades = []
        self.closed = []

    def _normalize_close_reason(self, reason: str) -> str:
        return main.PaperTradingEngine._normalize_close_reason(reason)

    def close_via_engine(self, pos: dict, price: float, reason: str, source: str = ""):
        self.closed.append(
            {
                "symbol": pos.get("symbol"),
                "price": price,
                "reason": reason,
                "source": source,
            }
        )


def _base_v3_position(**overrides):
    now_ms = int(time.time() * 1000)
    pos = {
        "id": "v3-pos-1",
        "symbol": "TESTUSDT",
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "exitOwner": main.V3_EXIT_OWNER,
        "runnerContext": main.V3_RUNNER_CONTEXT_TREND,
        "baseRunnerContext": main.V3_RUNNER_CONTEXT_TREND,
        "side": "LONG",
        "entryPrice": 100.0,
        "currentPrice": 100.0,
        "stopLoss": 98.0,
        "trailingStop": 98.0,
        "initialStopLoss": 98.0,
        "trailDistance": 1.8,
        "trailActivation": 101.0,
        "contracts": 10.0,
        "size": 10.0,
        "sizeUsd": 1000.0,
        "initialMargin": 100.0,
        "leverage": 10,
        "atr": 1.0,
        "openTime": now_ms - 60_000,
        "partial_tp_state": {},
        "tp_ladder_levels": [
            {"key": "tp1", "pct": 0.8, "close_pct": 0.15},
            {"key": "tp2", "pct": 1.6, "close_pct": 0.20},
            {"key": "tp3", "pct": 2.4, "close_pct": 0.25},
            {"key": "tp_final", "pct": 3.2, "close_pct": 0.40},
        ],
        "tpCloseSplit": "15/20/25/40",
        "tp1ArmDelaySec": 90,
        "tp1ArmAtrReq": 0.50,
        "beAfterTrailAtrReq": 0.00,
        "runner_be_buffer_mult": 0.90,
        "signalScore": 92,
    }
    pos.update(overrides)
    return pos


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    monkeypatch.setattr(main, "active_signals", {})
    monkeypatch.setattr(
        main,
        "global_paper_trader",
        SimpleNamespace(min_confidence_score=74, strategy_mode=main.STRATEGY_MODE_SMART_V3_RUNNER),
    )


def test_runner_context_resolves_trend_aligned_profile():
    controls = main.resolve_runner_exit_controls(
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "coinDailyTrend": "BULLISH",
        }
    )
    assert controls["runner_context"] == main.V3_RUNNER_CONTEXT_TREND
    assert controls["exit_owner"] == main.V3_EXIT_OWNER
    assert controls["trail_act_mult"] == 1.55
    assert controls["trail_dist_mult"] == 1.8
    assert controls["tp_tighten"] == 1.0
    assert controls["be_buffer_mult"] == 0.9
    assert controls["tp_close_split"] == (0.15, 0.20, 0.25, 0.40)
    assert controls["trail_arm_delay_sec"] == 90
    assert controls["trail_arm_atr_req"] == 0.50


def test_runner_context_resolves_countertrend_profile():
    controls = main.resolve_runner_exit_controls(
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "coinDailyTrend": "BEARISH",
        }
    )
    assert controls["runner_context"] == main.V3_RUNNER_CONTEXT_COUNTER
    assert controls["trail_act_mult"] == 1.40
    assert controls["trail_dist_mult"] == 1.65
    assert controls["tp_tighten"] == 0.92
    assert controls["be_buffer_mult"] == 1.0
    assert controls["tp_close_split"] == (0.20, 0.25, 0.25, 0.30)


def test_profitable_v3_trail_exit_confirmation_is_slower():
    pos = _base_v3_position(isTrailingActive=True)
    assert main.get_trail_exit_confirmation_requirements(pos, profitable=True) == (4, 12)
    assert main.get_trail_exit_confirmation_requirements(pos, profitable=False) == (2, 5)


def test_v3_fallback_tp_ladder_keeps_runner_split():
    ladder = main.build_v3_runner_fallback_tp_ladder(
        entry_price=100.0,
        atr=1.2,
        side="LONG",
        leverage=10,
        spread_pct=0.05,
        volume_ratio=1.4,
        adx=30.0,
        hurst=0.58,
        coin_daily_trend="BULLISH",
        exec_score=84.0,
        spread_level="Normal",
        tp_tighten=1.0,
        close_split=(0.15, 0.20, 0.25, 0.40),
    )
    assert ladder["version"] == "v3-fallback-4tier"
    assert [lv["key"] for lv in ladder["levels"]] == ["tp1", "tp2", "tp3", "tp_final"]
    assert [lv["close_pct"] for lv in ladder["levels"]] == [0.15, 0.20, 0.25, 0.40]
    assert ladder["telemetry"]["close_split"] == "15/20/25/40"


@pytest.mark.asyncio
async def test_v3_tp1_arms_trail_without_immediate_activation():
    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    pos = _base_v3_position(currentPrice=101.0)
    actions = {"trail_activated": [], "time_reduced": [], "time_closed": [], "partial_tp": [], "checked": 0}

    handled = await manager._handle_v3_runner_position(trader, pos, 101.0, actions)

    assert handled is True
    assert pos["partial_tp_state"]["tp1"] is True
    assert pos["trailArmed"] is True
    assert pos.get("isTrailingActive", False) is False
    assert pos.get("breakeven_activated", False) is False
    assert pos["lastExitDecision"] == "TP_PARTIAL_TP1"
    assert len(trader.trades) == 1
    assert trader.trades[0]["reason"] == "TP_PARTIAL_TP1"
    assert trader.balance > 1000.0


@pytest.mark.asyncio
async def test_v3_trail_activation_waits_for_delay_and_atr_move():
    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    pos = _base_v3_position(
        isTrailingActive=False,
        trailArmed=True,
        trailArmSince=time.time() - 95,
        trailArmPrice=101.0,
        currentPrice=101.4,
        partial_tp_state={"tp1": True},
    )
    actions = {"trail_activated": [], "time_reduced": [], "time_closed": [], "partial_tp": [], "checked": 0}

    handled = await manager._handle_v3_runner_position(trader, pos, 101.6, actions)

    assert handled is True
    assert pos["trailArmed"] is False
    assert pos["isTrailingActive"] is True
    assert pos["lastExitDecision"] == "TRAIL_ARMED_ACTIVATED"
    assert pos["breakeven_activated"] is True
    assert pos["trailingStop"] >= pos["entryPrice"]


@pytest.mark.asyncio
async def test_v3_recovery_stage1_tightens_stop(monkeypatch):
    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(
        main,
        "active_signals",
        {"TESTUSDT": {"side": "SHORT", "score": 90.0, "last_refresh_ts": time.time()}},
    )
    pos = _base_v3_position(
        currentPrice=99.39,
        trailingStop=98.0,
        stopLoss=98.0,
        oppositePressureCount=1,
        oppositeLastSeenTs=time.time() - 30,
        tp_ladder_levels=[],
    )
    actions = {"trail_activated": [], "time_reduced": [], "time_closed": [], "partial_tp": [], "checked": 0}

    handled = await manager._handle_v3_runner_position(trader, pos, 99.39, actions)

    assert handled is True
    assert pos["recoveryStage"] == 1
    assert pos["runnerContext"] == main.V3_RUNNER_CONTEXT_RECOVERY
    assert pos["trailingStop"] > 98.0
    assert pos["lastExitDecision"] == "RECOVERY_TIGHTEN_STAGE1"


@pytest.mark.asyncio
async def test_v3_recovery_stage2_reduces_position(monkeypatch):
    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(
        main,
        "active_signals",
        {"TESTUSDT": {"side": "SHORT", "score": 91.0, "last_refresh_ts": time.time()}},
    )
    pos = _base_v3_position(
        currentPrice=98.99,
        trailingStop=98.0,
        stopLoss=98.0,
        oppositePressureCount=3,
        oppositeLastSeenTs=time.time() - 30,
        recoveryStage=1,
        tp_ladder_levels=[],
    )
    actions = {"trail_activated": [], "time_reduced": [], "time_closed": [], "partial_tp": [], "checked": 0}

    handled = await manager._handle_v3_runner_position(trader, pos, 98.99, actions)

    assert handled is True
    assert pos["recoveryStage"] == 2
    assert pos["runnerContext"] == main.V3_RUNNER_CONTEXT_RECOVERY
    assert pos["contracts"] == pytest.approx(7.5)
    assert trader.trades[-1]["reason"] == "RECOVERY_REDUCE"
    assert pos["lastExitDecision"] == "RECOVERY_TIGHTEN_STAGE2"


@pytest.mark.asyncio
async def test_v3_recovery_stage3_closes_position(monkeypatch):
    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    monkeypatch.setattr(
        main,
        "active_signals",
        {"TESTUSDT": {"side": "SHORT", "score": 95.0, "last_refresh_ts": time.time()}},
    )
    pos = _base_v3_position(
        currentPrice=98.39,
        trailingStop=98.0,
        stopLoss=98.0,
        oppositePressureCount=5,
        oppositeLastSeenTs=time.time() - 30,
        recoveryStage=2,
        tp_ladder_levels=[],
    )
    actions = {"trail_activated": [], "time_reduced": [], "time_closed": [], "partial_tp": [], "checked": 0}

    handled = await manager._handle_v3_runner_position(trader, pos, 98.39, actions)

    assert handled is True
    assert pos["recoveryStage"] == 3
    assert pos["lastExitDecision"] == "RECOVERY_EXIT"
    assert trader.closed[-1]["reason"] == "RECOVERY_EXIT"


@pytest.mark.asyncio
async def test_v3_protection_retry_sets_internal_only_and_recovers(monkeypatch):
    class FailingThenSuccessLiveTrader:
        def __init__(self):
            self.calls = 0

        async def set_stop_loss(self, symbol, side, contracts, stop_price):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("-2021")
            return {"id": "sl-1", "stopPrice": stop_price}

    trader = DummyPaperTrader()
    manager = main.TimeBasedPositionManager()
    live = FailingThenSuccessLiveTrader()
    monkeypatch.setattr(main, "live_binance_trader", live)

    pos = _base_v3_position(
        isLive=True,
        contracts=8.0,
        currentPrice=101.0,
        trailingStop=100.2,
        stopLoss=100.2,
    )

    await manager._sync_v3_protective_order(pos, 100.2, "V3_BE_AFTER_TRAIL")
    assert pos["internalProtectionOnly"] is True
    assert pos["protectionRetryAt"] > 0
    assert pos["protectionRetryPrice"] == pytest.approx(100.2)

    pos["protectionRetryAt"] = int(time.time()) - 1
    await manager._maybe_retry_v3_protection(pos, 101.5)

    assert pos["internalProtectionOnly"] is False
    assert pos["protectionRetryAt"] == 0
    assert pos["exchange_sl_order_id"] == "sl-1"
    assert pos["exchange_protective_order_ids"] == ["sl-1"]
