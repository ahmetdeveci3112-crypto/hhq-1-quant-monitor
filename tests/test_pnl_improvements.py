import time

import main
from hyperopt import hhq_hyperoptimizer
from risk.sl_tp_engine import compute_tp_ladder_v2


def test_execution_cost_size_adjustment_reduces_notional(monkeypatch):
    monkeypatch.setattr(main, "_RFX_MODULES_AVAILABLE", True)
    monkeypatch.setattr(main, "rfx_estimate_slippage_bps", lambda *args, **kwargs: 18.5)

    result = main.compute_execution_cost_size_adjustment(
        planned_notional_usd=1000.0,
        leverage=10,
        spread_pct=0.22,
        depth_usd=850.0,
        tradeability_score=48.0,
        entry_exec_score=60.0,
    )

    assert result["applied"] is True
    assert result["size_mult"] == 0.65
    assert result["adjusted_notional_usd"] == 650.0
    assert "SLIP_CRITICAL" in result["reason"]


def test_execution_cost_size_adjustment_keeps_healthy_size(monkeypatch):
    monkeypatch.setattr(main, "_RFX_MODULES_AVAILABLE", True)
    monkeypatch.setattr(main, "rfx_estimate_slippage_bps", lambda *args, **kwargs: 4.2)

    result = main.compute_execution_cost_size_adjustment(
        planned_notional_usd=1000.0,
        leverage=10,
        spread_pct=0.04,
        depth_usd=5000.0,
        tradeability_score=82.0,
        entry_exec_score=88.0,
    )

    assert result["applied"] is False
    assert result["size_mult"] == 1.0
    assert result["adjusted_notional_usd"] == 1000.0


def test_tp_ladder_v2_structural_target_anchors_final_tp():
    ladder = compute_tp_ladder_v2(
        entry_price=100.0,
        atr=2.0,
        side="LONG",
        leverage=10,
        structural_target_price=108.0,
        structural_target_source="resistance_target",
    )

    prices = ladder["prices"]
    assert ladder["telemetry"]["structural_anchor_applied"] is True
    assert ladder["telemetry"]["structural_target_source"] == "resistance_target"
    assert prices["tp1"] < prices["tp2"] < prices["tp3"] < prices["tp_final"]
    assert prices["tp_final"] == 108.0


def test_revalidate_pending_entry_warns_on_mild_drift():
    now_ms = int(time.time() * 1000)
    order = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "spreadPct": 0.05,
        "signalScore": 82,
        "minConfidenceScoreSnapshot": 74,
        "createdAt": now_ms - (22 * 60 * 1000),
        "entryPrice": 100.0,
        "atr": 1.0,
        "pullbackPct": 0.8,
    }
    opportunity = {
        "signalAction": "NONE",
        "spreadPct": 0.25,
        "volumeRatio": 0.5,
        "obImbalanceTrend": 0.0,
        "price": 101.0,
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result["decision"] == "WARN_WAIT"
    assert any(reason.startswith("DRIFT_WAIT") for reason in result["reasons"])


def test_revalidate_pending_entry_drops_on_extreme_stale_drift():
    now_ms = int(time.time() * 1000)
    order = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "spreadPct": 0.05,
        "signalScore": 82,
        "minConfidenceScoreSnapshot": 74,
        "createdAt": now_ms - (25 * 60 * 1000),
        "entryPrice": 100.0,
        "atr": 1.0,
        "pullbackPct": 0.8,
    }
    opportunity = {
        "signalAction": "LONG",
        "spreadPct": 0.05,
        "volumeRatio": 1.2,
        "obImbalanceTrend": 3.0,
        "price": 104.5,
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result["decision"] == "FAIL_DROP"
    assert result["hard_reject"] is True
    assert any(reason.startswith("DRIFT_REJECT") for reason in result["reasons"])


def test_breakeven_buffer_uses_live_slippage_snapshot():
    base_pos = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "exchangeBreakEvenPrice": 100.0,
        "spreadPct": 0.05,
        "spreadLevel": "Low",
        "isLive": True,
        "exec_snapshot": {},
    }
    stressed_pos = {
        **base_pos,
        "exec_snapshot": {"rolling_exit_slippage_p90": 0.8},
    }

    base_buffer = main.resolve_buffered_breakeven_pct(base_pos, reason="BREAKEVEN")
    stressed_buffer = main.resolve_buffered_breakeven_pct(stressed_pos, reason="BREAKEVEN")
    stressed_ctx = main.compute_buffered_breakeven_price(stressed_pos, reason="BREAKEVEN")

    assert stressed_buffer > base_buffer
    assert stressed_ctx["price"] > stressed_ctx["anchor_price"]


def test_hyperopt_status_exposes_reject_feedback(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_reject_attribution_optimizer_hints",
        lambda: {
            "enabled": True,
            "fn_rate": 33.3,
            "candidate_hints": [{"reason": "TRADEABILITY_REJECT", "sample_size": 6}],
        },
    )

    status = hhq_hyperoptimizer.get_status()

    assert status["reject_feedback"]["enabled"] is True
    assert status["reject_feedback"]["candidate_hints"][0]["reason"] == "TRADEABILITY_REJECT"


def test_build_soft_signal_allocator_plan_reorders_and_scales_competition(monkeypatch):
    monkeypatch.setattr(
        main,
        "COIN_CLUSTER_MAP",
        {"AAAUSDT": "majors", "BBBUSDT": "majors", "CCCUSDT": "alts"},
        raising=False,
    )
    signals = [
        {"symbol": "BBBUSDT", "action": "LONG", "confidenceScore": 89, "sizeMultiplier": 1.0, "volumeRatio": 1.2, "spreadPct": 0.05},
        {"symbol": "AAAUSDT", "action": "LONG", "confidenceScore": 92, "sizeMultiplier": 1.0, "volumeRatio": 1.3, "spreadPct": 0.04},
        {"symbol": "CCCUSDT", "action": "LONG", "confidenceScore": 88, "sizeMultiplier": 1.0, "volumeRatio": 1.1, "spreadPct": 0.05},
    ]

    plan = main.build_soft_signal_allocator_plan(signals, positions=[], pending_orders=[], max_positions=2)

    assert [sig["symbol"] for sig in plan] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert plan[0]["allocatorRank"] == 1
    assert plan[0]["allocatorSizeMult"] == 1.0
    assert plan[1]["allocatorSizeMult"] == 0.88
    assert "QUEUE_CLUSTER_STACK" in plan[1]["allocatorReason"]
    assert plan[2]["allocatorSizeMult"] == 0.82
    assert "RANK_OUTSIDE_FREE" in plan[2]["allocatorReason"]


def test_compute_symbol_execution_memory_adjustment_caps_slippy_symbol():
    trades = [
        {"symbol": "TESTUSDT", "pnl": -12.0, "entry_slippage": 0.55, "entry_spread": 0.24},
        {"symbol": "TESTUSDT", "pnl": -8.0, "entry_slippage": 0.61, "entry_spread": 0.25},
        {"symbol": "TESTUSDT", "pnl": 2.0, "entry_slippage": 0.48, "entry_spread": 0.23},
        {"symbol": "TESTUSDT", "pnl": -6.0, "entry_slippage": 0.74, "entry_spread": 0.28},
        {"symbol": "TESTUSDT", "pnl": -4.0, "entry_slippage": 0.82, "entry_spread": 0.26},
    ]

    result = main.compute_symbol_execution_memory_adjustment(
        symbol="TESTUSDT",
        trades=trades,
        trade_pattern_penalty=-10,
        tracker_penalty=18,
    )

    assert result["applied"] is True
    assert result["size_mult"] == 0.68
    assert result["leverage_cap"] == 5
    assert "SYMBOL_SLIP_CRITICAL" in result["reason"]
    assert result["entry_slippage_p90"] >= 0.74


def test_compute_structural_trail_candidate_tightens_long_stop_to_broken_resistance():
    sr_snapshot = {
        "supports": [{"price": 101.5, "touches": 3, "strength": 2.1}],
        "resistances": [
            {"price": 105.0, "touches": 3, "strength": 2.8},
            {"price": 106.5, "touches": 2, "strength": 2.4},
        ],
    }

    result = main.compute_structural_trail_candidate(
        side="LONG",
        current_stop=102.0,
        current_price=108.0,
        entry_price=100.0,
        sr_levels=sr_snapshot,
        atr=2.0,
        spread_pct=0.05,
        volume_ratio=1.3,
        trailing_active=True,
        breakeven_activated=True,
    )

    assert result["applied"] is True
    assert result["structural_level"] == 106.5
    assert result["candidate_stop"] == 106.3
    assert result["source"] == "resistance_reclaim"


def test_compute_pending_capital_recycler_plan_prefers_stale_waiting_pending(monkeypatch):
    monkeypatch.setattr(
        main,
        "COIN_CLUSTER_MAP",
        {
            "NEWUSDT": "majors",
            "OLDUSDT": "majors",
            "KEEPUSDT": "alts",
        },
        raising=False,
    )
    now_ms = int(time.time() * 1000)
    new_signal = {
        "symbol": "NEWUSDT",
        "action": "LONG",
        "confidenceScore": 92,
        "volumeRatio": 1.4,
        "spreadPct": 0.04,
        "ev": 0.02,
        "trend_mode": True,
        "entryExecScore": 0.88,
        "allocatorPriorityScore": 94.0,
    }
    pending_orders = [
        {
            "id": "PO_OLD",
            "symbol": "OLDUSDT",
            "side": "LONG",
            "signalScore": 76,
            "signalScoreRaw": 78,
            "createdAt": now_ms - (18 * 60 * 1000),
            "confirmed": False,
            "feedbackReason": "WARN_WAIT_DRIFT",
            "recheckDecision": "WARN_WAIT",
            "entryExecScore": 0.61,
            "minConfidenceScoreSnapshot": 74,
        },
        {
            "id": "PO_KEEP",
            "symbol": "KEEPUSDT",
            "side": "LONG",
            "signalScore": 88,
            "signalScoreRaw": 90,
            "createdAt": now_ms - (4 * 60 * 1000),
            "confirmed": True,
            "entryExecScore": 0.82,
            "minConfidenceScoreSnapshot": 74,
        },
    ]

    result = main.compute_pending_capital_recycler_plan(
        new_signal,
        pending_orders,
        reason_code="DIRECTION_EXPOSURE",
        now_ms=now_ms,
    )

    assert result["applied"] is True
    assert result["candidate_order_id"] == "PO_OLD"
    assert result["candidate_symbol"] == "OLDUSDT"
    assert result["same_side"] is True
    assert result["priority_gap"] > result["required_gap"]


def test_compute_symbol_microstructure_cooldown_plan_triggers_on_bad_fill_history():
    result = main.compute_symbol_microstructure_cooldown_plan(
        {
            "trade_count": 6,
            "entry_slippage_p90": 0.78,
            "avg_entry_spread": 0.27,
            "win_rate": 28.0,
            "avg_pnl": -11.0,
            "tracker_penalty": 18,
            "trade_pattern_penalty": -10,
        },
        signal_score=84.0,
        entry_exec_score=0.73,
    )

    assert result["triggered"] is True
    assert result["severity"] == 3
    assert result["cooldown_sec"] == 3600
    assert "SLIP_HIGH" in result["reason"]


def test_compute_symbol_microstructure_cooldown_plan_allows_high_quality_override():
    result = main.compute_symbol_microstructure_cooldown_plan(
        {
            "trade_count": 5,
            "entry_slippage_p90": 0.67,
            "avg_entry_spread": 0.24,
            "win_rate": 46.0,
            "avg_pnl": -2.0,
            "tracker_penalty": 0,
            "trade_pattern_penalty": 0,
        },
        signal_score=93.0,
        entry_exec_score=0.91,
    )

    assert result["triggered"] is False
    assert result["override_high_quality"] is True


def test_resolve_position_time_budget_profile_varies_by_regime():
    trend_profile = main.resolve_position_time_budget_profile(
        {
            "strategyMode": "SMART_V3_RUNNER",
            "trend_mode": True,
            "truth_snapshot": {"struct_regime": "TRENDING_UP"},
            "entryCostEstSlippageBps": 4.0,
            "entryDepthCover": 2.0,
        }
    )
    range_profile = main.resolve_position_time_budget_profile(
        {
            "strategyMode": "LEGACY",
            "trend_mode": False,
            "truth_snapshot": {"struct_regime": "RANGING"},
            "entryCostEstSlippageBps": 6.0,
            "entryDepthCover": 1.4,
        }
    )

    assert trend_profile["regime_bucket"] == "TREND"
    assert trend_profile["age_mult"] > 1.0
    assert trend_profile["hard_limit_mult"] > 1.0
    assert range_profile["regime_bucket"] == "RANGE"
    assert range_profile["age_mult"] < 1.0
    assert range_profile["bounce_mult"] < 1.0


def test_calculate_signal_ev_prefers_context_stats_when_available():
    engine = main.PaperTradingEngine.__new__(main.PaperTradingEngine)
    engine.score_band_stats = {
        "80-90": {
            "wins": 30,
            "losses": 20,
            "total_win": 30.0,
            "total_loss": -22.0,
            "avg_win": 1.0,
            "avg_loss": -1.1,
        }
    }
    engine.score_band_stats_by_context = {
        "RUNNER:TREND": {
            "80-90": {
                "wins": 12,
                "losses": 3,
                "total_win": 21.6,
                "total_loss": -2.4,
                "avg_win": 1.8,
                "avg_loss": -0.8,
            }
        }
    }
    base_engine = main.PaperTradingEngine.__new__(main.PaperTradingEngine)
    base_engine.score_band_stats = engine.score_band_stats
    base_engine.score_band_stats_by_context = {}
    signal = {
        "symbol": "CTXUSDT",
        "_rawConfidenceScore": 84,
        "confidenceScore": 84,
        "spreadPct": 0.05,
        "strategyMode": "SMART_V3_RUNNER",
        "trend_mode": True,
        "truth_snapshot": {"struct_regime": "TRENDING_UP"},
    }

    ctx_ev = engine.calculate_signal_ev(signal)
    base_ev = base_engine.calculate_signal_ev(signal)

    assert ctx_ev > base_ev
    assert ctx_ev > 0


def test_compute_opportunity_forecast_blends_fill_and_overlay(monkeypatch):
    class DummyEntryForecastService:
        enabled = True

        @staticmethod
        def build_features_dict(raw):
            return raw

        @staticmethod
        def predict_proba(features):
            return {
                "prob": 0.66,
                "uncertainty": 0.20,
                "source": "model",
                "model_version": "test_v1",
            }

    class DummyFreqAI:
        enabled = True
        is_trained = False

    monkeypatch.setattr(main, "entry_forecast_service", DummyEntryForecastService())
    monkeypatch.setattr(main, "freqai_model", DummyFreqAI())

    signal = {
        "action": "LONG",
        "confidenceScore": 88,
        "entryExecScore": 84,
        "spreadPct": 0.05,
        "volumeRatio": 1.3,
        "obImbalanceTrend": 1.2,
        "zscore": -2.1,
        "hurst": 0.44,
        "adx": 27.0,
        "leaderLagScore": 6.0,
        "liqEchoScore": 4.0,
        "microstructureScore": 2.0,
    }

    forecast = main.compute_opportunity_forecast(signal, side="LONG", include_ev=False)

    assert forecast["edge_prob"] > 0.70
    assert forecast["band"] in {"MEDIUM", "STRONG"}
    assert "fill" in forecast["components"]
    assert "overlay" in forecast["components"]


def test_evaluate_entry_quality_forecast_rescue_preview_soft_passes_strong_band():
    result = main.evaluate_entry_quality_forecast_rescue_preview(
        signal={"confidenceScore": 90},
        forecast={"band": "STRONG", "edge_prob": 0.67},
        eq_pass_count=1,
        exec_score=82.0,
        spread_pct=0.12,
        volume_ratio=0.95,
        strategy_mode="LEGACY",
    )

    assert result["decision"] == "SOFT_PASS"
    assert result["size_mult"] == 0.65
    assert result["leverage_cap"] == 8


def test_evaluate_forecast_alpha_rescue_soft_passes_obi_gate():
    result = main.evaluate_forecast_alpha_rescue(
        signal={"entryQualityReasons": ["A:Vol(1.4x)"], "entryExecScore": 81.0, "spreadPct": 0.12},
        gate_type="OBI_NEUTRAL_LOW_VOL",
        context={"eq_count": 1, "exec_score": 81.0, "spread_pct": 0.12},
        forecast={"band": "STRONG", "edge_prob": 0.69},
    )

    assert result["decision"] == "SOFT_PASS"
    assert result["score_penalty"] == 5


def test_compute_leader_lag_overlay_detects_cluster_laggard(monkeypatch):
    monkeypatch.setattr(
        main,
        "COIN_CLUSTER_MAP",
        {"AAAUSDT": "l1", "BBBUSDT": "l1", "CCCUSDT": "l1"},
        raising=False,
    )

    overlay = main.compute_leader_lag_overlay(
        "BBBUSDT",
        "LONG",
        signal={"priceChange24h": 2.1, "confidenceScore": 86},
        opportunities=[
            {"symbol": "AAAUSDT", "signalAction": "LONG", "priceChange24h": 4.4, "signalScore": 92},
            {"symbol": "BBBUSDT", "signalAction": "LONG", "priceChange24h": 2.1, "signalScore": 86},
            {"symbol": "CCCUSDT", "signalAction": "SHORT", "priceChange24h": -1.2, "signalScore": 74},
        ],
    )

    assert overlay["applied"] is True
    assert overlay["role"] == "LAGGARD"
    assert overlay["leader_symbol"] == "AAAUSDT"
    assert overlay["score"] > 0


def test_compute_liquidation_echo_overlay_detects_sell_exhaustion():
    now_ts = time.time()

    class DummyTracker:
        recent_liquidations = {
            "TESTUSDT": [
                {"timestamp": now_ts - 8, "side": "SELL", "usd": 42000.0},
                {"timestamp": now_ts - 18, "side": "SELL", "usd": 28000.0},
                {"timestamp": now_ts - 12, "side": "BUY", "usd": 8000.0},
            ]
        }

    overlay = main.compute_liquidation_echo_overlay(
        "TESTUSDT",
        "LONG",
        signal={"zscore": -2.2, "obImbalanceTrend": 0.1, "spreadPct": 0.08, "volumeRatio": 1.1},
        tracker=DummyTracker(),
    )

    assert overlay["applied"] is True
    assert overlay["state"] == "SELL_EXHAUSTION"
    assert overlay["score"] >= 6.0
    assert overlay["impulse_usd"] >= 50000.0


def test_compute_microstructure_overlay_flags_toxic_setup():
    overlay = main.compute_microstructure_overlay(
        {
            "action": "LONG",
            "spreadPct": 0.26,
            "tradeabilityScore": 48.0,
            "entryExecScore": 58.0,
            "entryCostEstSlippageBps": 14.0,
            "obImbalanceTrend": -0.9,
            "orderBookContextImbalance": -0.4,
        }
    )

    assert overlay["flow_toxicity_score"] >= 65.0
    assert overlay["microstructure_score"] < 0
    assert overlay["refill_failure_score"] > 0


def test_compute_microstructure_snapshot_from_events_detects_supportive_long_flow():
    now_ms = int(time.time() * 1000)
    depth_events = [
        {
            "ts_ms": now_ms - 900,
            "bids": [(100.0, 10.0), (99.9, 8.0), (99.8, 6.0), (99.7, 5.0), (99.6, 4.0)],
            "asks": [(100.1, 9.0), (100.2, 7.0), (100.3, 6.0), (100.4, 5.0), (100.5, 4.0)],
        },
        {
            "ts_ms": now_ms - 600,
            "bids": [(100.0, 12.5), (99.9, 8.5), (99.8, 6.5), (99.7, 5.0), (99.6, 4.0)],
            "asks": [(100.1, 8.5), (100.2, 6.8), (100.3, 5.8), (100.4, 4.8), (100.5, 3.8)],
        },
        {
            "ts_ms": now_ms - 300,
            "bids": [(100.0, 13.0), (99.9, 8.8), (99.8, 6.7), (99.7, 5.2), (99.6, 4.1)],
            "asks": [(100.1, 8.2), (100.2, 6.7), (100.3, 5.6), (100.4, 4.6), (100.5, 3.7)],
        },
    ]
    trade_events = [
        {"ts_ms": now_ms - 1800, "sign": 1.0, "notional": 8000.0},
        {"ts_ms": now_ms - 1300, "sign": 1.0, "notional": 9000.0},
        {"ts_ms": now_ms - 500, "sign": 1.0, "notional": 12000.0},
    ]

    snapshot = main.compute_microstructure_snapshot_from_events("LONG", depth_events, trade_events, now_ms=now_ms)

    assert snapshot["available"] is True
    assert snapshot["fresh"] is True
    assert snapshot["microstructure_score"] > 0
    assert snapshot["flow_toxicity_score"] < 50.0
    assert snapshot["same_side_refill_ratio"] > snapshot["same_side_cancel_ratio"]


def test_build_microstructure_watchlist_prioritizes_signals_pending_and_positions():
    watchlist = main.build_microstructure_watchlist(
        signals=[
            {"symbol": "AAAUSDT", "confidenceScore": 92},
            {"symbol": "BBBUSDT", "confidenceScore": 88},
        ],
        opportunities=[
            {"symbol": "CCCUSDT", "signalScore": 86, "volume24h": 5_000_000},
            {"symbol": "DDDUSDT", "signalScore": 84, "volume24h": 4_000_000},
        ],
        positions=[{"symbol": "EEEUSDT", "signalScore": 70, "unrealizedPnlPercent": 12.0}],
        pending_orders=[{"symbol": "FFFUSDT", "signalScore": 80, "forecastEdgeProb": 0.64}],
        max_watchers=4,
    )

    assert watchlist == ["AAAUSDT", "BBBUSDT", "FFFUSDT", "EEEUSDT"]


def test_evaluate_event_alpha_trigger_prefers_liquidation_echo_for_long():
    trigger = main.evaluate_event_alpha_trigger(
        {
            "symbol": "TESTUSDT",
            "zscore": -1.25,
            "spreadPct": 0.08,
            "volumeRatio": 1.1,
            "entryExecScore": 82.0,
            "microstructureScore": 2.0,
            "flowToxicityScore": 42.0,
        },
        leader_lag={"score": 2.0, "role": "NONE"},
        liq_echo={"score": 7.0, "state": "SELL_EXHAUSTION", "impulse_usd": 64000.0},
        micro_snapshot={"microstructure_score": 2.0, "flow_toxicity_score": 42.0},
    )

    assert trigger["triggered"] is True
    assert trigger["side"] == "LONG"
    assert trigger["score_bonus"] >= 8
    assert "LIQ_ECHO" in trigger["reason"]
