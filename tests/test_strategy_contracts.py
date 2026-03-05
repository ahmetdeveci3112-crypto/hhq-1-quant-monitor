"""Strategy routing and runner control contract tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    apply_runner_be_buffer,
    get_smart_v2_strategy_profile,
    resolve_runner_exit_controls,
)
from risk.strategy_profile import (
    STRATEGY_MODE_LEGACY,
    STRATEGY_MODE_SMART_V2,
    STRATEGY_MODE_SMART_V3_RUNNER,
)


def test_legacy_mode_keeps_identity_strategy_profile():
    profile = get_smart_v2_strategy_profile(
        mode=STRATEGY_MODE_LEGACY,
        signal_side="LONG",
        hurst=0.62,
        adx=35,
        spread_pct=0.05,
        volume_ratio=2.0,
        volatility_ratio=1.1,
        market_regime="TRENDING_UP",
        is_volume_spike=True,
        imbalance=6.0,
        ob_imbalance_trend=4.0,
        macro_trend_dir="UP",
    )
    assert profile["strategy_mode"] == STRATEGY_MODE_LEGACY
    assert profile["active_strategy"] == "legacy"
    assert profile["threshold_mult"] == 1.0
    assert profile["entry_mult"] == 1.0
    assert profile["exit_mult"] == 1.0
    assert profile["leverage_mult"] == 1.0


def test_smart_v2_trend_follow_profile_stays_smart():
    profile = get_smart_v2_strategy_profile(
        mode=STRATEGY_MODE_SMART_V2,
        signal_side="LONG",
        hurst=0.60,
        adx=30,
        spread_pct=0.07,
        volume_ratio=1.3,
        volatility_ratio=1.0,
        market_regime="TRENDING_UP",
        is_volume_spike=False,
        imbalance=3.0,
        ob_imbalance_trend=2.5,
        macro_trend_dir="UP",
    )
    assert profile["strategy_mode"] == STRATEGY_MODE_SMART_V2
    assert profile["active_strategy"] == "trend_follow"
    assert profile["strategy_label"] == "Trend Takibi"
    assert profile["entry_mult"] < 1.0
    assert profile["exit_mult"] > 1.0
    assert "TREND" in profile["notes"]


def test_runner_mode_uses_same_entry_router_but_keeps_runner_mode():
    profile = get_smart_v2_strategy_profile(
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        signal_side="LONG",
        hurst=0.65,
        adx=34,
        spread_pct=0.05,
        volume_ratio=1.5,
        volatility_ratio=1.0,
        market_regime="TRENDING_UP",
        is_volume_spike=True,
        imbalance=5.0,
        ob_imbalance_trend=3.0,
        macro_trend_dir="UP",
    )
    assert profile["strategy_mode"] == STRATEGY_MODE_SMART_V3_RUNNER
    assert profile["active_strategy"] == "momentum_breakout"
    assert profile["entry_mult"] < 1.0
    assert profile["exit_mult"] > 1.0


def test_runner_controls_are_neutral_when_mode_is_not_runner():
    controls = resolve_runner_exit_controls(
        payload={
            "strategyMode": STRATEGY_MODE_SMART_V2,
            "runner_trail_act_mult": 1.7,
            "runner_trail_dist_mult": 1.6,
            "runner_tp_tighten": 0.8,
            "runner_be_buffer_mult": 1.4,
        }
    )
    assert controls["mode"] == STRATEGY_MODE_SMART_V2
    assert controls["enabled"] is False
    assert controls["trail_act_mult"] == 1.0
    assert controls["trail_dist_mult"] == 1.0
    assert controls["tp_tighten"] == 1.0
    assert controls["be_buffer_mult"] == 1.0


def test_runner_controls_clamp_overrides_in_runner_mode():
    controls = resolve_runner_exit_controls(
        payload={
            "strategyMode": STRATEGY_MODE_SMART_V3_RUNNER,
            "runner_trail_act_mult": 5.0,
            "runner_trail_dist_mult": 0.1,
            "runner_tp_tighten": 0.2,
            "runner_be_buffer_mult": 3.0,
        }
    )
    assert controls["mode"] == STRATEGY_MODE_SMART_V3_RUNNER
    assert controls["enabled"] is True
    assert controls["trail_act_mult"] == 1.8
    assert controls["trail_dist_mult"] == 0.6
    assert controls["tp_tighten"] == 0.7
    assert controls["be_buffer_mult"] == 1.6


def test_apply_runner_be_buffer_is_noop_when_disabled():
    adjusted = apply_runner_be_buffer(0.003, {"enabled": False, "be_buffer_mult": 1.6})
    assert adjusted == 0.003


def test_apply_runner_be_buffer_clamps_extremes():
    adjusted = apply_runner_be_buffer(0.03, {"enabled": True, "be_buffer_mult": 2.0})
    assert adjusted == 0.02
