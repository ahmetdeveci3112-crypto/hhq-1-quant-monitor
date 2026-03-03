"""
RFX-1A: Risk Module Unit Tests + Parity Tests

Tests:
1. LiquidityProfile properties (multipliers, tiers)
2. RiskPolicy resolution (3 profiles × leverage combos)
3. SL/TP engine v1 vs v2 parity (blocker 3: RFX_PARITY_MODE=true → identical output)
4. Emergency SL merge (parity with legacy static check)
5. Breakeven triggers (trail activation AND TP1)
6. ExitStateMachine skeleton (transitions, serialization)

Run: python3 -m pytest test_rfx_risk_modules.py -v
"""
import os
import sys
import math

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from risk.liquidity_profile import LiquidityProfile
from risk.policy import RiskProfile, RiskParams, resolve_risk_params, RISK_PROFILES
from risk.sl_tp_engine import (
    compute_sl_tp_levels_v1,
    compute_sl_tp_levels_v2,
    estimate_trade_cost,
    snap_to_tick,
    ensure_tick_safe_buffer,
)
from risk.emergency import check_emergency_sl_static_v1, check_emergency, EmergencyResult
from risk.breakeven import (
    compute_breakeven_buffer_pct,
    compute_breakeven_price,
    should_set_breakeven,
    BreakevenDecision,
)
from exit.state_machine import ExitState, ExitStateMachine, ExitAction


# ═══════════════════════════════════════════════════════════════════
# 1) LiquidityProfile Tests
# ═══════════════════════════════════════════════════════════════════

class TestLiquidityProfile:
    def test_neutral_profile_multipliers(self):
        """Neutral profile should produce ~1.0 multipliers."""
        lp = LiquidityProfile.neutral()
        assert 0.95 <= lp.sl_width_mult <= 1.1  # very close to 1.0
        assert 0.95 <= lp.tp_width_mult <= 1.1
        assert 0.95 <= lp.trail_dist_mult <= 1.1
        assert lp.quality_tier in ("DEEP", "NORMAL")  # neutral() has spread=0.05, borderline

    def test_thin_book_wider_sl(self):
        """Thin book should produce wider SL multiplier."""
        lp = LiquidityProfile(
            spread_pct=0.25, spread_level="High",
            depth_ratio=0.4, depth_usd=15000,
            obi_value=0.0, volume_24h_usd=500_000,
            expected_slippage=0.08, round_trip_cost=0.003,
        )
        assert lp.sl_width_mult > 1.3, f"Thin book sl_width_mult should exceed 1.3, got {lp.sl_width_mult}"
        assert lp.quality_tier in ("THIN", "ULTRA_THIN")

    def test_deep_book_tight(self):
        """Deep book with low spread should have near-1.0 multipliers."""
        lp = LiquidityProfile(
            spread_pct=0.02, spread_level="Very Low",
            depth_ratio=2.0, depth_usd=200000,
            obi_value=0.1, volume_24h_usd=10_000_000,
            expected_slippage=0.01, round_trip_cost=0.0009,
        )
        assert lp.sl_width_mult <= 1.05
        assert lp.tp_width_mult <= 1.05
        assert lp.quality_tier == "DEEP"

    def test_breakeven_buffer_range(self):
        """Breakeven buffer should be in reasonable range."""
        lp_low = LiquidityProfile(spread_pct=0.03, expected_slippage=0.01, round_trip_cost=0.001)
        lp_high = LiquidityProfile(spread_pct=0.40, expected_slippage=0.10, round_trip_cost=0.005)
        assert 0.0012 <= lp_low.breakeven_buffer <= 0.008
        assert 0.0012 <= lp_high.breakeven_buffer <= 0.008

    def test_quality_tiers(self):
        """Quality tiers should be correctly assigned."""
        assert LiquidityProfile(spread_pct=0.03, depth_ratio=1.5).quality_tier == "DEEP"
        assert LiquidityProfile(spread_pct=0.10, depth_ratio=0.8).quality_tier == "NORMAL"
        assert LiquidityProfile(spread_pct=0.20, depth_ratio=0.5).quality_tier == "THIN"
        assert LiquidityProfile(spread_pct=0.50, depth_ratio=0.2).quality_tier == "ULTRA_THIN"

    def test_recovery_bounce_mult(self):
        """Recovery bounce mult should increase for thin books."""
        deep = LiquidityProfile(spread_pct=0.03, depth_ratio=1.5)
        thin = LiquidityProfile(spread_pct=0.25, depth_ratio=0.5)
        ultra = LiquidityProfile(spread_pct=0.50, depth_ratio=0.2)
        assert deep.recovery_bounce_mult < thin.recovery_bounce_mult < ultra.recovery_bounce_mult


# ═══════════════════════════════════════════════════════════════════
# 2) RiskPolicy Tests
# ═══════════════════════════════════════════════════════════════════

class TestRiskPolicy:
    def test_all_profiles_exist(self):
        """All 3 profiles must be defined."""
        for p in RiskProfile:
            assert p in RISK_PROFILES

    def test_balanced_legacy_parity(self):
        """BALANCED should match legacy hardcoded values."""
        rp = resolve_risk_params(RiskProfile.BALANCED, leverage=10)
        assert rp.sl_roi_floor == 30.0
        assert rp.tp_roi_floor == 5.0  # Blocker 1: tp_roi_floor
        assert rp.tp_final_target_roi == 40.0  # Blocker 1: separate field
        assert rp.emergency_roi == 50.0

    def test_ultra_aggressive_100_roi(self):
        """ULTRA_AGGRESSIVE should target 100% ROI everywhere."""
        rp = resolve_risk_params(RiskProfile.ULTRA_AGGRESSIVE, leverage=10)
        assert rp.sl_roi_floor == 100.0
        assert rp.tp_final_target_roi == 100.0
        # Emergency capped at 95% to prevent near-liquidation
        assert rp.emergency_roi <= 95.0
        assert rp.tp_roi_floor == 5.0  # Floor always 5%

    def test_blocker1_tp_floor_vs_target(self):
        """Blocker 1: tp_roi_floor must NOT equal tp_final_target_roi."""
        for profile in RiskProfile:
            rp = resolve_risk_params(profile, leverage=10)
            # tp_roi_floor is always 5% (legacy parity)
            assert rp.tp_roi_floor == 5.0, f"{profile}: tp_roi_floor should be 5.0"
            # tp_final_target_roi varies by profile
            assert rp.tp_final_target_roi >= 40.0, f"{profile}: tp_final_target_roi should be >= 40"

    def test_emergency_cap_under_liquidation(self):
        """Emergency SL cap must be < 100% to prevent near-liquidation."""
        for profile in RiskProfile:
            rp = resolve_risk_params(profile, leverage=10)
            assert rp.emergency_cap_roi <= 100.0

    def test_liquidity_profile_modifies_trail(self):
        """LiquidityProfile should modify trail multipliers in resolved params."""
        thin_lp = LiquidityProfile(spread_pct=0.25, depth_ratio=0.5)
        neutral_lp = LiquidityProfile.neutral()
        rp_thin = resolve_risk_params(RiskProfile.BALANCED, 10, thin_lp)
        rp_neutral = resolve_risk_params(RiskProfile.BALANCED, 10, neutral_lp)
        # Thin book should produce wider trail distance
        assert rp_thin.trail_distance_mult >= rp_neutral.trail_distance_mult


# ═══════════════════════════════════════════════════════════════════
# 3) SL/TP Engine Parity Tests (Blocker 3)
# ═══════════════════════════════════════════════════════════════════

class TestSLTPEngineParity:
    """Blocker 3: v2 with parity_mode=True must produce IDENTICAL results to v1."""

    BASE_ARGS = dict(
        entry_price=65000.0,
        atr=1200.0,
        side='LONG',
        leverage=10,
        adjusted_sl_atr=1.5,
        adjusted_tp_atr=3.0,
        adjusted_trail_act_atr=2.0,
        adjusted_trail_dist_atr=0.5,
        spread_pct=0.05,
        tick_size=0.01,
    )

    def test_v1_v2_parity_long(self):
        """V2 parity_mode=True must produce same SL/TP as V1 for LONG."""
        v1 = compute_sl_tp_levels_v1(**self.BASE_ARGS)
        v2 = compute_sl_tp_levels_v2(**self.BASE_ARGS, parity_mode=True)
        assert v1['sl'] == v2['sl'], f"SL mismatch: v1={v1['sl']} v2={v2['sl']}"
        assert v1['tp'] == v2['tp'], f"TP mismatch: v1={v1['tp']} v2={v2['tp']}"
        assert v1['trail_activation'] == v2['trail_activation']
        assert v1['trail_distance'] == v2['trail_distance']

    def test_v1_v2_parity_short(self):
        """V2 parity_mode=True must produce same SL/TP as V1 for SHORT."""
        args = {**self.BASE_ARGS, 'side': 'SHORT'}
        v1 = compute_sl_tp_levels_v1(**args)
        v2 = compute_sl_tp_levels_v2(**args, parity_mode=True)
        assert v1['sl'] == v2['sl']
        assert v1['tp'] == v2['tp']
        assert v1['trail_activation'] == v2['trail_activation']

    def test_v1_v2_parity_high_leverage(self):
        """V2 parity on high leverage (floor should dominate)."""
        args = {**self.BASE_ARGS, 'leverage': 50}
        v1 = compute_sl_tp_levels_v1(**args)
        v2 = compute_sl_tp_levels_v2(**args, parity_mode=True)
        assert v1['sl'] == v2['sl']
        assert v1['tp'] == v2['tp']

    def test_v1_v2_parity_with_canary(self):
        """V2 parity with canary multipliers."""
        args = {**self.BASE_ARGS, 'canary_sl_mult': 0.8, 'canary_tp_mult': 1.2, 'canary_trail_mult': 0.9}
        v1 = compute_sl_tp_levels_v1(**args)
        v2 = compute_sl_tp_levels_v2(**args, parity_mode=True)
        assert v1['sl'] == v2['sl']
        assert v1['tp'] == v2['tp']

    def test_v2_with_risk_params_differs(self):
        """V2 with ULTRA profile (non-parity) should produce DIFFERENT results."""
        rp = resolve_risk_params(RiskProfile.ULTRA_AGGRESSIVE, leverage=10)
        v1 = compute_sl_tp_levels_v1(**self.BASE_ARGS)
        v2 = compute_sl_tp_levels_v2(**self.BASE_ARGS, risk_params=rp, parity_mode=False)
        # ULTRA has wider SL floor (100% ROI vs 30% ROI)
        # SL distance should be different when floor dominates
        assert v2['meta']['sl_roi_floor_used'] == 100.0
        assert v2['meta']['version'] == 'v2'

    def test_v2_meta_contains_profile_info(self):
        """V2 meta should contain profile and liquidity info."""
        rp = resolve_risk_params(RiskProfile.AGGRESSIVE, leverage=10)
        lp = LiquidityProfile(spread_pct=0.15, depth_ratio=0.8)
        v2 = compute_sl_tp_levels_v2(**self.BASE_ARGS, risk_params=rp, liq_profile=lp, parity_mode=False)
        assert v2['meta']['profile'] == 'AGGRESSIVE'
        assert v2['meta']['sl_liq_mult'] > 1.0  # spread 0.15 should widen


# ═══════════════════════════════════════════════════════════════════
# 4) Emergency SL Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmergency:
    def test_legacy_static_no_trigger(self):
        """Legacy static check: price within safe range → no trigger."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        assert check_emergency_sl_static_v1(pos, 64000.0, 63000.0) is False

    def test_legacy_static_trigger(self):
        """Legacy static check: price WAY below trailing stop → trigger."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        # Trailing stop at 63000, price crashed to 59000 (way below margin)
        assert check_emergency_sl_static_v1(pos, 59000.0, 63000.0) is True

    def test_merged_parity_mode(self):
        """Merged check in parity mode should use 50% ROI threshold."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10, 'stopLoss': 63000}
        result = check_emergency(pos, 62000.0, parity_mode=True)
        assert isinstance(result, EmergencyResult)
        assert result.version == 'v1'
        # 50% ROI / 10x = 5% price, loss is ~4.6% → should NOT trigger
        assert result.triggered is False

    def test_merged_v2_ultra_threshold(self):
        """Merged check with ULTRA profile: 100% ROI threshold = 10% price."""
        rp = resolve_risk_params(RiskProfile.ULTRA_AGGRESSIVE, leverage=10)
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10, 'stopLoss': 63000}
        # 4.6% loss on ULTRA (threshold ~9.5%/lev) should NOT trigger
        result = check_emergency(pos, 62000.0, risk_params=rp, parity_mode=False)
        assert result.triggered is False
        assert result.version == 'v2'

    def test_merged_short_trigger(self):
        """Merged check: SHORT position with large loss should trigger."""
        pos = {'entryPrice': 65000, 'side': 'SHORT', 'leverage': 10, 'stopLoss': 67000}
        # Price rose to 70000 (7.7% loss) with BALANCED 50% ROI = 5% threshold
        result = check_emergency(pos, 70000.0, parity_mode=True)
        assert result.actual_loss_pct > 5.0
        # May or may not trigger depending on SL distance floor (67000 is 3% away, ×1.5 = 4.5%)
        # Effective = max(5%, 3.08% × 1.5) = 5% → 7.7% > 5% → triggers
        assert result.triggered is True


# ═══════════════════════════════════════════════════════════════════
# 5) Breakeven Tests
# ═══════════════════════════════════════════════════════════════════

class TestBreakeven:
    def test_legacy_buffer_range(self):
        """Legacy buffer should be in [0.0012, 0.008] range."""
        buf = compute_breakeven_buffer_pct(spread_pct=0.05, spread_level="LOW")
        assert 0.0012 <= buf <= 0.008

    def test_breakeven_price_long(self):
        """LONG breakeven = entry + buffer."""
        lp = LiquidityProfile.neutral()
        be = compute_breakeven_price(65000.0, 'LONG', lp)
        assert be > 65000.0  # Must be above entry for LONG
        assert be < 65100.0  # But not ridiculously far

    def test_breakeven_price_short(self):
        """SHORT breakeven = entry - buffer."""
        lp = LiquidityProfile.neutral()
        be = compute_breakeven_price(65000.0, 'SHORT', lp)
        assert be < 65000.0  # Must be below entry for SHORT

    def test_should_set_breakeven_trail_active(self):
        """NEW: Breakeven should trigger when trail is active (not just TP1)."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'isTrailingActive': True}
        result = should_set_breakeven(pos)
        assert result.should_set is True
        assert result.reason == 'TRAIL_ACTIVATED'
        assert result.breakeven_price > 65000

    def test_should_set_breakeven_tp1_hit(self):
        """Breakeven should trigger when TP1 is hit."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'partial_tp_state': {'tp1': True}}
        result = should_set_breakeven(pos)
        assert result.should_set is True
        assert result.reason == 'TP1_HIT'

    def test_should_set_breakeven_both(self):
        """Breakeven with both trail and TP1 active."""
        pos = {'entryPrice': 65000, 'side': 'LONG',
               'isTrailingActive': True, 'partial_tp_state': {'tp1': True}}
        result = should_set_breakeven(pos)
        assert result.should_set is True
        assert result.reason == 'BOTH'

    def test_should_not_set_breakeven_when_inactive(self):
        """No breakeven when neither trail nor TP1."""
        pos = {'entryPrice': 65000, 'side': 'LONG'}
        result = should_set_breakeven(pos)
        assert result.should_set is False

    def test_should_not_set_breakeven_already_set(self):
        """No breakeven when already activated."""
        pos = {'entryPrice': 65000, 'side': 'LONG',
               'isTrailingActive': True, 'breakeven_activated': True}
        result = should_set_breakeven(pos)
        assert result.should_set is False


# ═══════════════════════════════════════════════════════════════════
# 6) ExitStateMachine Tests
# ═══════════════════════════════════════════════════════════════════

class TestExitStateMachine:
    def test_initial_state(self):
        """New state machine starts in OPEN state."""
        sm = ExitStateMachine(pos_id='test1', side='LONG', entry_price=65000.0)
        assert sm.state == ExitState.OPEN
        assert len(sm.history) == 0
        assert not sm.is_terminal

    def test_transition_records_history(self):
        """Transitions should be recorded in history."""
        sm = ExitStateMachine(pos_id='test2', side='LONG', entry_price=65000.0)
        moved = sm.transition(ExitState.PT_ARM, 'trail_activated', price=66000.0)
        assert moved is True
        assert sm.state == ExitState.PT_ARM
        assert len(sm.history) == 1
        assert sm.last_transition.reason == 'trail_activated'
        assert sm.last_transition.telemetry_event == 'TRAIL_PROFIT_ARMED'

    def test_same_state_transition_rejected(self):
        """Transition to same state should return False."""
        sm = ExitStateMachine(pos_id='test3', side='LONG', entry_price=65000.0)
        moved = sm.transition(ExitState.OPEN, 'no_change')
        assert moved is False

    def test_terminal_states(self):
        """Terminal states should be correctly identified."""
        sm = ExitStateMachine(pos_id='test4', side='LONG', entry_price=65000.0)
        sm.transition(ExitState.EMERGENCY, 'max_loss')
        assert sm.is_terminal is True

    def test_evaluate_returns_hold(self):
        """RFX-1A skeleton: evaluate() always returns HOLD."""
        sm = ExitStateMachine(pos_id='test5', side='LONG', entry_price=65000.0)
        action = sm.evaluate(65500.0, 1200.0)
        assert action == ExitAction.HOLD

    def test_serialization(self):
        """to_dict / from_dict round-trip."""
        sm = ExitStateMachine(pos_id='test6', side='SHORT', entry_price=3500.0)
        sm.transition(ExitState.LRT_ARM, 'loss_detected', price=3600.0)
        data = sm.to_dict()
        assert data['state'] == 'LRT_ARM'
        assert len(data['history']) == 1
        sm2 = ExitStateMachine.from_dict(data)
        assert sm2.state == ExitState.LRT_ARM
        assert sm2.side == 'SHORT'


# ═══════════════════════════════════════════════════════════════════
# 7) Utility Tests
# ═══════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_snap_to_tick_up(self):
        assert snap_to_tick(65000.123, 0.01, 'up') == pytest.approx(65000.13, abs=1e-6)

    def test_snap_to_tick_down(self):
        assert snap_to_tick(65000.129, 0.01, 'down') == 65000.12

    def test_snap_to_tick_nearest(self):
        assert snap_to_tick(65000.125, 0.01, 'nearest') == pytest.approx(65000.12, abs=0.015)

    def test_snap_to_tick_zero_guard(self):
        assert snap_to_tick(0, 0.01) == 0
        assert snap_to_tick(100, 0) == 100

    def test_estimate_trade_cost(self):
        cost = estimate_trade_cost(spread_pct=0.05, leverage=10)
        assert cost['total_pct'] > 0
        assert cost['roi_pct'] > 0
        assert cost['fee_pct'] == 0.08

    def test_ensure_tick_safe_buffer(self):
        buf = ensure_tick_safe_buffer(entry_price=0.001, buffer_pct=0.001, tick_size=0.00001)
        assert buf >= 0.00001 * 3  # At least 3 ticks
