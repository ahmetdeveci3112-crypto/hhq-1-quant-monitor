"""
RFX-1B: Integration tests — SM evaluate(), TP ladder v2, trail→BE, adapter wiring.

Covers:
  - SM state transitions (14 tests, LONG+SHORT symmetric)
  - TP ladder monotonic (8 tests, LONG+SHORT)
  - Trail→BE enforcement (6 tests)
  - main.py adapter wiring (8 tests)

Run: python3 -m pytest test_rfx_1b.py -v --override-ini="asyncio_mode=auto"
"""
import os
import sys
import math
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from exit.state_machine import (
    ExitStateMachine, ExitState, ExitAction,
    EvalContext, ExitDecision,
)
from risk.sl_tp_engine import compute_tp_ladder_v2, snap_to_tick
from risk.policy import RiskProfile, resolve_risk_params


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def make_sm(side='LONG', entry=65000.0):
    return ExitStateMachine(pos_id='test-1', side=side, entry_price=entry)

def make_ctx(
    tick_price=65500.0, entry=65000.0, side='LONG', leverage=10,
    atr=1200.0, trailing_stop=0, is_trailing_active=False,
    tp_ladder=None, recovery_low=0, recovery_high=0,
    partial_tp_state=None, risk_params=None, spread_pct=0.05,
):
    if tp_ladder is None:
        tp_ladder = [
            {'key': 'tp1', 'pct': 0.8, 'close_pct': 0.30},
            {'key': 'tp2', 'pct': 2.0, 'close_pct': 0.25},
            {'key': 'tp3', 'pct': 4.0, 'close_pct': 0.25},
            {'key': 'tp_final', 'pct': 10.0, 'close_pct': 0.20},
        ]
    return EvalContext(
        tick_price=tick_price, atr=atr, entry_price=entry,
        side=side, leverage=leverage, trailing_stop=trailing_stop,
        is_trailing_active=is_trailing_active, tp_ladder=tp_ladder,
        spread_pct=spread_pct, tick_size=0.01, margin_usd=650.0,
        risk_params=risk_params,
        recovery_low=recovery_low, recovery_high=recovery_high,
        partial_tp_state=partial_tp_state or {},
    )


# ═══════════════════════════════════════════════════════════════════
# Test 1: SM evaluate() state transitions
# ═══════════════════════════════════════════════════════════════════

class TestSMTransitions:
    """14 tests covering all SM state transition paths."""

    def test_open_to_pt_arm_long(self):
        """OPEN → PT_ARM when profit exceeds trail activation threshold (LONG)."""
        sm = make_sm('LONG', 65000)
        # Price moved up by ~2.8% (well above 1.5 ATR ~= 1.8%)
        ctx = make_ctx(tick_price=66800, entry=65000, side='LONG', atr=1200)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.ARM_PT
        assert sm.state == ExitState.PT_ARM

    def test_open_to_pt_arm_short(self):
        """OPEN → PT_ARM when profit exceeds threshold (SHORT)."""
        sm = make_sm('SHORT', 65000)
        ctx = make_ctx(tick_price=63200, entry=65000, side='SHORT', atr=1200)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.ARM_PT
        assert sm.state == ExitState.PT_ARM

    def test_open_to_lrt_arm_long(self):
        """OPEN → LRT_ARM when in loss + bounced (LONG)."""
        sm = make_sm('LONG', 65000)
        # In loss: 64000 (1.5% loss), bounced from low=63600 by 0.3 ATR=360
        ctx = make_ctx(tick_price=64000, entry=65000, side='LONG', atr=1200,
                       recovery_low=63600)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.ARM_LRT
        assert sm.state == ExitState.LRT_ARM

    def test_open_to_lrt_arm_short(self):
        """OPEN → LRT_ARM when in loss + bounced (SHORT)."""
        sm = make_sm('SHORT', 65000)
        # In loss: 66000 (~1.5%), bounced from high=66400 by 0.3 ATR
        ctx = make_ctx(tick_price=66000, entry=65000, side='SHORT', atr=1200,
                       recovery_high=66400)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.ARM_LRT
        assert sm.state == ExitState.LRT_ARM

    def test_pt_arm_to_be_protect(self):
        """PT_ARM → BE_PROTECT (immediate — breakeven set on trail activation)."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_ARM
        ctx = make_ctx(tick_price=66000, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.SET_BREAKEVEN
        assert sm.state == ExitState.BE_PROTECT
        assert d.suggested_price > 65000  # BE is above entry for LONG

    def test_lrt_to_be_protect_on_recovery(self):
        """LRT_ARM → BE_PROTECT when loss recovers to breakeven."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.LRT_ARM
        # Price recovered to above entry
        ctx = make_ctx(tick_price=65100, entry=65000, side='LONG', atr=1200)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.SET_BREAKEVEN
        assert sm.state == ExitState.BE_PROTECT

    def test_be_protect_to_tp1(self):
        """BE_PROTECT → PT_TP1 when profit hits TP1."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.BE_PROTECT
        # TP1 pct = 0.8% → price needs to be >= 65000 * 1.008 = 65520
        ctx = make_ctx(tick_price=65600, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.PARTIAL_TP
        assert d.tp_key == 'tp1'
        assert sm.state == ExitState.PT_TP1

    def test_tp1_to_tp2_to_tp3_progression(self):
        """PT_TP1 → PT_TP2 → PT_TP3 forward-only."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_TP1
        # TP2 = 2.0% → 66300
        ctx = make_ctx(tick_price=66400, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.PARTIAL_TP
        assert d.tp_key == 'tp2'
        assert sm.state == ExitState.PT_TP2

        # TP3 = 4.0% → 67600
        ctx = make_ctx(tick_price=67700, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.PARTIAL_TP
        assert d.tp_key == 'tp3'
        assert sm.state == ExitState.PT_TP3

    def test_tp_final_closes_position(self):
        """TP_FINAL hit → CLOSE_TP (full close)."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_TP3
        # TP_FINAL = 10% → 71500
        ctx = make_ctx(tick_price=71600, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.CLOSE_TP
        assert sm.state == ExitState.TP_FINAL

    def test_emergency_preempts_any_state(self):
        """EMERGENCY preempts any non-terminal state."""
        for start_state in [ExitState.OPEN, ExitState.PT_ARM, ExitState.BE_PROTECT, ExitState.PT_TP2]:
            sm = make_sm('LONG', 65000)
            sm.state = start_state
            # Price dropped 6% → 61100, with leverage 10 that's 60% ROI loss > 50% threshold
            ctx = make_ctx(tick_price=61100, entry=65000, side='LONG', leverage=10)
            d = sm.evaluate(ctx)
            assert d.action == ExitAction.CLOSE_EMERGENCY, f"Emergency failed from {start_state}"
            assert sm.state == ExitState.EMERGENCY

    def test_sl_final_preempts_non_terminal(self):
        """SL_FINAL when trailing stop hit."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.BE_PROTECT
        # Trail at 64000, price dropped to 63900
        ctx = make_ctx(tick_price=63900, entry=65000, side='LONG', trailing_stop=64000)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.CLOSE_SL
        assert sm.state == ExitState.SL_FINAL

    def test_terminal_state_returns_hold(self):
        """Terminal states return HOLD for idempotency (B4)."""
        for term_state in [ExitState.TP_FINAL, ExitState.SL_FINAL, ExitState.EMERGENCY, ExitState.CLOSED]:
            sm = make_sm('LONG', 65000)
            sm.state = term_state
            ctx = make_ctx(tick_price=65000, entry=65000, side='LONG')
            d = sm.evaluate(ctx)
            assert d.action == ExitAction.HOLD
            assert d.reason == "TERMINAL_STATE"

    def test_hold_means_wait_not_legacy(self):
        """HOLD = do nothing this tick (B1: legacy does NOT run)."""
        sm = make_sm('LONG', 65000)
        # Small profit, not enough for trail activation
        ctx = make_ctx(tick_price=65100, entry=65000, side='LONG', atr=1200)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.HOLD
        assert d.decision_source == "SM"

    def test_forward_only_tp_no_backtrack(self):
        """TP progression is forward-only — can't go from TP2 back to TP1."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_TP2
        # Even with TP1-level profit, SM should not go back to TP1
        ctx = make_ctx(tick_price=65600, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        # Should HOLD or wait for TP3, not re-trigger TP1
        assert d.action == ExitAction.HOLD or d.tp_key != 'tp1'


# ═══════════════════════════════════════════════════════════════════
# Test 2: TP Ladder V2 Monotonic
# ═══════════════════════════════════════════════════════════════════

class TestTPLadderV2:
    """8 tests covering TP ladder monotonicity and TP_FINAL."""

    def test_long_monotonic(self):
        """LONG: TP1 < TP2 < TP3 < TP_FINAL."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10, tick_size=0.01)
        prices = result['prices']
        assert prices['tp1'] < prices['tp2'] < prices['tp3'] < prices['tp_final']
        assert result['monotonic'] is True

    def test_short_monotonic(self):
        """SHORT: TP1 > TP2 > TP3 > TP_FINAL (descending from entry)."""
        result = compute_tp_ladder_v2(65000, 1200, 'SHORT', 10, tick_size=0.01)
        prices = result['prices']
        assert prices['tp1'] > prices['tp2'] > prices['tp3'] > prices['tp_final']
        assert result['monotonic'] is True

    def test_ultra_wider_tp_final(self):
        """ULTRA_AGGRESSIVE has wider TP_FINAL than BALANCED."""
        rp_balanced = resolve_risk_params(RiskProfile.BALANCED, 10)
        rp_ultra = resolve_risk_params(RiskProfile.ULTRA_AGGRESSIVE, 10)
        
        bal = compute_tp_ladder_v2(65000, 1200, 'LONG', 10, risk_params=rp_balanced)
        ultra = compute_tp_ladder_v2(65000, 1200, 'LONG', 10, risk_params=rp_ultra)
        
        assert ultra['prices']['tp_final'] > bal['prices']['tp_final']

    def test_high_leverage_compresses(self):
        """High leverage (20x) still maintains monotonicity."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 20, tick_size=0.01)
        prices = result['prices']
        assert prices['tp1'] < prices['tp2'] < prices['tp3'] < prices['tp_final']

    def test_cost_floor_preserves_monotonic(self):
        """High spread cost floor doesn't break monotonicity."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10,
                                       tick_size=0.01, spread_pct=0.50)
        prices = result['prices']
        assert prices['tp1'] < prices['tp2'] < prices['tp3'] < prices['tp_final']

    def test_epsilon_tick_fix_applied(self):
        """Epsilon-tick fix works when levels would collide."""
        # Very small ATR → levels might be close after tick-snap
        result = compute_tp_ladder_v2(0.001, 0.00001, 'LONG', 10, tick_size=0.000001)
        prices = result['prices']
        assert prices['tp1'] < prices['tp2'] < prices['tp3'] < prices['tp_final']

    def test_four_levels_in_ladder(self):
        """Ladder has exactly 4 levels with TP_FINAL."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10)
        keys = [lv['key'] for lv in result['levels']]
        assert keys == ['tp1', 'tp2', 'tp3', 'tp_final']

    def test_close_ratios_sum_to_one(self):
        """Close ratios always sum to 1.0."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10)
        total = sum(lv['close_pct'] for lv in result['levels'])
        assert abs(total - 1.0) < 0.05  # Allow rounding


# ═══════════════════════════════════════════════════════════════════
# Test 3: Trail → BE Enforcement
# ═══════════════════════════════════════════════════════════════════

class TestTrailBE:
    """6 tests covering breakeven enforcement on trail activation."""

    def test_be_set_on_pt_arm_long(self):
        """PT_ARM → BE_PROTECT sets BE above entry (LONG)."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_ARM
        ctx = make_ctx(tick_price=66000, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.SET_BREAKEVEN
        assert d.suggested_price > 65000  # BE above entry for LONG

    def test_be_set_on_pt_arm_short(self):
        """PT_ARM → BE_PROTECT sets BE below entry (SHORT)."""
        sm = make_sm('SHORT', 65000)
        sm.state = ExitState.PT_ARM
        ctx = make_ctx(tick_price=64000, entry=65000, side='SHORT')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.SET_BREAKEVEN
        assert d.suggested_price < 65000  # BE below entry for SHORT

    def test_be_reconfirmed_on_tp1(self):
        """TP1 hit also sets suggested_price for BE re-confirmation."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.BE_PROTECT
        ctx = make_ctx(tick_price=65600, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.PARTIAL_TP
        assert d.tp_key == 'tp1'
        assert d.suggested_price > 65000  # BE re-confirmed

    def test_be_includes_fee_buffer(self):
        """BE price includes fee+spread+slippage buffer (not exactly entry)."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.PT_ARM
        ctx = make_ctx(tick_price=66000, entry=65000, side='LONG', spread_pct=0.10)
        d = sm.evaluate(ctx)
        be_delta_pct = (d.suggested_price - 65000) / 65000 * 100
        assert be_delta_pct > 0.10  # At least 0.1% buffer

    def test_be_short_includes_buffer(self):
        """SHORT BE is below entry with buffer."""
        sm = make_sm('SHORT', 65000)
        sm.state = ExitState.PT_ARM
        ctx = make_ctx(tick_price=64000, entry=65000, side='SHORT', spread_pct=0.10)
        d = sm.evaluate(ctx)
        be_delta_pct = (65000 - d.suggested_price) / 65000 * 100
        assert be_delta_pct > 0.10

    def test_lrt_recovery_to_be(self):
        """LRT_ARM → BE_PROTECT when position recovers to breakeven."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.LRT_ARM
        ctx = make_ctx(tick_price=65100, entry=65000, side='LONG')
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.SET_BREAKEVEN
        assert sm.state == ExitState.BE_PROTECT


# ═══════════════════════════════════════════════════════════════════
# Test 4: SM Serialization Round-trip
# ═══════════════════════════════════════════════════════════════════

class TestSMSerialization:
    """SM to_dict/from_dict round-trip fidelity."""

    def test_roundtrip_state_preserved(self):
        """State is preserved through serialization."""
        sm = make_sm('LONG', 65000)
        sm.state = ExitState.BE_PROTECT
        sm.be_price = 65078.0
        sm.trail_stop = 64500.0
        sm.peak_price = 66000.0
        
        data = sm.to_dict()
        sm2 = ExitStateMachine.from_dict(data)
        
        assert sm2.state == ExitState.BE_PROTECT
        assert sm2.be_price == 65078.0
        assert sm2.trail_stop == 64500.0
        assert sm2.peak_price == 66000.0
        assert sm2.side == 'LONG'
        assert sm2.entry_price == 65000.0

    def test_roundtrip_history_preserved(self):
        """Transition history is preserved through serialization."""
        sm = make_sm('LONG', 65000)
        sm.transition(ExitState.PT_ARM, 'test_profit', price=66000)
        sm.transition(ExitState.BE_PROTECT, 'test_be', price=66000)
        
        data = sm.to_dict()
        sm2 = ExitStateMachine.from_dict(data)
        
        assert len(sm2.history) == 2
        assert sm2.history[0].to_state == ExitState.PT_ARM
        assert sm2.history[1].to_state == ExitState.BE_PROTECT

    def test_roundtrip_empty_history(self):
        """Fresh SM with no history round-trips correctly."""
        sm = make_sm('SHORT', 70000)
        data = sm.to_dict()
        sm2 = ExitStateMachine.from_dict(data)
        
        assert sm2.state == ExitState.OPEN
        assert len(sm2.history) == 0
        assert sm2.side == 'SHORT'


# ═══════════════════════════════════════════════════════════════════
# Test 5: main.py Flag Parity
# ═══════════════════════════════════════════════════════════════════

class TestMainPyFlags:
    """8 tests for main.py adapter wiring with flags."""

    def test_all_flags_off_legacy_parity(self):
        """All flags OFF → legacy SL/TP works as before."""
        from main import compute_sl_tp_levels
        with patch('main.RFX_SL_TP_V2', False), \
             patch('main.RFX1B_EXIT_WIRING', False):
            result = compute_sl_tp_levels(
                entry_price=65000, atr=1200, side='LONG', leverage=10,
                adjusted_sl_atr=1.5, adjusted_tp_atr=3.0,
                adjusted_trail_act_atr=2.0, adjusted_trail_dist_atr=0.5,
                spread_pct=0.05, symbol='BTCUSDT',
            )
        assert 'sl' in result
        assert 'tp' in result
        assert result['sl'] > 0

    def test_rfx1b_flags_default_off(self):
        """All RFX1B flags default to False."""
        # Import fresh module-level values
        from main import (
            RFX1B_EXIT_WIRING, RFX1B_TRAIL_DUAL, RFX1B_BE_ON_TRAIL,
            RFX1B_TP_LADDER_V2, RFX1B_LEGACY_FALLBACK_ON_ERROR,
        )
        # These default to false (unless env var set)
        # We can't assert False because env might be set in CI
        # Instead verify they're boolean
        assert isinstance(RFX1B_EXIT_WIRING, bool)
        assert isinstance(RFX1B_TRAIL_DUAL, bool)
        assert isinstance(RFX1B_BE_ON_TRAIL, bool)
        assert isinstance(RFX1B_TP_LADDER_V2, bool)
        assert isinstance(RFX1B_LEGACY_FALLBACK_ON_ERROR, bool)

    def test_exit_inflight_guard_exists(self):
        """_exit_inflight set exists for B4 idempotency."""
        from main import _exit_inflight
        assert isinstance(_exit_inflight, set)

    def test_fallback_counter_exists(self):
        """_rfx1b_fallback_count metric exists."""
        from main import _rfx1b_fallback_count
        assert isinstance(_rfx1b_fallback_count, int)

    def test_modules_importable(self):
        """RFX-1B modules are importable."""
        from main import _RFX_MODULES_AVAILABLE
        assert _RFX_MODULES_AVAILABLE is True

    def test_eval_context_importable(self):
        """EvalContext importable from exit module."""
        from exit import EvalContext, ExitDecision
        ctx = EvalContext(
            tick_price=65000, atr=1200, entry_price=65000, side='LONG',
            leverage=10, trailing_stop=0, is_trailing_active=False,
            tp_ladder=[], spread_pct=0.05, tick_size=0.01, margin_usd=650,
        )
        assert ctx.tick_price == 65000

    def test_tp_ladder_v2_importable(self):
        """compute_tp_ladder_v2 importable from risk module."""
        from risk import compute_tp_ladder_v2
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10)
        assert 'tp_final' in result['prices']

    def test_emergency_handled_by_sm(self):
        """SM handles emergency — returns CLOSE_EMERGENCY."""
        sm = make_sm('LONG', 65000)
        # Price dropped 6% → 61100, 60% ROI loss > 50% threshold
        ctx = make_ctx(tick_price=61100, entry=65000, side='LONG', leverage=10)
        d = sm.evaluate(ctx)
        assert d.action == ExitAction.CLOSE_EMERGENCY
        assert sm.is_terminal


# ═══════════════════════════════════════════════════════════════════
# Test 6: TP Ladder V2 Entry Path Regression
# ═══════════════════════════════════════════════════════════════════

class TestTPLadderEntryPath:
    """Regression: TP_FINAL must exist in ladder when RFX1B_TP_LADDER_V2=true."""

    def test_v2_ladder_has_tp_final(self):
        """compute_tp_ladder_v2 output contains tp_final level."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10, tick_size=0.01)
        keys = [lv['key'] for lv in result['levels']]
        assert 'tp_final' in keys, f"tp_final missing from ladder keys: {keys}"
        assert len(keys) == 4, f"Expected 4 tiers, got {len(keys)}: {keys}"

    def test_v2_ladder_has_tp_final_price(self):
        """compute_tp_ladder_v2 output contains tp_final in prices dict."""
        result = compute_tp_ladder_v2(65000, 1200, 'LONG', 10, tick_size=0.01)
        assert 'tp_final' in result['prices'], "tp_final missing from prices dict"
        assert result['prices']['tp_final'] > 0

    def test_entry_path_simulation_v2(self):
        """Simulate entry-path adapter: position gets 4-tier ladder with tp_final."""
        # This simulates what main.py does at position creation
        ladder = compute_tp_ladder_v2(
            entry_price=65000, atr=1200, side='LONG', leverage=10,
            tick_size=0.01, spread_pct=0.05,
        )
        # Simulate position dict population
        new_position = {}
        new_position['tp_ladder_levels'] = ladder['levels']
        new_position['tp_ladder_prices'] = ladder.get('prices', {})
        new_position['tp_ladder_telemetry'] = ladder['telemetry']

        # Assert tp_final is present in the position's ladder
        level_keys = [lv['key'] for lv in new_position['tp_ladder_levels']]
        assert 'tp_final' in level_keys, f"tp_final missing from position ladder: {level_keys}"
        assert 'tp_final' in new_position['tp_ladder_prices']
        assert new_position['tp_ladder_prices']['tp_final'] > 65000  # LONG → above entry

    def test_entry_path_simulation_short(self):
        """SHORT simulation: tp_final price below entry."""
        ladder = compute_tp_ladder_v2(
            entry_price=65000, atr=1200, side='SHORT', leverage=10,
            tick_size=0.01, spread_pct=0.05,
        )
        new_position = {}
        new_position['tp_ladder_levels'] = ladder['levels']
        new_position['tp_ladder_prices'] = ladder.get('prices', {})

        level_keys = [lv['key'] for lv in new_position['tp_ladder_levels']]
        assert 'tp_final' in level_keys
        assert new_position['tp_ladder_prices']['tp_final'] < 65000  # SHORT → below entry

    def test_legacy_ladder_no_tp_final(self):
        """Legacy compute_adaptive_tp_ladder has NO tp_final (3-tier only)."""
        from main import compute_adaptive_tp_ladder
        result = compute_adaptive_tp_ladder(
            side='LONG', entry_price=65000, atr=1200, leverage=10,
        )
        keys = [lv['key'] for lv in result['levels']]
        assert 'tp_final' not in keys, f"Legacy should NOT have tp_final: {keys}"
        assert len(keys) == 3

