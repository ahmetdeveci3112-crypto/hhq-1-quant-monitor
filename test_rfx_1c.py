"""
RFX-1C: Order-size-aware thin-book depth gate tests.

Covers:
  - compute_order_impact: small vs large orders, side-aware depth, floors, edges
  - Shadow/enforce mode behavior
  - Critical acceptance: same coin, small order PASS / large order BLOCK
  - Side-depth asymmetry (LONG=ask, SHORT=bid)

Run: python3 -m pytest test_rfx_1c.py -v --override-ini="asyncio_mode=auto"
"""
import os
import sys
import math
import inspect

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from risk.depth_gate import compute_order_impact, DepthImpact, estimate_slippage_bps


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

DEFAULT_FLOOR = 5000.0
DEFAULT_MAX_IMPACT = 0.10  # 10%


# ═══════════════════════════════════════════════════════════════════
# Test 1: Core Impact Computation
# ═══════════════════════════════════════════════════════════════════

class TestComputeOrderImpact:
    """Basic impact ratio computation and decision logic."""

    def test_small_order_thick_book_passes(self):
        """$50 margin × 10x = $500 notional on $50K depth → 1% impact → PASS."""
        result = compute_order_impact(
            planned_margin_usd=50, leverage=10, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=50000,
            max_impact_pct=DEFAULT_MAX_IMPACT, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        assert result.passes is True
        assert result.impact_ratio == pytest.approx(0.01, abs=0.001)
        assert result.reason == 'PASS'
        assert result.telemetry_tag == 'RFX1C_PASS'

    def test_large_order_thin_book_blocks(self):
        """$500 margin × 10x = $5000 notional on $8K depth → 62.5% impact → BLOCK."""
        result = compute_order_impact(
            planned_margin_usd=500, leverage=10, side='LONG',
            bid_depth_usd=8000, ask_depth_usd=8000,
            max_impact_pct=DEFAULT_MAX_IMPACT, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        assert result.passes is False
        assert result.impact_ratio > 0.5
        assert result.reason == 'BLOCK_IMPACT'
        assert result.telemetry_tag == 'RFX1C_BLOCK'

    def test_same_depth_different_sizes_divergent_verdict(self):
        """CRITICAL ACCEPTANCE: Same coin/depth, small order passes, large order blocks."""
        depth_bid = 10000
        depth_ask = 10000
        
        # Small: $20 margin × 10x = $200 notional on $10K ask → 2% → PASS
        small = compute_order_impact(
            planned_margin_usd=20, leverage=10, side='LONG',
            bid_depth_usd=depth_bid, ask_depth_usd=depth_ask,
            max_impact_pct=DEFAULT_MAX_IMPACT, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        # Large: $500 margin × 10x = $5000 notional on $10K ask → 50% → BLOCK
        large = compute_order_impact(
            planned_margin_usd=500, leverage=10, side='LONG',
            bid_depth_usd=depth_bid, ask_depth_usd=depth_ask,
            max_impact_pct=DEFAULT_MAX_IMPACT, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        
        assert small.passes is True, f"Small order should pass, got {small.reason}"
        assert large.passes is False, f"Large order should block, got {large.reason}"

    def test_impact_ratio_correct(self):
        """impact_ratio = notional / used_depth."""
        result = compute_order_impact(
            planned_margin_usd=100, leverage=20, side='LONG',
            bid_depth_usd=10000, ask_depth_usd=20000,
        )
        # Notional = 100 × 20 = $2000, LONG uses ask = $20K
        expected = 2000 / 20000  # 0.10
        assert result.impact_ratio == pytest.approx(expected, abs=0.001)

    def test_notional_formula(self):
        """planned_notional_usd = planned_margin × leverage."""
        result = compute_order_impact(
            planned_margin_usd=75, leverage=15, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=50000,
        )
        assert result.planned_notional_usd == pytest.approx(75 * 15, abs=0.1)
        assert result.planned_margin_usd == pytest.approx(75, abs=0.1)


# ═══════════════════════════════════════════════════════════════════
# Test 2: Side-Aware Depth
# ═══════════════════════════════════════════════════════════════════

class TestSideAwareDepth:
    """LONG uses ask depth, SHORT uses bid depth."""

    def test_long_uses_ask_depth(self):
        """LONG order should use ask-side depth."""
        result = compute_order_impact(
            planned_margin_usd=100, leverage=10, side='LONG',
            bid_depth_usd=100000, ask_depth_usd=5000,
        )
        assert result.used_depth_usd == pytest.approx(5000, abs=1)
        assert result.side == 'LONG'
        # $1000 notional / $5K ask = 20% → BLOCK
        assert result.impact_ratio == pytest.approx(0.20, abs=0.01)

    def test_short_uses_bid_depth(self):
        """SHORT order should use bid-side depth."""
        result = compute_order_impact(
            planned_margin_usd=100, leverage=10, side='SHORT',
            bid_depth_usd=5000, ask_depth_usd=100000,
        )
        assert result.used_depth_usd == pytest.approx(5000, abs=1)
        assert result.side == 'SHORT'
        # $1000 notional / $5K bid = 20% → BLOCK
        assert result.impact_ratio == pytest.approx(0.20, abs=0.01)

    def test_asymmetric_depth_long_block_short_pass(self):
        """CRITICAL: Low ask (LONG blocks), high bid (SHORT passes)."""
        # Ask only $2K, bid $50K
        long_result = compute_order_impact(
            planned_margin_usd=50, leverage=10, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=2000,
            max_impact_pct=0.10, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        short_result = compute_order_impact(
            planned_margin_usd=50, leverage=10, side='SHORT',
            bid_depth_usd=50000, ask_depth_usd=2000,
            max_impact_pct=0.10, min_depth_floor_usd=DEFAULT_FLOOR,
        )
        # LONG: $500 / $2K = 25% → BLOCK (also floor: $2K < $5K)
        assert long_result.passes is False
        # SHORT: $500 / $50K = 1% → PASS
        assert short_result.passes is True

    def test_total_depth_in_telemetry(self):
        """total_depth_usd is bid + ask (for telemetry only)."""
        result = compute_order_impact(
            planned_margin_usd=50, leverage=10, side='LONG',
            bid_depth_usd=30000, ask_depth_usd=20000,
        )
        assert result.total_depth_usd == pytest.approx(50000, abs=1)


# ═══════════════════════════════════════════════════════════════════
# Test 3: Floor Enforcement
# ═══════════════════════════════════════════════════════════════════

class TestFloorEnforcement:
    """Absolute minimum depth floor regardless of order size."""

    def test_floor_blocks_below_minimum(self):
        """Even tiny order blocked if depth < floor."""
        result = compute_order_impact(
            planned_margin_usd=5, leverage=5, side='LONG',
            bid_depth_usd=4000, ask_depth_usd=4000,  # Below $5K floor
            max_impact_pct=1.0,  # Very permissive impact
            min_depth_floor_usd=5000,
        )
        assert result.passes is False
        assert result.reason == 'BLOCK_FLOOR'
        assert result.depth_sufficient is False

    def test_floor_pass_above_minimum(self):
        """Depth above floor + low impact → passes."""
        result = compute_order_impact(
            planned_margin_usd=10, leverage=5, side='LONG',
            bid_depth_usd=6000, ask_depth_usd=6000,
            max_impact_pct=0.10,
            min_depth_floor_usd=5000,
        )
        # $50 notional / $6K depth = 0.8% → PASS
        assert result.passes is True
        assert result.depth_sufficient is True
        assert result.impact_within_budget is True

    def test_floor_zero_disables_floor_check(self):
        """Floor=0 means depth_sufficient is always True (impact is only gate)."""
        result = compute_order_impact(
            planned_margin_usd=10, leverage=5, side='LONG',
            bid_depth_usd=100, ask_depth_usd=100,
            max_impact_pct=1.0,  # Very permissive
            min_depth_floor_usd=0.0,
        )
        # 0.0 floor → depth >= 0 → depth_sufficient = True
        assert result.depth_sufficient is True
        assert result.passes is True

    def test_custom_floor_respected(self):
        """Custom floor $8K should block on $7K depth."""
        result = compute_order_impact(
            planned_margin_usd=10, leverage=5, side='LONG',
            bid_depth_usd=7000, ask_depth_usd=7000,
            max_impact_pct=1.0,
            min_depth_floor_usd=8000,
        )
        assert result.passes is False
        assert result.reason == 'BLOCK_FLOOR'


# ═══════════════════════════════════════════════════════════════════
# Test 4: Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Zero depth, zero notional, extreme values."""

    def test_zero_depth_always_blocks(self):
        """Zero depth → always BLOCK_ZERO_DEPTH."""
        result = compute_order_impact(
            planned_margin_usd=50, leverage=10, side='LONG',
            bid_depth_usd=0, ask_depth_usd=0,
        )
        assert result.passes is False
        assert result.reason == 'BLOCK_ZERO_DEPTH'
        assert result.impact_ratio == float('inf')

    def test_zero_notional_always_passes(self):
        """Zero margin → zero notional → 0% impact → PASS (if depth > floor)."""
        result = compute_order_impact(
            planned_margin_usd=0, leverage=10, side='LONG',
            bid_depth_usd=10000, ask_depth_usd=10000,
        )
        assert result.passes is True
        assert result.impact_ratio == pytest.approx(0.0, abs=0.001)

    def test_zero_depth_zero_notional(self):
        """Both zero → BLOCK (zero depth takes precedence)."""
        result = compute_order_impact(
            planned_margin_usd=0, leverage=10, side='LONG',
            bid_depth_usd=0, ask_depth_usd=0,
        )
        assert result.passes is False
        assert result.reason == 'BLOCK_ZERO_DEPTH'

    def test_negative_values_clamped(self):
        """Negative inputs clamped to 0."""
        result = compute_order_impact(
            planned_margin_usd=-100, leverage=-5, side='LONG',
            bid_depth_usd=-1000, ask_depth_usd=-1000,
        )
        assert result.planned_margin_usd == 0
        assert result.passes is False

    def test_very_high_leverage(self):
        """50x leverage amplifies impact correctly."""
        result = compute_order_impact(
            planned_margin_usd=100, leverage=50, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=50000,
        )
        # $5000 notional / $50K = 10% → exactly at boundary
        assert result.impact_ratio == pytest.approx(0.10, abs=0.001)
        assert result.impact_within_budget is True  # <= threshold


# ═══════════════════════════════════════════════════════════════════
# Test 5: DepthImpact Fields
# ═══════════════════════════════════════════════════════════════════

class TestDepthImpactFields:
    """All fields in DepthImpact populated correctly."""

    def test_all_fields_present(self):
        """DepthImpact has all required fields."""
        result = compute_order_impact(
            planned_margin_usd=100, leverage=10, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=40000,
            max_impact_pct=0.10, min_depth_floor_usd=5000,
        )
        assert hasattr(result, 'impact_ratio')
        assert hasattr(result, 'passes')
        assert hasattr(result, 'depth_sufficient')
        assert hasattr(result, 'impact_within_budget')
        assert hasattr(result, 'planned_notional_usd')
        assert hasattr(result, 'planned_margin_usd')
        assert hasattr(result, 'used_depth_usd')
        assert hasattr(result, 'bid_depth_usd')
        assert hasattr(result, 'ask_depth_usd')
        assert hasattr(result, 'total_depth_usd')
        assert hasattr(result, 'max_impact_pct')
        assert hasattr(result, 'min_depth_floor_usd')
        assert hasattr(result, 'side')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'telemetry_tag')

    def test_telemetry_tag_values(self):
        """Tag is either RFX1C_PASS or RFX1C_BLOCK."""
        pass_result = compute_order_impact(
            planned_margin_usd=10, leverage=5, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=50000,
        )
        block_result = compute_order_impact(
            planned_margin_usd=1000, leverage=20, side='LONG',
            bid_depth_usd=1000, ask_depth_usd=1000,
        )
        assert pass_result.telemetry_tag == 'RFX1C_PASS'
        assert block_result.telemetry_tag == 'RFX1C_BLOCK'


# ═══════════════════════════════════════════════════════════════════
# Test 6: Slippage Estimation
# ═══════════════════════════════════════════════════════════════════

class TestSlippageEstimation:
    """estimate_slippage_bps basic sanity."""

    def test_small_order_low_slippage(self):
        """Small order on deep book → low slippage."""
        bps = estimate_slippage_bps(500, 100000, 0.05)
        assert bps < 5  # Less than 5 bps

    def test_large_order_high_slippage(self):
        """Large order on thin book → high slippage."""
        bps = estimate_slippage_bps(50000, 10000, 0.20)
        assert bps > 100  # Over 100 bps


# ═══════════════════════════════════════════════════════════════════
# Test 7: main.py Flag Presence
# ═══════════════════════════════════════════════════════════════════

class TestMainPyFlags:
    """RFX-1C flags exist and have correct defaults."""

    def test_flags_exist(self):
        """All RFX-1C flags importable."""
        from main import (
            RFX1C_DEPTH_GATE_MODE,
            RFX1C_MAX_IMPACT_PCT,
            RFX1C_MIN_DEPTH_FLOOR_USD,
            RFX1C_IMPACT_TELEMETRY_SAMPLE_RATE,
        )
        assert isinstance(RFX1C_DEPTH_GATE_MODE, str)
        assert isinstance(RFX1C_MAX_IMPACT_PCT, float)
        assert isinstance(RFX1C_MIN_DEPTH_FLOOR_USD, float)
        assert isinstance(RFX1C_IMPACT_TELEMETRY_SAMPLE_RATE, float)

    def test_default_mode_off(self):
        """Default mode is 'off' (no behavioral change)."""
        from main import RFX1C_DEPTH_GATE_MODE
        # May be overridden by env, so just check it's a valid value
        assert RFX1C_DEPTH_GATE_MODE in ('off', 'shadow', 'enforce')

    def test_depth_gate_importable(self):
        """rfx_compute_order_impact importable from main's risk import."""
        from main import _RFX_MODULES_AVAILABLE
        assert _RFX_MODULES_AVAILABLE is True
        from risk import compute_order_impact, DepthImpact
        result = compute_order_impact(
            planned_margin_usd=100, leverage=10, side='LONG',
            bid_depth_usd=50000, ask_depth_usd=50000,
        )
        assert isinstance(result, DepthImpact)
        assert result.passes is True

    def test_process_signal_depth_gate_uses_global_trader_state(self):
        """Depth gate in process_signal_for_paper_trading must not reference method-scope self."""
        import main

        src = inspect.getsource(main.process_signal_for_paper_trading)
        assert 'global_paper_trader.balance' in src
        assert 'global_paper_trader.risk_per_trade' in src
        assert 'self.balance * self.risk_per_trade' not in src
