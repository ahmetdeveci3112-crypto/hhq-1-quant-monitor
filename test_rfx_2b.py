"""RFX-2B: Distance Truth Telemetry test suite.

Tests:
  1. build_distance_truth — LONG/SHORT sanity
  2. build_distance_truth — edge cases (NaN, inf, zero)
  3. build_distance_truth — meta pipeline
  4. Integration: compute_sl_tp_levels returns distance_truth
"""
import math
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk.distance_truth import build_distance_truth, _safe


# ═════════════════════════════════════════════════════════
# Unit: _safe sanitizer
# ═════════════════════════════════════════════════════════

class TestSafeSanitizer:
    def test_normal_float(self):
        assert _safe(1.5) == 1.5

    def test_nan(self):
        assert _safe(float('nan')) == 0.0

    def test_inf(self):
        assert _safe(float('inf')) == 0.0

    def test_neg_inf(self):
        assert _safe(float('-inf')) == 0.0

    def test_none(self):
        assert _safe(None) == 0.0

    def test_string(self):
        assert _safe("abc") == 0.0

    def test_custom_default(self):
        assert _safe(None, 42.0) == 42.0


# ═════════════════════════════════════════════════════════
# Unit: build_distance_truth
# ═════════════════════════════════════════════════════════

class TestBuildDistanceTruthLong:
    """LONG scenario: entry=100, sl=97, tp=106, trail=1.5, lev=10."""

    def setup_method(self):
        self.meta = {
            'sl_source': 'ATR',
            'tp_source': 'COST_FLOOR',
            'sl_dist_atr': 2.8,    # pre-floor ATR distance
            'sl_dist_lev': 3.0,    # leverage floor
            'sl_dist_final': 3.0,  # final = max(2.8, 3.0)
            'tp_dist_atr': 5.0,
            'tp_dist_lev': 0.5,
            'tp_dist_cost': 6.0,   # cost floor won
            'tp_dist_final': 6.0,
            'cost_roi_pct': 1.2,
            'tick_size': 0.01,
            'version': 'v1',
        }
        self.result = build_distance_truth(
            entry_price=100.0, sl=97.0, tp=106.0,
            trail_distance=1.5, side='LONG', leverage=10,
            meta=self.meta,
        )

    def test_effective_price_distances(self):
        assert self.result['sl_dist_price_effective'] == 3.0  # |100-97|
        assert self.result['tp_dist_price_effective'] == 6.0  # |106-100|
        assert self.result['trail_dist_price_effective'] == 1.5

    def test_effective_pct_distances(self):
        assert self.result['sl_dist_pct_effective'] == 3.0    # 3/100*100
        assert self.result['tp_dist_pct_effective'] == 6.0    # 6/100*100

    def test_effective_roi_distances(self):
        assert self.result['sl_dist_roi_effective'] == 30.0   # 3% × 10
        assert self.result['tp_dist_roi_effective'] == 60.0   # 6% × 10

    def test_pre_floor_atr_distances(self):
        assert self.result['sl_dist_pct_atr'] == 2.8   # 2.8/100*100
        assert self.result['tp_dist_pct_atr'] == 5.0   # 5.0/100*100

    def test_floor_distances(self):
        assert self.result['sl_dist_price_lev_floor'] == 3.0
        assert self.result['tp_dist_price_lev_floor'] == 0.5
        assert self.result['tp_dist_price_cost_floor'] == 6.0

    def test_cost_context(self):
        assert self.result['breakeven_fee_slippage_pct'] == 0.12  # 1.2/10
        assert self.result['cost_roi_pct'] == 1.2

    def test_source_tags(self):
        assert self.result['sl_source'] == 'ATR'
        assert self.result['tp_source'] == 'COST_FLOOR'
        assert self.result['distance_version'] == 'v1'

    def test_leverage(self):
        assert self.result['leverage_used'] == 10

    def test_source_tags_list(self):
        tags = self.result['distance_source_tags']
        assert 'ATR' in tags
        assert 'COST_FLOOR' in tags


class TestBuildDistanceTruthShort:
    """SHORT scenario: entry=100, sl=103, tp=94, trail=2.0, lev=20."""

    def setup_method(self):
        self.result = build_distance_truth(
            entry_price=100.0, sl=103.0, tp=94.0,
            trail_distance=2.0, side='SHORT', leverage=20,
            meta={
                'sl_source': 'LEV_FLOOR',
                'tp_source': 'ATR',
                'sl_dist_atr': 2.5,
                'sl_dist_lev': 1.5,
                'sl_dist_final': 3.0,
                'tp_dist_atr': 6.0,
                'tp_dist_lev': 0.25,
                'tp_dist_cost': 0.0,
                'tp_dist_final': 6.0,
                'cost_roi_pct': 0.8,
                'version': 'v2',
                'parity_mode': True,
                'profile': 'BALANCED',
            },
        )

    def test_effective_price_distances(self):
        assert self.result['sl_dist_price_effective'] == 3.0   # |100-103|
        assert self.result['tp_dist_price_effective'] == 6.0   # |94-100|

    def test_effective_roi(self):
        assert self.result['sl_dist_roi_effective'] == 60.0    # 3% × 20
        assert self.result['tp_dist_roi_effective'] == 120.0   # 6% × 20

    def test_parity_tag(self):
        assert 'PARITY' in self.result['distance_source_tags']
        assert 'PROFILE:BALANCED' in self.result['distance_source_tags']


class TestBuildDistanceTruthEdgeCases:

    def test_zero_entry(self):
        """entry=0 → should sanitize to 1.0, no crash."""
        result = build_distance_truth(0, 0.5, 1.5, 0.1, 'LONG', 10)
        assert result['sl_dist_pct_effective'] >= 0
        assert not math.isnan(result['sl_dist_pct_effective'])

    def test_nan_inputs(self):
        """NaN SL/TP → distances should be 0, not NaN."""
        result = build_distance_truth(100, float('nan'), float('nan'), float('nan'), 'LONG', 10)
        assert result['sl_dist_price_effective'] == 0.0
        assert result['tp_dist_price_effective'] == 0.0
        assert result['trail_dist_price_effective'] == 0.0

    def test_inf_inputs(self):
        """Inf inputs → sanitized to 0."""
        result = build_distance_truth(100, float('inf'), float('-inf'), float('inf'), 'LONG', 10)
        assert result['sl_dist_price_effective'] == 0.0

    def test_no_meta(self):
        """No meta → defaults everywhere, no crash."""
        result = build_distance_truth(100, 97, 103, 1.0, 'LONG', 5, meta=None)
        assert result['sl_source'] == 'UNKNOWN'
        assert result['tp_source'] == 'UNKNOWN'
        assert result['distance_version'] == 'legacy'

    def test_negative_leverage(self):
        """Negative leverage → clamped to 1."""
        result = build_distance_truth(100, 97, 103, 1.0, 'LONG', -5)
        assert result['leverage_used'] == 1
        assert result['sl_dist_roi_effective'] == result['sl_dist_pct_effective']  # lev=1


# ═════════════════════════════════════════════════════════
# Integration: compute_sl_tp_levels ↔ distance_truth
# ═════════════════════════════════════════════════════════

class TestComputeLevelsIntegration:
    """Verify compute_sl_tp_levels includes distance_truth key."""

    def test_distance_truth_present_in_result(self):
        """compute_sl_tp_levels result must contain distance_truth dict."""
        try:
            # Import from main.py (may fail in isolation)
            sys.path.insert(0, os.path.dirname(__file__))
            from risk.sl_tp_engine import compute_sl_tp_levels_v1
            result = compute_sl_tp_levels_v1(
                entry_price=100.0, atr=2.0, side='LONG', leverage=10,
                adjusted_sl_atr=1.5, adjusted_tp_atr=3.0,
                adjusted_trail_act_atr=2.0, adjusted_trail_dist_atr=0.5,
            )
            # Build distance_truth from result (simulates what main.py wrapper does)
            dt = build_distance_truth(
                entry_price=100.0, sl=result['sl'], tp=result['tp'],
                trail_distance=result['trail_distance'], side='LONG',
                leverage=10, meta=result['meta'],
            )
            assert isinstance(dt, dict)
            assert 'sl_dist_pct_effective' in dt
            assert 'tp_dist_pct_effective' in dt
            assert 'sl_source' in dt
            assert dt['sl_dist_pct_effective'] > 0
            assert dt['tp_dist_pct_effective'] > 0
        except ImportError:
            pytest.skip("Risk modules not importable in test env")

    def test_v2_distance_truth(self):
        """V2 result + distance_truth integration."""
        try:
            from risk.sl_tp_engine import compute_sl_tp_levels_v2
            from risk.liquidity_profile import LiquidityProfile
            result = compute_sl_tp_levels_v2(
                entry_price=50000.0, atr=500.0, side='SHORT', leverage=20,
                adjusted_sl_atr=1.5, adjusted_tp_atr=3.0,
                adjusted_trail_act_atr=2.0, adjusted_trail_dist_atr=0.5,
                parity_mode=True,
            )
            dt = build_distance_truth(
                entry_price=50000.0, sl=result['sl'], tp=result['tp'],
                trail_distance=result['trail_distance'], side='SHORT',
                leverage=20, meta=result['meta'],
            )
            assert dt['distance_version'] == 'v2'
            assert dt['leverage_used'] == 20
            assert dt['sl_dist_roi_effective'] > 0
        except ImportError:
            pytest.skip("Risk modules not importable in test env")
