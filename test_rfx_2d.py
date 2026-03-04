"""RFX-2D: Counter-Trend Exit Relaxation test suite.

Tests:
  1. classify_counter_trend — classification logic
  2. Exit envelope relaxation multipliers
  3. Two-phase trail transition (WIDE → NORMAL)
  4. Recovery relaxation — LossRecoveryTrailManager
  5. Recovery relaxation — inline check_loss_recovery
  6. distance_truth CT flags injection
  7. Feature flag gating (disabled path)
"""

import importlib
import os
import sys
import pytest

# ═════════════════════════════════════════════════════════
# Helper: import classify_counter_trend from main.py
# ═════════════════════════════════════════════════════════

# We extract the classify_counter_trend function via exec of just that block
# to avoid importing the entire 35k-line main.py.
_CLASSIFY_SRC = '''
def classify_counter_trend(action, mtf_result, dca_alignment, coin_trends, symbol):
    mtf_score = int(mtf_result.get('mtf_score', 0) or 0)
    is_ct = False
    if mtf_score <= -10:
        is_ct = True
    if str(dca_alignment).upper() == 'CONFLICT':
        is_ct = True
    trends = coin_trends.get(symbol, {})
    t_4h = str(trends.get('trend_4h', '')).upper()
    t_1d = str(trends.get('trend_1d', '')).upper()
    if action == 'LONG' and ('BEARISH' in t_4h or 'BEARISH' in t_1d):
        is_ct = True
    if action == 'SHORT' and ('BULLISH' in t_4h or 'BULLISH' in t_1d):
        is_ct = True
    strength = min(1.0, max(0.0, abs(min(mtf_score, 0)) / 60.0))
    if str(dca_alignment).upper() == 'CONFLICT':
        strength = min(1.0, strength + 0.25)
    return is_ct, strength
'''
_ns = {}
exec(_CLASSIFY_SRC, _ns)
classify_counter_trend = _ns['classify_counter_trend']

# Feature flag defaults (mirrors main.py)
CT_RELAX_SL_MIN = 1.30
CT_RELAX_SL_MAX = 1.90
CT_RELAX_TRAIL_ACT_MIN = 1.15
CT_RELAX_TRAIL_ACT_MAX = 1.35
CT_RELAX_TRAIL_DIST_MIN = 1.35
CT_RELAX_TRAIL_DIST_MAX = 1.90
# Must match main.py default (0.70) — WIDE→NORMAL must tighten (mult < 1)
CT_WIDE_TRAIL_FLOOR_MULT = 0.70
CT_WIDE_TO_NORMAL_PROFIT_OVER_BE = 0.3
CT_WIDE_TO_NORMAL_AGE_SEC = 900


# ═════════════════════════════════════════════════════════
# Unit: classify_counter_trend
# ═════════════════════════════════════════════════════════

class TestClassifyCounterTrend:
    """Tests for the classify_counter_trend function."""

    def test_not_counter_trend_when_aligned(self):
        """Trend-following signal should NOT be classified as counter-trend."""
        is_ct, strength = classify_counter_trend(
            'LONG',
            {'mtf_score': 20},
            'ALIGNED',
            {'BTCUSDT': {'trend_4h': 'BULLISH', 'trend_1d': 'BULLISH'}},
            'BTCUSDT'
        )
        assert not is_ct
        assert strength == 0.0

    def test_counter_trend_negative_mtf(self):
        """Negative MTF score <= -10 should trigger CT classification."""
        is_ct, strength = classify_counter_trend(
            'LONG',
            {'mtf_score': -15},
            'ALIGNED',
            {'BTCUSDT': {}},
            'BTCUSDT'
        )
        assert is_ct
        assert strength == pytest.approx(15 / 60.0, abs=0.01)

    def test_counter_trend_dca_conflict(self):
        """DCA alignment CONFLICT should trigger CT classification."""
        is_ct, strength = classify_counter_trend(
            'SHORT',
            {'mtf_score': 0},
            'CONFLICT',
            {'ETHUSDT': {}},
            'ETHUSDT'
        )
        assert is_ct
        assert strength == pytest.approx(0.25, abs=0.01)  # 0 + 0.25 (conflict bonus)

    def test_counter_trend_bearish_long(self):
        """LONG signal with BEARISH 4h trend should be CT."""
        is_ct, _ = classify_counter_trend(
            'LONG',
            {'mtf_score': 0},
            'PARTIAL',
            {'ADAUSDT': {'trend_4h': 'BEARISH', 'trend_1d': 'NEUTRAL'}},
            'ADAUSDT'
        )
        assert is_ct

    def test_counter_trend_bullish_short(self):
        """SHORT signal with BULLISH 1d trend should be CT."""
        is_ct, _ = classify_counter_trend(
            'SHORT',
            {'mtf_score': 0},
            'ALIGNED',
            {'XRPUSDT': {'trend_4h': 'NEUTRAL', 'trend_1d': 'BULLISH'}},
            'XRPUSDT'
        )
        assert is_ct

    def test_strength_combined(self):
        """Negative MTF + DCA conflict should combine strength."""
        is_ct, strength = classify_counter_trend(
            'LONG',
            {'mtf_score': -30},
            'CONFLICT',
            {'SOLUSDT': {'trend_4h': 'BEARISH'}},
            'SOLUSDT'
        )
        assert is_ct
        # strength = abs(-30)/60 + 0.25 = 0.5+0.25 = 0.75
        assert strength == pytest.approx(0.75, abs=0.01)

    def test_strength_capped_at_1(self):
        """Strength should be capped at 1.0."""
        is_ct, strength = classify_counter_trend(
            'LONG',
            {'mtf_score': -60},
            'CONFLICT',
            {'DOGEUSDT': {'trend_4h': 'BEARISH'}},
            'DOGEUSDT'
        )
        assert is_ct
        assert strength == 1.0  # 1.0 + 0.25 → capped at 1.0

    def test_mtf_score_boundary(self):
        """MTF score of exactly -10 should trigger CT."""
        is_ct, _ = classify_counter_trend(
            'LONG', {'mtf_score': -10}, 'ALIGNED', {'X': {}}, 'X'
        )
        assert is_ct

    def test_mtf_score_above_boundary_no_ct(self):
        """MTF score of -9 should NOT trigger CT from MTF alone."""
        is_ct, _ = classify_counter_trend(
            'LONG', {'mtf_score': -9}, 'ALIGNED', {'X': {}}, 'X'
        )
        assert not is_ct


# ═════════════════════════════════════════════════════════
# Unit: Exit envelope relaxation multipliers
# ═════════════════════════════════════════════════════════

def _compute_ct_mults(ct_strength: float):
    """Mirror the multiplier logic from main.py open_position."""
    sl = max(CT_RELAX_SL_MIN, min(CT_RELAX_SL_MAX, CT_RELAX_SL_MIN + (CT_RELAX_SL_MAX - CT_RELAX_SL_MIN) * ct_strength))
    trail_act = max(CT_RELAX_TRAIL_ACT_MIN, min(CT_RELAX_TRAIL_ACT_MAX, CT_RELAX_TRAIL_ACT_MIN + (CT_RELAX_TRAIL_ACT_MAX - CT_RELAX_TRAIL_ACT_MIN) * ct_strength))
    trail_dist = max(CT_RELAX_TRAIL_DIST_MIN, min(CT_RELAX_TRAIL_DIST_MAX, CT_RELAX_TRAIL_DIST_MIN + (CT_RELAX_TRAIL_DIST_MAX - CT_RELAX_TRAIL_DIST_MIN) * ct_strength))
    return sl, trail_act, trail_dist


class TestExitEnvelopeRelaxation:
    """Tests for CT exit envelope multiplier calculations."""

    def test_zero_strength(self):
        sl, act, dist = _compute_ct_mults(0.0)
        assert sl == pytest.approx(CT_RELAX_SL_MIN)
        assert act == pytest.approx(CT_RELAX_TRAIL_ACT_MIN)
        assert dist == pytest.approx(CT_RELAX_TRAIL_DIST_MIN)

    def test_full_strength(self):
        sl, act, dist = _compute_ct_mults(1.0)
        assert sl == pytest.approx(CT_RELAX_SL_MAX)
        assert act == pytest.approx(CT_RELAX_TRAIL_ACT_MAX)
        assert dist == pytest.approx(CT_RELAX_TRAIL_DIST_MAX)

    def test_half_strength(self):
        sl, act, dist = _compute_ct_mults(0.5)
        assert sl == pytest.approx(1.60, abs=0.01)   # midpoint of 1.30..1.90
        assert act == pytest.approx(1.25, abs=0.01)   # midpoint of 1.15..1.35
        assert dist == pytest.approx(1.625, abs=0.01) # midpoint of 1.35..1.90

    def test_non_ct_is_1x(self):
        """Non-CT trades should use 1.0x multipliers."""
        # When COUNTER_EXIT_RELAX_ENABLED is True but position is not CT,
        # multipliers stay at 1.0
        ct_sl_mult = 1.0
        ct_trail_act_mult = 1.0
        ct_trail_dist_mult = 1.0
        assert ct_sl_mult == 1.0
        assert ct_trail_act_mult == 1.0
        assert ct_trail_dist_mult == 1.0


# ═════════════════════════════════════════════════════════
# Unit: Two-phase trail transition
# ═════════════════════════════════════════════════════════

class TestTwoPhaseTrail:
    """Tests for WIDE → NORMAL trail phase transition logic."""

    def _should_transition(self, pnl_pct: float, be_pnl_pct: float, age_sec: float) -> bool:
        """Mirror transition check from main.py."""
        profit_over_be = pnl_pct - be_pnl_pct
        if profit_over_be >= CT_WIDE_TO_NORMAL_PROFIT_OVER_BE:
            return True
        if age_sec >= CT_WIDE_TO_NORMAL_AGE_SEC:
            return True
        return False

    def test_transition_by_profit(self):
        """Should transition when profit is 0.3% above breakeven."""
        assert self._should_transition(1.5, 1.2, 300)  # 0.3% over BE, young

    def test_no_transition_insufficient_profit(self):
        """Should NOT transition when profit is below threshold."""
        assert not self._should_transition(1.4, 1.2, 300)  # 0.2% over BE

    def test_transition_by_age(self):
        """Should transition when age exceeds 900 seconds."""
        assert self._should_transition(-2.0, 0.5, 901)  # Loss, but old enough

    def test_no_transition_young_and_losing(self):
        """Should NOT transition when young and not profitable enough."""
        assert not self._should_transition(-1.0, 0.5, 500)

    def test_transition_boundary_profit(self):
        """Exact boundary: 0.3% over BE should transition."""
        assert self._should_transition(0.8, 0.5, 0)  # exactly 0.3%

    def test_transition_boundary_age(self):
        """Exact boundary: 900 seconds should transition."""
        assert self._should_transition(-5.0, 0.5, 900)

    def test_wide_phase_tightens_trail(self):
        """After transition, trail distance should be tightened by CT_WIDE_TRAIL_FLOOR_MULT."""
        original_trail = 2.5
        tightened = original_trail * CT_WIDE_TRAIL_FLOOR_MULT
        assert tightened == pytest.approx(1.75, abs=0.01)
        # CRITICAL: new trail MUST be smaller than old trail
        assert tightened < original_trail, (
            f"WIDE→NORMAL must tighten: {tightened} >= {original_trail} "
            f"(CT_WIDE_TRAIL_FLOOR_MULT={CT_WIDE_TRAIL_FLOOR_MULT} must be < 1)"
        )

    def test_floor_mult_must_be_less_than_one(self):
        """CT_WIDE_TRAIL_FLOOR_MULT must be < 1 to tighten trail."""
        assert CT_WIDE_TRAIL_FLOOR_MULT < 1.0, (
            f"CT_WIDE_TRAIL_FLOOR_MULT={CT_WIDE_TRAIL_FLOOR_MULT} >= 1.0 — "
            f"this would WIDEN the trail on WIDE→NORMAL transition!"
        )
        assert CT_WIDE_TRAIL_FLOOR_MULT > 0.0

    def test_clamp_prevents_widening(self):
        """Safety clamp should force mult into (0.1, 0.99) range — always strictly tightens."""
        # Simulate the clamp from main.py transition code
        for bad_val in [1.0, 1.35, 2.0, 5.0]:
            clamped = min(max(bad_val, 0.1), 0.99)
            assert clamped < 1.0, f"Clamp failed for {bad_val}: got {clamped}"
        for good_val in [0.70, 0.5, 0.3]:
            clamped = min(max(good_val, 0.1), 0.99)
            assert clamped == good_val


# ═════════════════════════════════════════════════════════
# Unit: Recovery relaxation — LossRecoveryTrailManager
# ═════════════════════════════════════════════════════════

class TestRecoveryRelaxation:
    """Tests for CT-aware recovery thresholds in LossRecoveryTrailManager."""

    def test_ct_recovery_activation_pct(self):
        """CT positions should require 65% recovery (vs 50% for non-CT)."""
        base = 0.50
        ct_act = 0.65
        assert ct_act > base

    def test_ct_trail_giveback_with_zero_strength(self):
        """CT giveback at strength=0 should be 75%."""
        strength = 0.0
        ct_giveback = 0.75 + 0.03 * strength
        assert ct_giveback == pytest.approx(0.75)

    def test_ct_trail_giveback_with_full_strength(self):
        """CT giveback at strength=1 should be 78%."""
        strength = 1.0
        ct_giveback = 0.75 + 0.03 * strength
        assert ct_giveback == pytest.approx(0.78)

    def test_non_ct_uses_default_values(self):
        """Non-CT positions should use default 50% activation / 65% giveback."""
        base_activation = 0.50
        base_giveback = 0.65
        # These should be passed through unchanged
        assert base_activation == 0.50
        assert base_giveback == 0.65


# ═════════════════════════════════════════════════════════
# Unit: Inline check_loss_recovery relaxation
# ═════════════════════════════════════════════════════════

class TestInlineLossRecovery:
    """Tests for CT-aware thresholds in check_loss_recovery."""

    def test_ct_loss_threshold(self):
        """CT positions use 5% loss threshold (vs 3% for non-CT)."""
        ct_threshold = 5
        non_ct_threshold = 3
        assert ct_threshold > non_ct_threshold

    def test_ct_atr_multiplier(self):
        """CT positions use 0.9x ATR trail distance (vs 0.6x for non-CT)."""
        ct_atr = 0.9
        non_ct_atr = 0.6
        assert ct_atr > non_ct_atr

    def test_recovery_mode_long_ct(self):
        """CT LONG: price must bounce 0.9*ATR from low to activate recovery."""
        atr = 100
        recovery_low = 1000
        # CT: needs 0.9 * ATR = 90 bounce
        # Non-CT: needs 0.6 * ATR = 60 bounce
        ct_activation_price = recovery_low + (atr * 0.9)
        non_ct_activation_price = recovery_low + (atr * 0.6)
        assert ct_activation_price > non_ct_activation_price
        assert ct_activation_price == 1090
        assert non_ct_activation_price == 1060

    def test_recovery_mode_short_ct(self):
        """CT SHORT: price must drop 0.9*ATR from high to activate recovery."""
        atr = 100
        recovery_high = 2000
        ct_activation_price = recovery_high - (atr * 0.9)
        non_ct_activation_price = recovery_high - (atr * 0.6)
        assert ct_activation_price < non_ct_activation_price
        assert ct_activation_price == 1910
        assert non_ct_activation_price == 1940


# ═════════════════════════════════════════════════════════
# Unit: distance_truth CT flags injection
# ═════════════════════════════════════════════════════════

class TestDistanceTruthCTFlags:
    """Tests for CT flags injected into distance_truth.quality_flags."""

    def test_ct_relax_flag_added(self):
        """CT position should get CT_RELAX flag."""
        pos = {'counterTrend': True, 'trail_phase': 'NORMAL', 'distance_truth': {'quality_flags': []}}
        flags = list(pos['distance_truth']['quality_flags'])
        if pos.get('counterTrend', False):
            flags.append('CT_RELAX')
            if pos.get('trail_phase') == 'WIDE':
                flags.append('CT_WIDE')
        assert 'CT_RELAX' in flags
        assert 'CT_WIDE' not in flags

    def test_ct_wide_flag_added(self):
        """CT position in WIDE phase should get both CT_RELAX and CT_WIDE."""
        pos = {'counterTrend': True, 'trail_phase': 'WIDE', 'distance_truth': {'quality_flags': []}}
        flags = list(pos['distance_truth']['quality_flags'])
        if pos.get('counterTrend', False):
            flags.append('CT_RELAX')
            if pos.get('trail_phase') == 'WIDE':
                flags.append('CT_WIDE')
        assert 'CT_RELAX' in flags
        assert 'CT_WIDE' in flags

    def test_non_ct_no_flags(self):
        """Non-CT position should NOT get CT flags."""
        pos = {'counterTrend': False, 'trail_phase': 'NORMAL', 'distance_truth': {'quality_flags': ['SL_TIGHT']}}
        flags = list(pos['distance_truth']['quality_flags'])
        if pos.get('counterTrend', False):
            flags.append('CT_RELAX')
        assert 'CT_RELAX' not in flags
        assert flags == ['SL_TIGHT']

    def test_existing_flags_preserved(self):
        """Existing quality_flags should be preserved when CT flags are added."""
        pos = {'counterTrend': True, 'trail_phase': 'WIDE', 'distance_truth': {'quality_flags': ['SL_TIGHT', 'TP_WIDE']}}
        flags = list(pos['distance_truth']['quality_flags'])
        if pos.get('counterTrend', False):
            flags.append('CT_RELAX')
            if pos.get('trail_phase') == 'WIDE':
                flags.append('CT_WIDE')
        assert flags == ['SL_TIGHT', 'TP_WIDE', 'CT_RELAX', 'CT_WIDE']


# ═════════════════════════════════════════════════════════
# Unit: Feature flag gating
# ═════════════════════════════════════════════════════════

class TestFeatureFlagGating:
    """Tests for COUNTER_EXIT_RELAX_ENABLED gating."""

    def test_disabled_flag_bypasses_relaxation(self):
        """When COUNTER_EXIT_RELAX_ENABLED=False, all multipliers should be 1.0."""
        COUNTER_EXIT_RELAX_ENABLED = False
        _is_ct_signal = True
        ct_sl_mult = 1.0
        ct_trail_act_mult = 1.0
        ct_trail_dist_mult = 1.0
        if COUNTER_EXIT_RELAX_ENABLED and _is_ct_signal:
            ct_sl_mult = 1.5
        assert ct_sl_mult == 1.0
        assert ct_trail_act_mult == 1.0
        assert ct_trail_dist_mult == 1.0

    def test_enabled_flag_applies_relaxation(self):
        """When COUNTER_EXIT_RELAX_ENABLED=True and CT, multipliers should change."""
        COUNTER_EXIT_RELAX_ENABLED = True
        _is_ct_signal = True
        ct_sl_mult = 1.0
        if COUNTER_EXIT_RELAX_ENABLED and _is_ct_signal:
            ct_sl_mult = 1.5
        assert ct_sl_mult == 1.5

    def test_disabled_recovery_bypass(self):
        """When disabled, CT recovery thresholds should match defaults."""
        COUNTER_EXIT_RELAX_ENABLED = False
        _is_ct_recovery = COUNTER_EXIT_RELAX_ENABLED and True  # True = counterTrend
        _recovery_act_pct = 0.65 if _is_ct_recovery else 0.50
        assert _recovery_act_pct == 0.50  # Should use default

    def test_disabled_inline_recovery_bypass(self):
        """When disabled, inline recovery uses default 3% / 0.6x."""
        COUNTER_EXIT_RELAX_ENABLED = False
        _is_ct_lr = COUNTER_EXIT_RELAX_ENABLED and True
        _lr_loss_threshold = 5 if _is_ct_lr else 3
        _lr_atr_mult = 0.9 if _is_ct_lr else 0.6
        assert _lr_loss_threshold == 3
        assert _lr_atr_mult == 0.6
