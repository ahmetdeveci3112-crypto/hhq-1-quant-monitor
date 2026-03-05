"""SMART_V3_RUNNER: Unit, parity, and integration tests.

Run:  python3 -m pytest tests/test_smart_v3_runner.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from risk.strategy_profile import (
    STRATEGY_MODE_LEGACY,
    STRATEGY_MODE_SMART_V2,
    STRATEGY_MODE_SMART_V3_RUNNER,
    VALID_STRATEGY_MODES,
    normalize_strategy_mode,
    is_adaptive_mode,
    is_smart_mode,
    StrategyExecutionProfile,
    resolve_strategy_execution_profile,
)
from risk.policy import RiskParams, RiskProfile, resolve_risk_params
from risk.sl_tp_engine import compute_sl_tp_levels_v1, compute_sl_tp_levels_v2
from risk.breakeven import compute_breakeven_price


# ═══════════════════════════════════════════════════════════════════
# 1. CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════

class TestConstants:
    def test_modes_exist(self):
        assert STRATEGY_MODE_LEGACY == "LEGACY"
        assert STRATEGY_MODE_SMART_V2 == "SMART_V2"
        assert STRATEGY_MODE_SMART_V3_RUNNER == "SMART_V3_RUNNER"

    def test_valid_modes_contains_all(self):
        assert STRATEGY_MODE_LEGACY in VALID_STRATEGY_MODES
        assert STRATEGY_MODE_SMART_V2 in VALID_STRATEGY_MODES
        assert STRATEGY_MODE_SMART_V3_RUNNER in VALID_STRATEGY_MODES


class TestNormalizeStrategyMode:
    @pytest.mark.parametrize("raw,expected", [
        ("LEGACY", STRATEGY_MODE_LEGACY),
        ("legacy", STRATEGY_MODE_LEGACY),
        ("SMART_V2", STRATEGY_MODE_SMART_V2),
        ("smart_v2", STRATEGY_MODE_SMART_V2),
        ("SMART_V3_RUNNER", STRATEGY_MODE_SMART_V3_RUNNER),
        ("smart_v3_runner", STRATEGY_MODE_SMART_V3_RUNNER),
        ("", STRATEGY_MODE_LEGACY),
        (None, STRATEGY_MODE_LEGACY),
        ("INVALID", STRATEGY_MODE_LEGACY),
        ("SMART_V4", STRATEGY_MODE_LEGACY),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_strategy_mode(raw) == expected


class TestIsAdaptiveMode:
    @pytest.mark.parametrize("mode,expected", [
        (STRATEGY_MODE_LEGACY, True),
        (STRATEGY_MODE_SMART_V2, True),
        (STRATEGY_MODE_SMART_V3_RUNNER, True),
        ("UNKNOWN", False),
    ])
    def test_adaptive(self, mode, expected):
        assert is_adaptive_mode(mode) == expected


class TestIsSmartMode:
    @pytest.mark.parametrize("mode,expected", [
        (STRATEGY_MODE_LEGACY, False),
        (STRATEGY_MODE_SMART_V2, True),
        (STRATEGY_MODE_SMART_V3_RUNNER, True),
        ("UNKNOWN", False),
    ])
    def test_smart(self, mode, expected):
        assert is_smart_mode(mode) == expected


# ═══════════════════════════════════════════════════════════════════
# 2. STRATEGY EXECUTION PROFILE
# ═══════════════════════════════════════════════════════════════════

class TestStrategyExecutionProfile:
    def test_neutral_is_identity(self):
        """Neutral profile must not alter any multipliers."""
        p = StrategyExecutionProfile.neutral("LEGACY")
        assert p.trail_activation_mult == 1.0
        assert p.trail_distance_mult == 1.0
        assert p.tp_tighten_intensity == 1.0
        assert p.be_buffer_mult == 1.0
        assert p.source == "neutral"

    def test_neutral_mode_preserved(self):
        p = StrategyExecutionProfile.neutral(STRATEGY_MODE_SMART_V2)
        assert p.mode == STRATEGY_MODE_SMART_V2

    def test_runner_has_wider_trail(self):
        """SMART_V3_RUNNER must have wider trail activation & distance."""
        p = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert p.trail_activation_mult > 1.0
        assert p.trail_distance_mult > 1.0
        assert p.mode == STRATEGY_MODE_SMART_V3_RUNNER
        assert p.source == "resolver"

    def test_runner_tp_tighten_less_than_1(self):
        """SMART_V3_RUNNER should have less aggressive TP tightening."""
        p = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert p.tp_tighten_intensity < 1.0

    def test_runner_be_buffer_wider(self):
        """SMART_V3_RUNNER should have wider breakeven buffer."""
        p = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert p.be_buffer_mult > 1.0


# ═══════════════════════════════════════════════════════════════════
# 3. PARITY: LEGACY / SMART_V2 → NEUTRAL PROFILE
# ═══════════════════════════════════════════════════════════════════

class TestResolverParity:
    """resolve_strategy_execution_profile must return neutral for non-RUNNER modes."""

    @pytest.mark.parametrize("mode", [STRATEGY_MODE_LEGACY, STRATEGY_MODE_SMART_V2])
    def test_legacy_and_v2_get_neutral(self, mode):
        p = resolve_strategy_execution_profile(mode=mode)
        neutral = StrategyExecutionProfile.neutral(mode)
        assert p.trail_activation_mult == neutral.trail_activation_mult
        assert p.trail_distance_mult == neutral.trail_distance_mult
        assert p.tp_tighten_intensity == neutral.tp_tighten_intensity
        assert p.be_buffer_mult == neutral.be_buffer_mult

    def test_unknown_mode_falls_to_legacy(self):
        p = resolve_strategy_execution_profile(mode="UNKNOWN_MODE")
        assert p.mode == STRATEGY_MODE_LEGACY
        assert p.trail_activation_mult == 1.0


# ═══════════════════════════════════════════════════════════════════
# 4. SL/TP ENGINE — V2 with strategy_profile
# ═══════════════════════════════════════════════════════════════════

class TestSlTpEngineV2:
    """Test that compute_sl_tp_levels_v2 correctly applies strategy_profile."""

    COMMON_ARGS = dict(
        entry_price=100.0,
        atr=2.0,
        side='LONG',
        leverage=10,
        adjusted_sl_atr=2.0,
        adjusted_tp_atr=3.0,
        adjusted_trail_act_atr=2.5,
        adjusted_trail_dist_atr=1.0,
        spread_pct=0.05,
        tick_size=0.01,
    )

    def test_v1_v2_parity_without_profile(self):
        """V2 in parity_mode must produce identical output to V1."""
        v1 = compute_sl_tp_levels_v1(**self.COMMON_ARGS)
        v2 = compute_sl_tp_levels_v2(**self.COMMON_ARGS, parity_mode=True)
        assert abs(v1['sl'] - v2['sl']) < 1e-8
        assert abs(v1['tp'] - v2['tp']) < 1e-8
        assert abs(v1['trail_activation'] - v2['trail_activation']) < 1e-8
        assert abs(v1['trail_distance'] - v2['trail_distance']) < 1e-8

    def test_neutral_profile_is_noop(self):
        """Neutral profile should not change output vs no profile."""
        neutral = StrategyExecutionProfile.neutral(STRATEGY_MODE_SMART_V2)
        without = compute_sl_tp_levels_v2(**self.COMMON_ARGS)
        with_neutral = compute_sl_tp_levels_v2(**self.COMMON_ARGS, strategy_profile=neutral)
        assert abs(without['trail_activation'] - with_neutral['trail_activation']) < 1e-8
        assert abs(without['trail_distance'] - with_neutral['trail_distance']) < 1e-8

    def test_runner_profile_widens_trail(self):
        """SMART_V3_RUNNER profile should widen trail activation & distance."""
        runner = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        without = compute_sl_tp_levels_v2(**self.COMMON_ARGS)
        with_runner = compute_sl_tp_levels_v2(**self.COMMON_ARGS, strategy_profile=runner)
        # Trail activation for LONG should be HIGHER (farther from entry)
        assert with_runner['trail_activation'] > without['trail_activation']
        # Trail distance should be wider
        assert with_runner['trail_distance'] > without['trail_distance']

    def test_parity_mode_ignores_profile(self):
        """parity_mode=True should ignore strategy_profile entirely."""
        runner = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        v1 = compute_sl_tp_levels_v1(**self.COMMON_ARGS)
        v2_parity = compute_sl_tp_levels_v2(
            **self.COMMON_ARGS, parity_mode=True, strategy_profile=runner
        )
        assert abs(v1['trail_activation'] - v2_parity['trail_activation']) < 1e-8
        assert abs(v1['trail_distance'] - v2_parity['trail_distance']) < 1e-8

    def test_short_side_runner(self):
        """SHORT side should also widen trail (activation lower for SHORT)."""
        args = {**self.COMMON_ARGS, 'side': 'SHORT'}
        runner = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        without = compute_sl_tp_levels_v2(**args)
        with_runner = compute_sl_tp_levels_v2(**args, strategy_profile=runner)
        # For SHORT, trail activation should be LOWER (farther from entry downward)
        assert with_runner['trail_activation'] < without['trail_activation']
        assert with_runner['trail_distance'] > without['trail_distance']


# ═══════════════════════════════════════════════════════════════════
# 5. BREAKEVEN — be_buffer_mult
# ═══════════════════════════════════════════════════════════════════

class TestBreakevenBufferMult:
    def test_default_mult_is_neutral(self):
        """be_buffer_mult=1.0 should not change output."""
        be1 = compute_breakeven_price(100.0, 'LONG', be_buffer_mult=1.0)
        be_default = compute_breakeven_price(100.0, 'LONG')
        assert abs(be1 - be_default) < 1e-10

    def test_wider_buffer_for_runner(self):
        """be_buffer_mult>1 should widen the breakeven buffer (LONG: higher BE)."""
        be_neutral = compute_breakeven_price(100.0, 'LONG', be_buffer_mult=1.0)
        be_runner = compute_breakeven_price(100.0, 'LONG', be_buffer_mult=1.2)
        assert be_runner > be_neutral

    def test_wider_buffer_short(self):
        """SHORT: wider buffer means lower breakeven price."""
        be_neutral = compute_breakeven_price(100.0, 'SHORT', be_buffer_mult=1.0)
        be_runner = compute_breakeven_price(100.0, 'SHORT', be_buffer_mult=1.2)
        assert be_runner < be_neutral


# ═══════════════════════════════════════════════════════════════════
# 6. RISK PARAMS — strategy_mode field
# ═══════════════════════════════════════════════════════════════════

class TestRiskParamsStrategyMode:
    def test_default_is_legacy(self):
        rp = resolve_risk_params()
        assert rp.strategy_mode == 'LEGACY'

    def test_runner_passes_through(self):
        rp = resolve_risk_params(strategy_mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert rp.strategy_mode == STRATEGY_MODE_SMART_V3_RUNNER

    def test_profiles_unchanged(self):
        """strategy_mode should not affect numeric risk parameters."""
        rp_leg = resolve_risk_params(strategy_mode='LEGACY')
        rp_run = resolve_risk_params(strategy_mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert rp_leg.sl_roi_floor == rp_run.sl_roi_floor
        assert rp_leg.tp_roi_floor == rp_run.tp_roi_floor
        assert rp_leg.emergency_roi == rp_run.emergency_roi


# ═══════════════════════════════════════════════════════════════════
# 7. GOLDEN SNAPSHOT — quantitative parity check
# ═══════════════════════════════════════════════════════════════════

class TestGoldenSnapshot:
    """Field-by-field numerical verification of known outputs."""

    def test_runner_trail_multiplier_values(self):
        """Pin the exact multiplier values for SMART_V3_RUNNER."""
        p = resolve_strategy_execution_profile(mode=STRATEGY_MODE_SMART_V3_RUNNER)
        assert p.trail_activation_mult == 1.30
        assert p.trail_distance_mult == 1.45
        assert p.tp_tighten_intensity == 0.70
        assert p.be_buffer_mult == 1.20
        assert p.wide_to_normal_profit_pct == 0.35
        assert p.wide_to_normal_age_sec == 900

    def test_legacy_exact_neutral(self):
        """LEGACY must return exact 1.0 for all multipliers."""
        p = resolve_strategy_execution_profile(mode=STRATEGY_MODE_LEGACY)
        assert p.trail_activation_mult == 1.0
        assert p.trail_distance_mult == 1.0
        assert p.tp_tighten_intensity == 1.0
        assert p.be_buffer_mult == 1.0
        assert p.ct_relax_base == 1.0
        assert p.ct_relax_max == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
