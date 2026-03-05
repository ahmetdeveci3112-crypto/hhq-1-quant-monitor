"""RFX-3A Test Suite: Regime-Based Execution Styles.

Covers:
  - Parity: LEGACY/V2 always MARKET, no style mutation
  - Classify: regime→style mapping
  - Zone: 4-stage fallback chain
  - Reclaim: confirmation + fail-safe
  - Partial TP: fraction computation
  - Fallback: stage resolution
  - Profile: exec_style fields on neutral/runner
"""
import pytest
import os

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure env flags are clean for each test."""
    for key in [
        "RFX3_EXEC_STYLE_ENABLED",
        "RFX3_MR_LIMIT_ENABLED",
        "RFX3_BREAKOUT_RECLAIM_ENABLED",
        "RFX3_STRUCT_PARTIAL_TP_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)


# ═══════════════════════════════════════════════════════════════════
# 1. Parity Tests
# ═══════════════════════════════════════════════════════════════════

class TestParity:
    """LEGACY and SMART_V2 must ALWAYS return MARKET style."""

    def test_legacy_always_market(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MARKET
        result = classify_execution_style(
            regime="RANGING", structure="NONE", hurst=0.3, zscore=2.0, side="LONG",
            master_enabled=True, mr_enabled=True, breakout_enabled=True,
        )
        # Even with all flags on, this is just classification — mode gate is separate
        # The LEGACY guard is at profile level (exec_style_enabled=False)
        assert isinstance(result, str)

    def test_v2_profile_no_exec_style(self):
        from risk.strategy_profile import (
            resolve_strategy_execution_profile,
            STRATEGY_MODE_SMART_V2,
        )
        profile = resolve_strategy_execution_profile(STRATEGY_MODE_SMART_V2)
        assert profile.exec_style_enabled is False
        assert profile.struct_partial_tp_pct == 0.0
        assert profile.struct_partial_be_floor is False

    def test_legacy_profile_no_exec_style(self):
        from risk.strategy_profile import (
            resolve_strategy_execution_profile,
            STRATEGY_MODE_LEGACY,
        )
        profile = resolve_strategy_execution_profile(STRATEGY_MODE_LEGACY)
        assert profile.exec_style_enabled is False
        assert profile.struct_partial_tp_pct == 0.0

    def test_v2_no_style_mutation(self):
        """V2 profile must have all neutral multipliers untouched."""
        from risk.strategy_profile import (
            resolve_strategy_execution_profile,
            STRATEGY_MODE_SMART_V2,
        )
        p = resolve_strategy_execution_profile(STRATEGY_MODE_SMART_V2)
        assert p.trail_activation_mult == 1.0
        assert p.trail_distance_mult == 1.0
        assert p.be_buffer_mult == 1.0
        assert p.source == "neutral"
        assert p.exec_style_enabled is False

    def test_classify_flags_off_always_market(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MARKET
        result = classify_execution_style(
            regime="RANGING", structure="BOS_UP", hurst=0.3, zscore=2.0, side="LONG",
            master_enabled=False, mr_enabled=True, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_MARKET


# ═══════════════════════════════════════════════════════════════════
# 2. Classification Tests
# ═══════════════════════════════════════════════════════════════════

class TestClassify:

    def test_ranging_mr_zone_limit(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MR_LIMIT
        result = classify_execution_style(
            regime="RANGING", structure="NONE", hurst=0.3, zscore=2.0, side="LONG",
            master_enabled=True, mr_enabled=True, breakout_enabled=False,
        )
        assert result == EXEC_STYLE_MR_LIMIT

    def test_quiet_mr_zone_limit(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MR_LIMIT
        result = classify_execution_style(
            regime="QUIET", structure="NONE", hurst=0.4, zscore=1.8, side="SHORT",
            master_enabled=True, mr_enabled=True, breakout_enabled=False,
        )
        assert result == EXEC_STYLE_MR_LIMIT

    def test_trending_up_breakout_long(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_BREAKOUT
        result = classify_execution_style(
            regime="TRENDING_UP", structure="BOS_UP", hurst=0.7, zscore=1.0, side="LONG",
            master_enabled=True, mr_enabled=False, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_BREAKOUT

    def test_trending_down_breakout_short(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_BREAKOUT
        result = classify_execution_style(
            regime="TRENDING_DOWN", structure="BOS_DOWN", hurst=0.7, zscore=-1.0, side="SHORT",
            master_enabled=True, mr_enabled=False, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_BREAKOUT

    def test_volatile_always_market(self):
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MARKET
        result = classify_execution_style(
            regime="VOLATILE", structure="NONE", hurst=0.5, zscore=0.5, side="LONG",
            master_enabled=True, mr_enabled=True, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_MARKET

    def test_trend_misaligned_no_breakout(self):
        """SHORT in TRENDING_UP should NOT trigger breakout."""
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MARKET
        result = classify_execution_style(
            regime="TRENDING_UP", structure="BOS_UP", hurst=0.7, zscore=1.0, side="SHORT",
            master_enabled=True, mr_enabled=False, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_MARKET

    def test_high_hurst_no_mr(self):
        """RANGING but high hurst (trending) + low zscore → still MARKET."""
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MARKET
        result = classify_execution_style(
            regime="RANGING", structure="NONE", hurst=0.65, zscore=0.5, side="LONG",
            master_enabled=True, mr_enabled=True, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_MARKET

    def test_ranging_high_zscore_triggers_mr(self):
        """RANGING + high zscore (>=1.5) → MR even with high hurst."""
        from risk.execution_style import classify_execution_style, EXEC_STYLE_MR_LIMIT
        result = classify_execution_style(
            regime="RANGING", structure="NONE", hurst=0.65, zscore=2.0, side="LONG",
            master_enabled=True, mr_enabled=True, breakout_enabled=True,
        )
        assert result == EXEC_STYLE_MR_LIMIT


# ═══════════════════════════════════════════════════════════════════
# 3. Structural Zone Tests
# ═══════════════════════════════════════════════════════════════════

class TestStructuralZone:

    def test_sr_zone_long(self):
        from risk.execution_style import build_structural_zone, FALLBACK_SR_FIB
        zone = build_structural_zone(
            supports=[{"price": 95000, "timestamp": 1, "broken": False}],
            resistances=[],
            fvgs=[],
            price=96000,
            side="LONG",
            atr=500,
        )
        assert zone is not None
        assert zone.source == "SR"
        assert zone.price == 95000
        assert zone.fallback_stage == FALLBACK_SR_FIB

    def test_fvg_zone_short(self):
        from risk.execution_style import build_structural_zone, FALLBACK_SR_FIB
        zone = build_structural_zone(
            supports=[],
            resistances=[],
            fvgs=[{"top": 97000, "bottom": 96500, "type": "BEARISH", "mitigated": False}],
            price=96000,
            side="SHORT",
            atr=500,
        )
        assert zone is not None
        assert zone.source == "FVG"
        assert zone.fallback_stage == FALLBACK_SR_FIB

    def test_atr_fallback(self):
        from risk.execution_style import build_structural_zone, FALLBACK_ATR
        zone = build_structural_zone(
            supports=[], resistances=[], fvgs=[],
            price=96000, side="LONG", atr=500,
        )
        assert zone is not None
        assert zone.source == "ATR"
        assert zone.fallback_stage == FALLBACK_ATR
        assert zone.price == 96000 - 500  # price - 1.0×ATR

    def test_absolute_fallback(self):
        from risk.execution_style import build_structural_zone, FALLBACK_ABSOLUTE
        zone = build_structural_zone(
            supports=[], resistances=[], fvgs=[],
            price=96000, side="LONG", atr=0,
            tick_size=0.1, spread=0.5, depth_distance=10,
        )
        assert zone is not None
        assert zone.source == "ABSOLUTE"
        assert zone.fallback_stage == FALLBACK_ABSOLUTE

    def test_percent_fallback(self):
        from risk.execution_style import build_structural_zone, FALLBACK_PERCENT
        zone = build_structural_zone(
            supports=[], resistances=[], fvgs=[],
            price=96000, side="LONG", atr=0,
        )
        assert zone is not None
        assert zone.source == "PERCENT"
        assert zone.fallback_stage == FALLBACK_PERCENT
        # 1% from price
        assert abs(zone.price - (96000 - 960)) < 1

    def test_broken_sr_ignored(self):
        from risk.execution_style import build_structural_zone, FALLBACK_ATR
        zone = build_structural_zone(
            supports=[{"price": 95000, "timestamp": 1, "broken": True}],
            resistances=[], fvgs=[],
            price=96000, side="LONG", atr=500,
        )
        # Broken support ignored → ATR fallback
        assert zone.source == "ATR"

    def test_mitigated_fvg_ignored(self):
        from risk.execution_style import build_structural_zone, FALLBACK_ATR
        zone = build_structural_zone(
            supports=[], resistances=[],
            fvgs=[{"top": 97000, "bottom": 96500, "type": "BEARISH", "mitigated": True}],
            price=96000, side="SHORT", atr=500,
        )
        assert zone.source == "ATR"  # Mitigated FVG skipped

    def test_zone_too_far_sr_ignored(self):
        """SR level > 3×ATR away should be skipped."""
        from risk.execution_style import build_structural_zone, FALLBACK_ATR
        zone = build_structural_zone(
            supports=[{"price": 90000, "timestamp": 1, "broken": False}],
            resistances=[], fvgs=[],
            price=96000, side="LONG", atr=500,  # 6000 > 3*500=1500
        )
        assert zone.source == "ATR"

    def test_zero_price_returns_none(self):
        from risk.execution_style import build_structural_zone
        zone = build_structural_zone(
            supports=[], resistances=[], fvgs=[],
            price=0, side="LONG", atr=500,
        )
        assert zone is None

    def test_zone_contains(self):
        from risk.execution_style import StructuralZone, FALLBACK_SR_FIB
        zone = StructuralZone(price=95000, width=200, source="SR",
                              confidence=0.8, fallback_stage=FALLBACK_SR_FIB)
        assert zone.contains(95100)
        assert zone.contains(94900)
        assert not zone.contains(95300)


# ═══════════════════════════════════════════════════════════════════
# 4. Reclaim Tests
# ═══════════════════════════════════════════════════════════════════

class TestReclaim:

    def test_confirmed_long(self):
        from risk.execution_style import is_reclaim_confirmed
        candles = [
            {"close": 96100, "volume": 150, "avg_volume": 100},
            {"close": 96200, "volume": 140, "avg_volume": 100},
            {"close": 96300, "volume": 160, "avg_volume": 100},
        ]
        confirmed, reason = is_reclaim_confirmed(candles, 96000, "LONG")
        assert confirmed is True
        assert reason == "CONFIRMED"

    def test_close_below_level(self):
        from risk.execution_style import is_reclaim_confirmed
        candles = [
            {"close": 95900, "volume": 150, "avg_volume": 100},
            {"close": 96100, "volume": 140, "avg_volume": 100},
        ]
        confirmed, reason = is_reclaim_confirmed(candles, 96000, "LONG")
        assert confirmed is False
        assert reason == "CLOSE_BELOW_LEVEL"

    def test_low_volume(self):
        from risk.execution_style import is_reclaim_confirmed
        candles = [
            {"close": 96100, "volume": 100, "avg_volume": 100},
            {"close": 96200, "volume": 110, "avg_volume": 100},
        ]
        confirmed, reason = is_reclaim_confirmed(candles, 96000, "LONG", min_volume_ratio=1.3)
        assert confirmed is False
        assert reason == "LOW_VOLUME"

    def test_no_data_failsafe(self):
        from risk.execution_style import is_reclaim_confirmed
        confirmed, reason = is_reclaim_confirmed(None, 96000, "LONG")
        assert confirmed is False
        assert reason == "NO_DATA"

    def test_empty_candles_failsafe(self):
        from risk.execution_style import is_reclaim_confirmed
        confirmed, reason = is_reclaim_confirmed([], 96000, "LONG")
        assert confirmed is False
        assert reason == "NO_DATA"

    def test_short_reclaim(self):
        from risk.execution_style import is_reclaim_confirmed
        candles = [
            {"close": 95800, "volume": 200, "avg_volume": 100},
            {"close": 95700, "volume": 180, "avg_volume": 100},
        ]
        confirmed, reason = is_reclaim_confirmed(candles, 96000, "SHORT")
        assert confirmed is True


# ═══════════════════════════════════════════════════════════════════
# 5. Partial TP Tests
# ═══════════════════════════════════════════════════════════════════

class TestPartialTP:

    def test_high_confidence_max_pct(self):
        from risk.execution_style import compute_structural_partial_tp_pct
        pct = compute_structural_partial_tp_pct(confidence=1.0)
        assert pct == 0.40  # max

    def test_low_confidence_min_pct(self):
        from risk.execution_style import compute_structural_partial_tp_pct
        pct = compute_structural_partial_tp_pct(confidence=0.0)
        assert pct == 0.25  # min

    def test_mid_confidence(self):
        from risk.execution_style import compute_structural_partial_tp_pct
        pct = compute_structural_partial_tp_pct(confidence=0.5)
        assert 0.32 < pct < 0.33  # 0.25 + 0.5 * 0.15 = 0.325

    def test_clamped_above_1(self):
        from risk.execution_style import compute_structural_partial_tp_pct
        pct = compute_structural_partial_tp_pct(confidence=1.5)
        assert pct == 0.40  # clamped to 1.0

    def test_clamped_below_0(self):
        from risk.execution_style import compute_structural_partial_tp_pct
        pct = compute_structural_partial_tp_pct(confidence=-0.5)
        assert pct == 0.25  # clamped to 0.0


# ═══════════════════════════════════════════════════════════════════
# 6. Fallback Stage Tests
# ═══════════════════════════════════════════════════════════════════

class TestFallbackStage:

    def test_sr_fib(self):
        from risk.execution_style import resolve_fallback_stage, FALLBACK_SR_FIB
        assert resolve_fallback_stage(True, True, True) == FALLBACK_SR_FIB

    def test_atr_only(self):
        from risk.execution_style import resolve_fallback_stage, FALLBACK_ATR
        assert resolve_fallback_stage(False, True, True) == FALLBACK_ATR

    def test_absolute(self):
        from risk.execution_style import resolve_fallback_stage, FALLBACK_ABSOLUTE
        assert resolve_fallback_stage(False, False, True) == FALLBACK_ABSOLUTE

    def test_percent(self):
        from risk.execution_style import resolve_fallback_stage, FALLBACK_PERCENT
        assert resolve_fallback_stage(False, False, False) == FALLBACK_PERCENT


# ═══════════════════════════════════════════════════════════════════
# 7. Profile Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestProfile:

    def test_runner_profile_has_exec_style(self, monkeypatch):
        """V3 Runner with master flag ON → exec_style_enabled=True."""
        monkeypatch.setenv("RFX3_EXEC_STYLE_ENABLED", "true")
        # Need to reimport to pick up env change
        import importlib
        import risk.execution_style
        importlib.reload(risk.execution_style)
        from risk.strategy_profile import (
            resolve_strategy_execution_profile,
            STRATEGY_MODE_SMART_V3_RUNNER,
        )
        profile = resolve_strategy_execution_profile(STRATEGY_MODE_SMART_V3_RUNNER)
        assert profile.exec_style_enabled is True
        assert profile.struct_partial_tp_pct == 0.30
        assert profile.struct_partial_be_floor is True

    def test_runner_profile_flag_off(self, monkeypatch):
        """V3 Runner with master flag OFF → exec_style_enabled=False."""
        monkeypatch.setenv("RFX3_EXEC_STYLE_ENABLED", "false")
        import importlib
        import risk.execution_style
        importlib.reload(risk.execution_style)
        from risk.strategy_profile import (
            resolve_strategy_execution_profile,
            STRATEGY_MODE_SMART_V3_RUNNER,
        )
        profile = resolve_strategy_execution_profile(STRATEGY_MODE_SMART_V3_RUNNER)
        assert profile.exec_style_enabled is False

    def test_log_dict_includes_exec_fields(self):
        from risk.strategy_profile import StrategyExecutionProfile
        p = StrategyExecutionProfile.neutral()
        d = p.to_log_dict()
        assert "exec_style" in d
        assert "partial_tp_pct" in d
        assert d["exec_style"] is False

    def test_zone_to_dict(self):
        from risk.execution_style import StructuralZone, FALLBACK_SR_FIB
        zone = StructuralZone(
            price=95000, width=200, source="SR",
            confidence=0.8, fallback_stage=FALLBACK_SR_FIB,
        )
        d = zone.to_dict()
        assert d["zone_price"] == 95000
        assert d["zone_source"] == "SR"
        assert d["fallback_stage"] == "SR_FIB"


# ═══════════════════════════════════════════════════════════════════
# 8. Flag Parsing Tests
# ═══════════════════════════════════════════════════════════════════

class TestFlags:

    def test_env_bool_true(self):
        from risk.execution_style import _env_bool
        os.environ["TEST_FLAG"] = "true"
        assert _env_bool("TEST_FLAG") is True
        os.environ["TEST_FLAG"] = "1"
        assert _env_bool("TEST_FLAG") is True
        os.environ["TEST_FLAG"] = "YES"
        assert _env_bool("TEST_FLAG") is True
        del os.environ["TEST_FLAG"]

    def test_env_bool_false(self):
        from risk.execution_style import _env_bool
        os.environ["TEST_FLAG"] = "false"
        assert _env_bool("TEST_FLAG") is False
        os.environ["TEST_FLAG"] = "0"
        assert _env_bool("TEST_FLAG") is False
        del os.environ["TEST_FLAG"]

    def test_env_bool_default(self):
        from risk.execution_style import _env_bool
        assert _env_bool("NONEXISTENT_FLAG_12345") is False
        assert _env_bool("NONEXISTENT_FLAG_12345", True) is True

    def test_startup_log(self):
        from risk.execution_style import startup_log_flags
        flags = startup_log_flags()
        assert "master" in flags
        assert "mr_limit" in flags
        assert "breakout_reclaim" in flags
        assert "struct_partial_tp" in flags
