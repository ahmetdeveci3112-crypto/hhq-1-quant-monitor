"""
RFX-1A.1: Integration tests — verify adapter wiring in main.py.

Tests that main.py functions correctly delegate to risk modules
when RFX flags are ON, and that parity mode produces identical results.

Run: python3 -m pytest test_rfx_integration.py -v --override-ini="asyncio_mode=auto"
"""
import os
import sys
import math
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

# Import main.py functions directly (as existing tests do)
from main import (
    compute_sl_tp_levels,
    compute_breakeven_buffer_pct,
    check_emergency_sl_static,
)


# ═══════════════════════════════════════════════════════════════════
# Test 1: compute_sl_tp_levels adapter — flag ON+parity vs flag OFF
# ═══════════════════════════════════════════════════════════════════

class TestSLTPAdapter:
    """main.compute_sl_tp_levels flag-on/off parity test."""

    BASE_ARGS = dict(
        entry_price=65000.0,
        atr=1200.0,
        side='LONG',
        leverage=10,
        symbol='BTCUSDT',
        adjusted_sl_atr=1.5,
        adjusted_tp_atr=3.0,
        adjusted_trail_act_atr=2.0,
        adjusted_trail_dist_atr=0.5,
        spread_pct=0.05,
    )

    def test_flag_off_legacy(self):
        """Flag OFF: legacy path should work normally."""
        with patch('main.RFX_SL_TP_V2', False):
            result = compute_sl_tp_levels(**self.BASE_ARGS)
        assert 'sl' in result
        assert 'tp' in result
        assert result['sl'] > 0
        assert result['tp'] > result['sl']

    def test_flag_on_parity_mode_matches_legacy(self):
        """Flag ON + parity_mode=True: should produce SAME results as legacy."""
        with patch('main.RFX_SL_TP_V2', False):
            legacy = compute_sl_tp_levels(**self.BASE_ARGS)
        
        with patch('main.RFX_SL_TP_V2', True), \
             patch('main.RFX_PARITY_MODE', True), \
             patch('main.RFX_RISK_PROFILE', 'BALANCED'):
            v2 = compute_sl_tp_levels(**self.BASE_ARGS)
        
        # Parity mode should produce identical SL/TP
        # Note: tick_size may differ (legacy calls get_tick_size internally)
        # but with parity floors (30%/5%), distances should match pre-snap
        assert v2['meta'].get('version') == 'v2' or v2['meta'].get('parity_mode') is True
        # Both should have reasonable values
        assert abs(legacy['sl'] - v2['sl']) / legacy['sl'] < 0.001, \
            f"SL divergence: legacy={legacy['sl']}, v2={v2['sl']}"
        assert abs(legacy['tp'] - v2['tp']) / legacy['tp'] < 0.001, \
            f"TP divergence: legacy={legacy['tp']}, v2={v2['tp']}"

    def test_flag_on_short_parity(self):
        """Flag ON parity for SHORT side."""
        args = {**self.BASE_ARGS, 'side': 'SHORT'}
        with patch('main.RFX_SL_TP_V2', False):
            legacy = compute_sl_tp_levels(**args)
        with patch('main.RFX_SL_TP_V2', True), \
             patch('main.RFX_PARITY_MODE', True), \
             patch('main.RFX_RISK_PROFILE', 'BALANCED'):
            v2 = compute_sl_tp_levels(**args)
        assert abs(legacy['sl'] - v2['sl']) / legacy['sl'] < 0.001
        assert abs(legacy['tp'] - v2['tp']) / legacy['tp'] < 0.001

    def test_flag_on_no_parity_uses_profile(self):
        """Flag ON + parity OFF: should use risk profile params."""
        with patch('main.RFX_SL_TP_V2', True), \
             patch('main.RFX_PARITY_MODE', False), \
             patch('main.RFX_RISK_PROFILE', 'ULTRA_AGGRESSIVE'):
            result = compute_sl_tp_levels(**self.BASE_ARGS)
        # ULTRA profile has wider SL floor (100% ROI)
        assert result['meta'].get('version') == 'v2'
        assert result['meta'].get('sl_roi_floor_used', 0) >= 80  # ULTRA or AGGRESSIVE


# ═══════════════════════════════════════════════════════════════════
# Test 2: check_emergency_sl_static adapter — parity with legacy
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyStaticAdapter:
    """main.check_emergency_sl_static flag-on parity test.
    
    CRITICAL: Must use 12% ROI trail-based logic (NOT 50% ROI entry-based).
    """

    def test_flag_off_legacy(self):
        """Flag OFF: legacy path returns False for normal position."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        with patch('main.RFX_EMERGENCY_V2', False):
            result = check_emergency_sl_static(pos, 64000.0, 63000.0)
        assert result is False

    def test_flag_on_parity_no_trigger(self):
        """Flag ON: same inputs that don't trigger legacy shouldn't trigger v1."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        with patch('main.RFX_EMERGENCY_V2', False):
            legacy_result = check_emergency_sl_static(pos, 64000.0, 63000.0)
        with patch('main.RFX_EMERGENCY_V2', True):
            rfx_result = check_emergency_sl_static(pos, 64000.0, 63000.0)
        assert legacy_result == rfx_result, \
            f"Parity broken: legacy={legacy_result}, rfx={rfx_result}"

    def test_flag_on_parity_trigger(self):
        """Flag ON: same inputs that trigger legacy should trigger v1."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        # Price crashed WAY below trailing stop
        with patch('main.RFX_EMERGENCY_V2', False):
            legacy_result = check_emergency_sl_static(pos, 59000.0, 63000.0)
        with patch('main.RFX_EMERGENCY_V2', True):
            rfx_result = check_emergency_sl_static(pos, 59000.0, 63000.0)
        assert legacy_result == rfx_result, \
            f"Parity broken on trigger: legacy={legacy_result}, rfx={rfx_result}"

    def test_flag_on_parity_short(self):
        """Flag ON parity for SHORT position."""
        pos = {'entryPrice': 65000, 'side': 'SHORT', 'leverage': 10}
        # Price rose above trailing stop → emergency
        with patch('main.RFX_EMERGENCY_V2', False):
            legacy = check_emergency_sl_static(pos, 72000.0, 67000.0)
        with patch('main.RFX_EMERGENCY_V2', True):
            rfx = check_emergency_sl_static(pos, 72000.0, 67000.0)
        assert legacy == rfx

    def test_flag_on_edge_case_zero_trailing(self):
        """Edge case: trailing_stop=0 should return False (both paths)."""
        pos = {'entryPrice': 65000, 'side': 'LONG', 'leverage': 10}
        with patch('main.RFX_EMERGENCY_V2', False):
            legacy = check_emergency_sl_static(pos, 60000.0, 0)
        with patch('main.RFX_EMERGENCY_V2', True):
            rfx = check_emergency_sl_static(pos, 60000.0, 0)
        assert legacy == rfx == False


# ═══════════════════════════════════════════════════════════════════
# Test 3: compute_breakeven_buffer_pct adapter — smoke test
# ═══════════════════════════════════════════════════════════════════

class TestBreakevenAdapter:
    """main.compute_breakeven_buffer_pct adapter smoke test."""

    def test_flag_off_legacy(self):
        """Flag OFF: legacy path returns reasonable buffer."""
        with patch('main.RFX_LIQUIDITY_NATIVE', False):
            buf = compute_breakeven_buffer_pct(spread_pct=0.05, spread_level="LOW")
        assert 0.0012 <= buf <= 0.008

    def test_flag_on_returns_buffer(self):
        """Flag ON: rfx path should also return reasonable buffer."""
        with patch('main.RFX_LIQUIDITY_NATIVE', True):
            buf = compute_breakeven_buffer_pct(spread_pct=0.05, spread_level="LOW")
        assert 0.0012 <= buf <= 0.008

    def test_flag_on_parity(self):
        """Flag ON should produce same output as flag OFF (extracted same logic)."""
        args = dict(spread_pct=0.10, expected_slippage_pct=0.03, 
                    is_live=True, spread_level="HIGH")
        with patch('main.RFX_LIQUIDITY_NATIVE', False):
            legacy = compute_breakeven_buffer_pct(**args)
        with patch('main.RFX_LIQUIDITY_NATIVE', True):
            rfx = compute_breakeven_buffer_pct(**args)
        assert abs(legacy - rfx) < 0.0001, \
            f"Breakeven parity broken: legacy={legacy}, rfx={rfx}"
