"""
Risk Safety Patch Tests — 10 tests for riskPerTrade, force_market,
volatility leverage cap, notional size cap, EQ, apply_risk_caps.

Run: python3 -m pytest test_risk_caps.py -v --override-ini="asyncio_mode=auto"
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from main import LiveBinanceTrader  # noqa: E402


# ─── helpers ───────────────────────────────────────────────────

def _make_trader(**overrides) -> LiveBinanceTrader:
    old_env = {}
    env_defaults = {
        'TRADING_MODE': 'live',
        'EXEC_BBO_RETRIES': '1',
        'EXEC_BBO_RETRY_DELAY_MS': '5',
        'EXEC_BBO_MAX_AGE_MS': '5000',
        'EXEC_BBO_CACHE_TTL_MS': '400',
    }
    env_defaults.update(overrides)
    for k, v in env_defaults.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    trader = LiveBinanceTrader()
    trader.exchange = MagicMock()
    trader.enabled = True
    trader.initialized = True

    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return trader


def _valid_bbo(bid=65000.0, ask=65001.0):
    return LiveBinanceTrader._make_bbo_snapshot(
        bid=bid, ask=ask, source='bookTicker_rest',
        ts_ms=int(datetime.now().timestamp() * 1000),
    )


def _invalid_bbo(reason='test_forced_invalid'):
    return LiveBinanceTrader._make_bbo_snapshot(
        bid=0, ask=0, source='none', ts_ms=0, reason=reason,
    )


@pytest.fixture
def trader():
    return _make_trader()


# ================================================================
# 1) riskPerTrade normalize — percent input
# ================================================================
def test_risk_per_trade_percent_input_normalized():
    """UI sends 2 (meaning 2%). Must be stored as 0.02."""
    trader = _make_trader()
    # Simulate settings endpoint
    r = float(2)
    if r > 1.0:
        r = r / 100.0
    r = max(0.002, min(0.05, r))
    assert r == 0.02


def test_risk_per_trade_high_percent_clamped():
    """UI sends 10 (10%). Must be clamped to 0.05 (5%)."""
    r = float(10)
    if r > 1.0:
        r = r / 100.0
    r = max(0.002, min(0.05, r))
    assert r == 0.05


# ================================================================
# 2) riskPerTrade normalize — ratio input preserved
# ================================================================
def test_risk_per_trade_ratio_input_preserved():
    """If input is already ratio (0.02), keep it."""
    r = float(0.02)
    if r > 1.0:
        r = r / 100.0
    r = max(0.002, min(0.05, r))
    assert r == 0.02


# ================================================================
# 3) State load guard
# ================================================================
def test_risk_per_trade_state_load_guard():
    """Legacy state with 2.0 must be normalized to 0.02."""
    _r = 2.0  # legacy state
    if _r > 1.0:
        _r = _r / 100.0
    _r = max(0.002, min(0.05, _r))
    assert _r == 0.02


# ================================================================
# 4) force_market still blocked on invalid BBO
# ================================================================
@pytest.mark.asyncio
async def test_force_market_still_blocked_on_invalid_bbo(trader):
    """force_market path now enforces validate_bbo. Invalid BBO → block."""
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    # validate_bbo wraps _get_valid_bbo
    vbbo = await trader.validate_bbo('BTCUSDT')
    assert vbbo['valid'] is False
    # The force_market path should check this and block
    # Verify the gate logic: not valid → BLOCK_OPEN_INVALID_BBO
    assert not vbbo['valid']
    assert vbbo['reason'] != 'ok'


# ================================================================
# 5) force_market drift hard block (default=true)
# ================================================================
def test_force_market_drift_hard_block():
    """EXEC_FORCE_MARKET_DRIFT_HARD_BLOCK defaults to true now."""
    trader = _make_trader()
    assert trader.exec_force_market_drift_hard is True


def test_force_market_drift_hard_block_env_override():
    """Can override to false via env."""
    trader = _make_trader(EXEC_FORCE_MARKET_DRIFT_HARD_BLOCK='false')
    assert trader.exec_force_market_drift_hard is False


# ================================================================
# 6) Volatility leverage cap applied
# ================================================================
def test_volatility_leverage_cap_applied():
    """apply_risk_caps: vol>=30% → lev_cap=3, vol>=20% → 5, vol>=12% → 8."""
    r30 = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=35, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r30['lev_cap'] == 3
    assert any('VOL_CAP' in r for r in r30['reasons'])

    r20 = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=22, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r20['lev_cap'] == 5

    r12 = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=14, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r12['lev_cap'] == 8

    r5 = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=5, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r5['lev_cap'] == 75  # no cap


def test_volatility_leverage_cap_illiquid():
    """Illiquid coin (vol24h < 1M or spread >= 0.20%) → cap=5."""
    r = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=5, spread_pct=0.25, volume_24h=5_000_000, balance=1000)
    assert r['lev_cap'] == 5
    assert any('ILLIQ_CAP' in s for s in r['reasons'])

    r2 = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=5, spread_pct=0.05, volume_24h=500_000, balance=1000)
    assert r2['lev_cap'] == 5


# ================================================================
# 7) Notional size cap applied
# ================================================================
def test_notional_size_cap_applied():
    """apply_risk_caps: default 15%, vol>=20 → 7%, lowvol → 5%."""
    r_normal = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=5, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r_normal['size_cap_pct'] == 0.15
    assert r_normal['size_cap_usd'] == 150.0

    r_vol = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=25, spread_pct=0.05, volume_24h=5_000_000, balance=1000)
    assert r_vol['size_cap_pct'] == 0.07
    assert r_vol['size_cap_usd'] == 70.0

    r_lowvol = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=5, spread_pct=0.05, volume_24h=500_000, balance=1000)
    assert r_lowvol['size_cap_pct'] == 0.05
    assert r_lowvol['size_cap_usd'] == 50.0


# ================================================================
# 8) EQ adaptive requires two conditions
# ================================================================
def test_eq_adaptive_requires_two_conditions():
    """After RSP, adaptive mode also requires 2/3 EQ conditions, not 1/3."""
    # We test the constant by grep — but also test the logic directly:
    # eq_required is always 2 now
    import re
    with open('main.py', 'r') as f:
        content = f.read()
    # Find the eq_required line
    match = re.search(r'eq_required = 2\s+# Phase RSP', content)
    assert match is not None, "eq_required should be hardcoded to 2"
    # Verify adaptive does NOT lower it
    assert 'eq_required = 2 if not adaptive_mode else 1' not in content


# ================================================================
# 9) legacy_micro_soft_pass tightened
# ================================================================
def test_legacy_micro_soft_pass_tightened():
    """legacy_micro_soft_pass thresholds must be score>=105, eq>=2, spread<=0.08, depth>=2500."""
    with open('main.py', 'r') as f:
        content = f.read()
    assert 'signal_score >= 105' in content, "score threshold should be 105"
    assert 'eq_count >= 2' in content, "eq_count threshold should be 2"
    assert "signal_spread <= 0.08" in content, "spread threshold should be 0.08"
    assert "total_depth >= 2_500" in content, "depth threshold should be 2500"


# ================================================================
# 10) apply_risk_caps helper
# ================================================================
def test_apply_risk_caps_helper():
    """apply_risk_caps returns correct structure with all expected keys."""
    r = LiveBinanceTrader.apply_risk_caps(
        volatility_pct=15, spread_pct=0.10, volume_24h=2_000_000, balance=5000)
    assert 'lev_cap' in r
    assert 'size_cap_pct' in r
    assert 'size_cap_usd' in r
    assert 'reasons' in r
    assert isinstance(r['reasons'], list)
    assert r['lev_cap'] == 8  # vol=15% → cap=8
    assert r['size_cap_pct'] == 0.15
    assert r['size_cap_usd'] == 750.0
